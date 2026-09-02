import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from src.anubis.utils.billing.system_prompt_estimate_cache import (
    record_system_prompt_token_estimate,
)
from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.classes.ImageDescriptionClass import ImageDescriptionClass
from src.anubis.utils.context import AssistantContext, GlobalContext, UserContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.store_cache import aget_through_cache
from src.anubis.utils.utility import (
    merge_dedup_threshold_documents,
    reduce_docs,
)

logger = logging.getLogger(__name__)

# ``nodes.py`` → ``utils`` → ``anubis`` → ``src`` → repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DEV_SYSTEM_PROMPT_PATH = _PROJECT_ROOT / "system_prompt.txt"


def _global_context_from_runtime(runtime) -> GlobalContext:
    """Resolve ``GlobalContext`` from a LangGraph or tool runtime."""
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, GlobalContext):
        return ctx
    return GlobalContext()


def _write_dev_system_prompt(system_message_str: str, runtime) -> None:
    """Dump the built system prompt when ``DEV=TRUE`` (dev-only debugging aid)."""
    context = _global_context_from_runtime(runtime)
    logger.info(f"context.dev: {context.dev}")
    if context.dev.upper() == "TRUE":
        logger.info("context.dev == TRUE: Writing dev system prompt")
    else:
        logger.info("context.dev == FALSE: system prompt is not being written")

    if context.dev.upper() != "TRUE":
        return
    try:
        _DEV_SYSTEM_PROMPT_PATH.write_text(system_message_str, encoding="utf-8")
    except OSError as write_error:
        # This dump exists to be read by a developer; it is not part of
        # answering the user. Letting it raise took down the whole turn — the
        # run died mid-stream and the client saw a reply that simply stopped.
        # A container writing through a bind mount hits this routinely: the
        # file is owned by another user on the host, and the write is refused.
        logger.warning(
            "Could not write the dev system prompt to %s: %s",
            _DEV_SYSTEM_PROMPT_PATH,
            write_error,
        )
        return
    logger.info("dev system prompt written to: %s", _DEV_SYSTEM_PROMPT_PATH)


def _resolve_user_timezone(tz_name: str | None):
    """Resolve a client-supplied IANA timezone name to a ``tzinfo``.

    The frontend sends the browser's IANA zone (e.g. ``"America/New_York"`` from
    ``st.context.timezone``) so the system clock injected into the prompt reflects
    the user's local time regardless of where the server runs. Falls back to UTC
    when the value is missing or not a recognized zone.
    """
    if not tz_name:
        return UTC
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown user timezone %r; falling back to UTC", tz_name)
        return UTC


def _image_url_from_content_block(block: dict) -> str | None:
    """Resolve image URL from LangChain or OpenAI-style multimodal blocks."""
    if not isinstance(block, dict):
        return None
    t = block.get("type")
    if t == "input_image":
        u = block.get("image_url")
        return u if isinstance(u, str) else None
    if t == "image_url":
        iu = block.get("image_url")
        if isinstance(iu, dict):
            u = iu.get("url")
            return u if isinstance(u, str) else None
        if isinstance(iu, str):
            return iu
    return None


def _text_from_content_block(block: dict) -> str | None:
    if not isinstance(block, dict):
        return None
    t = block.get("type")
    if t in ("text", "input_text"):
        tx = block.get("text")
        return tx if isinstance(tx, str) else None
    return None


async def resolve_human_message_images(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Replace multimodal HumanMessage (base64 image blocks) with plain text descriptions."""
    msgs = state.get("messages") or []
    if not msgs:
        return {}
    last = msgs[-1]
    if not isinstance(last, HumanMessage):
        return {}

    content = last.content
    if isinstance(content, str) or not isinstance(content, list):
        return {}

    has_image = any(
        _image_url_from_content_block(b) for b in content if isinstance(b, dict)
    )
    if not has_image:
        return {}

    if not last.id:
        logger.warning(
            "resolve_human_message_images: HumanMessage missing id; skipping replacement"
        )
        return {}

    filenames = (last.additional_kwargs or {}).get("image_filenames") or []
    descriptor = ImageDescriptionClass()
    img_index = 0
    text_chunks: list[str] = []
    image_sections: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        url = _image_url_from_content_block(block)
        if url:
            fname = (
                filenames[img_index]
                if img_index < len(filenames)
                else f"image_{img_index + 1}"
            )
            img_index += 1
            try:
                meta = await descriptor.describe(url, fname)
                desc = (meta.get("description") or "").strip()
            except Exception as exc:
                logger.exception("Image describe failed for %s: %s", fname, exc)
                desc = "[Image could not be described.]"
            image_sections.append(f"[{fname}]\n{desc}")
            continue
        tx = _text_from_content_block(block)
        if tx:
            text_chunks.append(tx)

    base_text = "\n\n".join(text_chunks).strip()
    out_parts: list[str] = []
    if base_text:
        out_parts.append(base_text)
    if image_sections:
        out_parts.append("---\nImage descriptions:\n" + "\n\n".join(image_sections))
    final_text = "\n\n".join(out_parts)

    return {
        "messages": [
            RemoveMessage(id=last.id),
            HumanMessage(content=final_text),
        ]
    }


async def _build_consciousness_system_message_update(
    state, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict:
    """Pure helper that rebuilds the avatar's system prompt + identity doc snapshots.

    Returns a dict with the same shape the ``load_consciousness`` node has always
    produced (``system_message`` pinned to a fixed UUID, ``user_identity_documents``,
    ``assistant_identity_documents``). The pinned ID lets ``add_messages`` replace
    rather than append, which is what gives the deep agent middleware a single
    "latest prompt" slot to read from on every LLM call.

    Accepts a generic ``state`` mapping so it can be called from both the outer
    LangGraph node (operates on ``GlobalState``) and the in-agent
    ``load_consciousness`` tool (operates on ``AvatarDeepAgentState``); both
    schemas expose the same keys this helper reads.


    # TODO: REDUCE FP; There are hundreds of non-salient documents being retrieved

    """
    _RETRIEVAL_LIMIT = 10
    _FILTER_SCORE = 0.5

    user_id = state["user_state"]["user_id"]
    assistant_id = state["assistant_state"]["assistant_id"]
    user_is_creator = state.get('user_is_creator', False)

    # Update Name and Description of User and Assistant if provided in the context
    logger.info("conscioussness breakpoint")
    if getattr(runtime, "context"):
        if isinstance(runtime.context.assistant_ctx, AssistantContext):
            assistant_name = getattr(runtime.context.assistant_ctx, "name", None)
            assistant_description = getattr(
                runtime.context.assistant_ctx, "description", None
            )
        else:
            assert type(runtime.context.assistant_ctx) is dict
            assistant_name = runtime.context.assistant_ctx.get("name", None)
            assistant_description = runtime.context.assistant_ctx.get(
                "description", None
            )

        if isinstance(runtime.context.user_ctx, UserContext):
            user_name = getattr(runtime.context.user_ctx, "name", None)
            user_description = getattr(runtime.context.user_ctx, "description", None)
        else:
            assert type(runtime.context.user_ctx) is dict
            user_name = runtime.context.user_ctx.get("name", None)
            user_description = runtime.context.user_ctx.get("description", None)
    else:
        assert type(config.get("assistant_ctx", {}) is dict)
        assistant_name = (
            config.get("configurable", {}).get("assistant_ctx", {}).get("name", None)
        )
        assistant_description = (
            config.get("configurable", {})
            .get("assistant_ctx", {})
            .get("description", None)
        )

        assert type(config.get("user_ctx", {}) is dict)
        user_name = config.get("user_ctx", {}).get("name", None)
        user_description = config.get("user_ctx", {}).get("description", None)

    if assistant_description is not None:
        state["assistant_state"].update(
            {"assistant_description": assistant_description}
        )

    if user_description is not None:
        state["user_state"].update({"user_description": user_description})

    # The identity namespaces are contractually loaded in full every turn (see the
    # READ ME in identity_tools.py). ``asearch`` defaults to limit=10 and, with no
    # query, returns an arbitrary slice — which silently dropped media-ingested
    # identity facts (e.g. an uploaded résumé's education history) from the prompt,
    # producing recall false-negatives. Pass the latest user message as the
    # relevance query and raise the limit to 1000 so every identity document is
    # surfaced (relevance-ranked only if the count ever exceeds the limit).

    """

    QUERY CREATION FOR SALIENT DOCUMENT RETRIEVAL

    """
    # embedding model microsoft/harrier-oss-v1-270m uses instructions in the query for retrieval as trained

    query = state["messages"][-1].content
    if isinstance(query, list):
        _TASK_DESCRIPTION = "Given the query, retrieve information that is salient to the conversation and semantically similar to the query text."
        query = f"Instruct: {_TASK_DESCRIPTION}\nQuery: {query[0]['text']}"

    creator_id = config["configurable"]["assistant_ctx"]["metadata"]["user_id"]

    user_identity_namespace = (assistant_id, user_id, "identity")
    assistant_identity_namespace = (creator_id, assistant_id, "identity")
    assistant_identity_memory_namespace = (
        creator_id,
        assistant_id,
        "identity_memory",
    )
    assistent_reference_image_identity_namespace = (
        creator_id,
        assistant_id,
        "reference_image",
    )
    assistant_memory_namespace = (user_id, assistant_id, "memory")
    # Owner-scoped, NOT conversing-user-scoped: calibrate_ground_truth writes the
    # style profile under (creator_id, assistant_id, "style_profile") and
    # invalidates that same cache key. Building this from ``user_id`` meant the
    # cached entry could never be the one the writer invalidates, and could never
    # hold a hit for anyone but the owner.
    style_profile_namespace = (creator_id, assistant_id, "style_profile")

    """

    CONCURRENT VECTORSTORE RETRIEVAL

    Every store lookup below depends only on the identifiers and the query
    computed above — no lookup depends on another lookup's result — so all the
    lookups are dispatched together through ``asyncio.gather`` and the total
    store latency collapses from the sum of every round-trip to the single
    slowest round-trip.

    """

    async def _skip_fallback_name_search() -> list:
        """Stand-in coroutine for when a fallback name search is unnecessary.

        Returning an empty list keeps the ``asyncio.gather`` result tuple
        positional when the name already arrived via the runtime context.
        """
        return []

    """ POSSIBLE IMPROVEMENT: CREATE A `NAME` NAMESPACE FOR STORAGE AND RETRIEVAL EXPLICITLY: """
    _TASK_DESCRIPTION_USER_NAME = (
        "Given the query, FIND THE ANSWER TO THE QUESTION WHAT IS YOUR NAME?"
    )

    (
        assistant_possible_name,
        user_possible_name,
        user_identity_document_items,
        assistant_identity_document_items,
        retrieved_identity_memories_items,
        reference_image_items,
        retrieved_memories_items,
        direct_quote_items,
        retrieved_knowledge_items,
        analyzed_trait_items,
        style_profile_ITEM,
    ) = await asyncio.gather(
        # Fallback name searches only run when the context did not provide a name
        (
            runtime.store.asearch((user_id, assistant_id, "identity"), query="name")
            if assistant_name is None
            else _skip_fallback_name_search()
        ),
        (
            runtime.store.asearch(
                (assistant_id, user_id, "identity"),
                query=f"Instruct: {_TASK_DESCRIPTION_USER_NAME}\nQuery: {'WHAT IS YOUR NAME?'}",
                limit=_RETRIEVAL_LIMIT,
            )
            if user_name is None
            else _skip_fallback_name_search()
        ),
        runtime.store.asearch(
            user_identity_namespace, query=query, limit=_RETRIEVAL_LIMIT
        ),
        runtime.store.asearch(
            assistant_identity_namespace, query=query, limit=_RETRIEVAL_LIMIT
        ),
        runtime.store.asearch(
            assistant_identity_memory_namespace, query=query, limit=_RETRIEVAL_LIMIT
        ),
        # Cached: the reference image only changes on upload or deletion; the
        # write and delete sites invalidate the cache entry (see store_cache.py).
        aget_through_cache(
            runtime.store, assistent_reference_image_identity_namespace, assistant_id
        ),
        runtime.store.asearch(
            assistant_memory_namespace, query=query, limit=_RETRIEVAL_LIMIT
        ),
        runtime.store.asearch(
            (creator_id, assistant_id, "quote"), query=query, limit=_RETRIEVAL_LIMIT
        ),
        runtime.store.asearch(
            (creator_id, assistant_id, "document"),
            query=query,
            limit=_RETRIEVAL_LIMIT,
        ),
        runtime.store.asearch(
            (creator_id, assistant_id, "analysis"), query=query, limit=_RETRIEVAL_LIMIT
        ),
        # Cached: the style profile only changes on stylometric recalibration;
        # the write and delete sites invalidate the cache entry (see store_cache.py).
        aget_through_cache(runtime.store, style_profile_namespace, "style_profile"),
    )

    if assistant_name is not None:
        state["assistant_state"].update({"assistant_name": assistant_name})
    else:
        if len(assistant_possible_name) > 0:
            assistant_name = (
                getattr(assistant_possible_name[0], "value")
                .get("document", {})
                .get("kwargs", {})
                .get("metadata", {})
                .get("fact", "")
            )
        else:
            assistant_name = ""

    if user_name is not None:
        state["user_state"].update({"user_name": user_name})
    else:
        if len(user_possible_name) > 0 and (
            getattr(user_possible_name[0], "score", 0) > _FILTER_SCORE
        ):
            user_name = (
                getattr(user_possible_name[0], "value")
                .get("document", {})
                .get("kwargs", {})
                .get("metadata", {})
                .get("fact", "")
            )
        else:
            user_name = ""

    """ 
    
    Load User Identity documents (always from store — checkpoint cache is not authoritative). 
    
    INFORMATION ABOUT THE USER SALIENT TO THE CONVERSATION
    
    """

    # Filter the retrieved documents to a salience threshold
    user_identity_document_items = [
        item
        for item in user_identity_document_items
        if item.score and item.score > _FILTER_SCORE
    ]

    # STATEFUL: merge the docs persisted in graph state with this turn's retrieval,
    # de-duplicated by stable id (the fresh store copy wins, so edits are reflected).
    # Identity channels are never salience-pruned — the avatar must not forget identity.
    user_identity = await merge_dedup_threshold_documents(
        state.get("user_identity_documents"),
        user_identity_document_items,
        query,
        apply_threshold=False,
    )

    """ 
    
    Load Assistant Identity documents 
    
    ASSISTANT IDENTITY DOCUMENTS ARE INFORMATION FROM A PRIMARY SOURCE (USES THE QUERY FOR CONVERSATION SALIENCE)
    
    """

    # Filter the retrieved documents to a salience threshold

    # IDENTITY RELATED FACTS ARE NOT FILTERED TO PERSIST FACTS OF THE AVATAR'S IDENTITY.
    # assistant_identity_document_items = [item for item in assistant_identity_document_items if item.score and item.score > _FILTER_SCORE]

    # STATEFUL: merge the persisted assistant identity docs with this turn's retrieval
    # (fresh store copy wins on id collision so in-place edits show new content; docs
    # deleted from the store were pruned from state by the edit/delete tools). Never
    # salience-pruned — the avatar must not forget its own identity.
    assistant_identity = await merge_dedup_threshold_documents(
        state.get("assistant_identity_documents"),
        assistant_identity_document_items,
        query,
        apply_threshold=False,
    )

    """ 

    LEARNED INFORMATION ABOUT THE IDENTITY OF THE AVATAR THROUGH NATURAL LANGUAGE FROM THE USER-CREATOR
    
    PERSISTENT INFORMATION ABOUT THE AVATAR'S IDENTITY

    """

    # Filter the retrieved identity_memories to a salience threshold

    # IDENTITY RELATED FACTS ARE NOT FILTERED TO PERSIST FACTS OF THE AVATAR'S IDENTITY.
    # retrieved_identity_memories_items = [item for item in retrieved_identity_memories_items if item.score and item.score > _FILTER_SCORE]

    # Merged into the same persisted channel: prior state already holds identity_memory
    # docs, so union via the dedup helper instead of a blind extend (fresh copy wins).
    assistant_identity = await merge_dedup_threshold_documents(
        assistant_identity,
        retrieved_identity_memories_items,
        query,
        apply_threshold=False,
    )

    """ 
    
    Always merge assistant reference image (creator namespace), including when identity docs are cached 
    
    DESCRIPTIVE REFERENCE IMAGES ARE PERSISTENT INFORMATION ABOUT THE AVATAR'S IDENTITY

    THE REFERENCE IMAGE LOOKUP IS CACHED (store_cache.py) AND ONLY RE-FETCHED WHEN
    THE REFERENCE IMAGE HAS BEEN UPLOADED OR DELETED SINCE THE CACHED READ.

    """

    reference_image_items_list: list = []
    if reference_image_items is not None:
        if isinstance(reference_image_items, (list, tuple)):
            reference_image_items_list = list(reference_image_items)
        else:
            reference_image_items_list = [reference_image_items]

    reference_image_doc = reduce_docs([], reference_image_items_list)
    assistant_identity = [
        d
        for d in assistant_identity
        if not (getattr(d, "metadata", None) or {}).get("reference_image")
    ]
    assistant_identity.extend(reference_image_doc)

    logger.info("breakpoint")

    """ 
    
    Retrieve memories 
    
    THESE ARE LEARNED MEMORIES SALIENT TO THE CONVERSATION (USES QUERY FOR SEARCH)
    
    """

    # STATEFUL + SALIENCE-PRUNED: episodic memory is the one channel that is
    # re-scored against the current query each turn. Persisted state memories are
    # merged with the fresh retrieval, de-duplicated (fresh copy wins), and every
    # surviving doc must clear the salience threshold — freshly retrieved items
    # reuse their store score, prior-state docs are re-embedded against the query —
    # so stale low-salience memories fall out of state instead of accumulating.
    retrieved_memories = await merge_dedup_threshold_documents(
        state.get("recalled_memory_documents"),
        retrieved_memories_items,
        query,
        apply_threshold=True,
        threshold=_FILTER_SCORE,
    )

    """ 
    
    Retrieve Direct Quotes 
    
    THIS IS WHAT THE AVATAR HAS SAID PRECISELY IN THE PAST (USES QUERY FOR CONVERSATION SALIENCE)
    
    """

    # Few Shot Example of Quotes and Writing style directly from the real-world assistant
    # The QUOTE namespace holds direct quotes from the real-world assistant

    # Filter the direct quotes to a salience threshold
    direct_quote_items = [
        item for item in direct_quote_items if item.score and item.score > _FILTER_SCORE
    ]

    logger.info(f"direct_quote_items: {direct_quote_items}")
    direct_quotes = reduce_docs([], direct_quote_items)

    """ 
    
    Retrieve Documents 
    
    RETRIEVED REFERENCE MATERIAL MUST BE SALIENT TO THE CONVERSATION (USES QUERY FOR THE RETRIEVAL)
    
    """
    # document namespace is reserved for non-quotes that the assistant has access to (bible, menu, reference documentation, etc.)
    logger.info(f"retrieved_knowledge_items: {retrieved_knowledge_items}")
    retrieved_knowledge_items = [
        item
        for item in retrieved_knowledge_items
        if item.score and item.score > _FILTER_SCORE
    ]
    retrieved_knowledge = reduce_docs([], retrieved_knowledge_items)

    """ 
    
    Retrieve Analyzed Latent Traits 
    
    ANALYZED TRAITS MUST BE SALIENT TO THE CONVERSATION (USES QUERY IN RETRIEVAL)
    
    """

    # The analysis namespace holds psycho-analysis findings about the target
    # (beliefs, emotional triggers, relationships, OCEAN, etc.) produced by the
    # process_media_graph analysis stage. Retrieve those relevant to the current
    # conversation by similarity to the user's message.
    logger.info(f"analyzed_trait_items: {analyzed_trait_items}")
    analyzed_traits = reduce_docs([], analyzed_trait_items)

    """ Retrieve Style Profile """
    # Already fetched once, owner-scoped and through the cache, in the concurrent
    # gather above. This used to re-read the same key with an uncached ``aget``,
    # which cost an extra store round-trip on every single turn and left the
    # gather's copy — keyed on the conversing user — permanently unused, seeding
    # the 64-entry LRU with a None entry per distinct visitor to the avatar.
    # style_profile_str will be "" if the style profile does not exist
    style_profile_str = getattr(style_profile_ITEM, "value", {}).get("value", "")

    """ Retrieve Signature Key Phrases """
    # The avatar's auto-discovered signature phrases (built by calibrate_ground_truth
    # from the direct quotes). Stored owner-scoped at
    # (creator_id, assistant_id, "key_phrase_profile") as a JSON list; rendered
    # here into the LLM-legible block injected as the <SIGNATURE PHRASES> section.
    # Empty string when none have been discovered yet.
    import json as _json

    key_phrase_profile_ITEM = await runtime.store.aget(
        (creator_id, assistant_id, "key_phrase_profile"), "key_phrase_profile"
    )
    key_phrase_list_str = getattr(key_phrase_profile_ITEM, "value", {}).get("value", "")
    try:
        key_phrase_list = _json.loads(key_phrase_list_str) if key_phrase_list_str else []
    except (TypeError, ValueError):
        key_phrase_list = []
    # Render-time guard: phrase sets stored before discovery cleaned its corpus
    # contain markup debris ("https t co ...") — never show those to the model.
    # The stored set itself heals on the avatar's next calibration.
    from src.anubis.utils.dataset.key_phrases import phrase_is_well_formed

    key_phrases_str = "\n".join(
        f'- "{phrase}"'
        for phrase in key_phrase_list
        if phrase_is_well_formed(phrase)
    )

    """ Retrieve Emotions """

    # from src.anubis.utils.prompts.psycho_analysis import plutchik_emotional_wheel_analysis_prompt
    # from src.anubis.utils.state import EmotionSummarization

    # if state['current_assistant_emotions'] is None or state['current_assistant_emotions'] == "":
    #     EMOTIONAL_ANALYSIS_PROMPT = plutchik_emotional_wheel_analysis_prompt
    #     emotional_model = init_model(context=runtime.context, response_format=EmotionSummarization)
    #     historical_assistant_emotion_items = await runtime.store.asearch(assistant_identity_namespace, query=["I am feeling", "feeling"])
    #     historical_assistant_emotion_documents = reduce_docs(historical_assistant_emotion_items)
    #     historical_feelings_str = "\n\n".join([document.metadata.get("fact") for document in historical_user_feelings_documents if document.metadata.get("fact", "") != ""])
    #     emotion_summarization = await emotional_model.ainvoke(input = [SystemMessage(content = EMOTIONAL_ANALYSIS_PROMPT), HumanMessage(content=historical_feelings_str)])
    #     current_assistant_emotions = emotion_summarization.emotional_summary

    # # Search user feelings
    # if state['current_user_feelings'] is None or state['current_user_feelings'] == "":
    #     EMOTIONAL_ANALYSIS_PROMPT = plutchik_emotional_wheel_analysis_prompt
    #     emotional_model = init_model(context=runtime.context, response_format=EmotionSummarization)

    #     historical_user_feelings_items = await runtime.store.asearch(user_identity_namespace, query=["I am feeling", "feeling"])
    #     historical_user_feelings_documents = reduce_docs(historical_user_feelings_items)
    #     historical_feelings_str = "\n\n".join([document.metadata.get("fact") for document in historical_user_feelings_documents if document.metadata.get("fact", "") != ""])

    #     historical_user_feelings_items = await runtime.store.asearch(user_id, assistant_id, "memory", query=["I am feeling", "feeling"])
    #     historical_user_feelings_documents = reduce_docs(historical_user_feelings_items)
    #     historical_feelings_str = historical_feelings_str + "\n\n".join([document.metadata.get("fact") for document in historical_user_feelings_documents if document.metadata.get("fact", "") != ""])

    #     emotion_summarization = await emotional_model.ainvoke(input = [SystemMessage(content = EMOTIONAL_ANALYSIS_PROMPT), HumanMessage(content=historical_feelings_str)])

    #     current_user_emotions = emotion_summarization.emotional_summary

    prompt_builder = DynamicPromptBuilder()

    # Localize the injected clock to the querying user's timezone (sent by the
    # client as an IANA name in config["configurable"]["user_timezone"]); UTC if absent.
    user_timezone = config.get("configurable", {}).get("user_timezone")
    system_time = datetime.now(tz=_resolve_user_timezone(user_timezone)).isoformat()

    # assistant_identity = state['assistant_state'].get('assistant_identity', [])
    assistant_name = state["assistant_state"].get("assistant_name", "")

    # user_identity = state['user_state'].get('user_identity', [])
    user_name = state["user_state"].get("user_name", "")

    """ Create System Prompt """

    populated_identity_template = prompt_builder.build_prompt(
        assistant_name=assistant_name,
        assistant_description=assistant_description,
        assistant_identity=assistant_identity,
        retrieved_memories=retrieved_memories,
        retrieved_knowledge=retrieved_knowledge,
        analyzed_traits=analyzed_traits,
        style_profile_str=style_profile_str,
        key_phrases_str=key_phrases_str,
        direct_quotes=direct_quotes,
        user_name=user_name,
        user_description=user_description,
        user_identity=user_identity,
        system_time=system_time,
        user_is_creator=user_is_creator
    )

    logger.info(f"populated_template: {populated_identity_template}")

    # prepend system message
    logger.info(f"state['messages']: {state['messages']}")

    system_message_str = populated_identity_template.messages[0].content

    # Data-analysis capability guidance, mirroring the ``think`` node's tool
    # gates exactly so the prompt never advertises tools the deep agent was
    # not given:
    # - one or more machines bound to THIS avatar → full analysis guidance plus
    #   the roster of connected machine names (the status block deliberately
    #   carries NO server address or directory paths — connection details must
    #   never surface in the conversation, though machine NAMES may, because the
    #   user chose those names);
    # - no connected machine but the conversing user OWNS this avatar → the small
    #   connect-on-request section (the connect_data_server tool is attached);
    # - otherwise (visitor on a shared avatar, or bound elsewhere) → nothing.
    from src.anubis.utils.prompts.system_prompts import (
        DATA_ANALYSIS_CAPABILITY_PROMPT,
        DATA_SERVER_CONNECT_PROMPT,
    )
    from src.anubis.utils.tools.data_analysis import bound_connections_for

    bound_mcp_connections = await bound_connections_for(
        runtime.store, user_id, assistant_id
    )
    assistant_metadata = (
        config.get("configurable", {}).get("assistant_ctx", {}).get("metadata", {})
    )
    avatar_owner_id = assistant_metadata.get("user_id")
    # The MCP capability is exclusive to the user's own personal avatar (owner
    # match AND the is_personal_avatar_of_creator flag) — mirror the think-node
    # gate so the prompt never claims a capability the tools will withhold.
    is_personal_avatar = (
        avatar_owner_id is not None
        and avatar_owner_id == user_id
        and assistant_metadata.get("is_personal_avatar_of_creator") is True
    )
    if bound_mcp_connections and is_personal_avatar:
        system_message_str = system_message_str + DATA_ANALYSIS_CAPABILITY_PROMPT
        # Naming the connected machines in the prompt lets the avatar answer
        # "which of my machines can you see?" without spending a tool call, and
        # keeps the model from inventing a machine name that is not connected.
        connected_machine_names = ", ".join(
            connection.device_label for connection in bound_mcp_connections
        )
        system_message_str += (
            "\n<MCP_CONNECTION_STATUS>\n"
            "The Neural Nexus MCP data server is connected for this avatar on "
            f"the following machines: {connected_machine_names}. "
            "Confirm this plainly when asked, naming the machines. Never reveal "
            "any machine's address, port, transport, or host directory paths in "
            "a reply — the machine names above are the only connection detail "
            "that may appear in a reply.\n"
            "</MCP_CONNECTION_STATUS>\n"
        )
    elif is_personal_avatar:
        system_message_str = system_message_str + DATA_SERVER_CONNECT_PROMPT

    # Mailbox capability, gated identically to the think node's tool gate: the
    # personal avatar, and the accounts bound to it. Same three-way shape as the
    # data-analysis block above — connected mailboxes get the capability
    # guidance plus a status block naming them, an owner with none gets the
    # connect-on-request section, and anyone else gets nothing.
    #
    # A mailbox whose stored password has stopped working is still listed, with
    # its state, so the avatar can tell the owner which mailbox to reconnect
    # instead of silently behaving as though the mailbox did not exist.
    if is_personal_avatar:
        from src.anubis.utils.connected_accounts import (
            STATUS_CONNECTED,
            bound_accounts_for,
        )
        from src.anubis.utils.prompts.system_prompts import (
            CONNECT_MAILBOX_PROMPT,
            MAILBOX_CAPABILITY_PROMPT,
        )

        bound_mailboxes = [
            account
            for account in await bound_accounts_for(runtime.store, user_id, assistant_id)
            if account.get("kind") == "mailbox"
        ]
        if bound_mailboxes:
            system_message_str = system_message_str + MAILBOX_CAPABILITY_PROMPT
            # Naming the mailboxes lets the avatar answer "which of my accounts
            # can you see?" without spending a tool call, and keeps the model
            # from inventing an account name that is not connected. Addresses
            # are included because the owner supplied them and needs to tell two
            # of their own accounts apart; nothing else about the record is.
            mailbox_descriptions = ", ".join(
                f"{account.get('display_label')} ({account.get('account_address')})"
                + (
                    ""
                    if account.get("status") == STATUS_CONNECTED
                    else " — needs to be reconnected"
                )
                for account in bound_mailboxes
            )
            system_message_str += (
                "\n<MAILBOX_STATUS>\n"
                "The following mailboxes are connected for this avatar: "
                f"{mailbox_descriptions}. Confirm this plainly when asked, "
                "naming the mailboxes. Never reveal a mailbox password, app "
                "password, or authentication token in a reply.\n"
                "</MAILBOX_STATUS>\n"
            )
        else:
            system_message_str = system_message_str + CONNECT_MAILBOX_PROMPT

    # Token usage is estimated when token usage occurs: the FINAL system prompt
    # is now assembled (including any data-analysis capability guidance appended
    # just above), so measure the prompt's tokens manually NOW and cache the
    # measurement for the message endpoints' pre-request input-token estimate.
    record_system_prompt_token_estimate(user_id, assistant_id, system_message_str)

    _write_dev_system_prompt(system_message_str, runtime)

    # Replace-snapshots: each persisted doc channel becomes exactly the merged,
    # de-duplicated, (memory-only) salience-pruned set built above — instead of the
    # default append, which would accumulate stale/edited/deleted copies across turns.
    input_update = {
        "user_identity_documents": {"op": "replace", "docs": user_identity},
        "assistant_identity_documents": {"op": "replace", "docs": assistant_identity},
        "recalled_memory_documents": {"op": "replace", "docs": retrieved_memories},
        "system_message": [
            SystemMessage(
                content=system_message_str, id="00000000-0000-0000-0000-0000000000000"
            )
        ],
    }

    return input_update


async def load_consciousness(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Outer-graph node wrapper around :func:`_build_consciousness_system_message_update`.

    Thin shim — all logic lives in the helper so the in-agent
    ``load_consciousness`` tool can share the exact same prompt-building path
    without duplicating the store reads.
    """
    return await _build_consciousness_system_message_update(state, config, runtime)


async def build_system_prompt_text_for_estimation(
    store, assistant_config: dict, latest_message_text: str
) -> str:
    """Build the real system prompt purely to MEASURE the prompt's tokens.

    The message endpoints' pre-request estimate must reflect the actual
    system prompt (identity documents, recalled memories, style profile) —
    not a guessed constant. When the system-prompt estimate cache holds no
    fresh measurement for a (user, avatar) pair, the endpoint calls this
    builder: the exact same prompt-building path as ``load_consciousness``
    (only store reads — no model call), driven by a minimal synthetic state,
    with the measurement recorded into the cache as a side effect of the
    build. The caller treats any failure as fail-closed (HTTP 422), because
    nothing unestimated may reach a model.

    ``assistant_config`` is the endpoint's LangGraph config dict
    (``app_metadata.assistant_config`` merged with the request's updates);
    the builder needs ``configurable.user_id``, ``configurable.assistant_id``,
    ``configurable.assistant_ctx`` (including ``metadata.user_id``, the
    creator), and optionally ``configurable.user_ctx``.
    """
    from types import SimpleNamespace

    configurable = assistant_config.get("configurable") or {}
    user_id = configurable.get("user_id")
    assistant_id = configurable.get("assistant_id")
    if not user_id or not assistant_id:
        raise ValueError(
            "System-prompt estimation requires configurable.user_id and "
            "configurable.assistant_id in the assistant config."
        )

    synthetic_state: dict = {
        "messages": [HumanMessage(content=latest_message_text or "")],
        "user_state": {"user_id": user_id},
        "assistant_state": {"assistant_id": assistant_id},
    }
    # ``_build_consciousness_system_message_update`` only reads
    # ``runtime.store`` and ``runtime.context.assistant_ctx`` /
    # ``runtime.context.user_ctx`` (as ``AssistantContext``/``UserContext``
    # instances or plain dicts), so a lightweight stand-in suffices.
    runtime_stand_in = SimpleNamespace(
        store=store,
        context=SimpleNamespace(
            assistant_ctx=configurable.get("assistant_ctx") or {},
            user_ctx=configurable.get("user_ctx") or {},
        ),
    )
    input_update = await _build_consciousness_system_message_update(
        synthetic_state, assistant_config, runtime_stand_in
    )
    return input_update["system_message"][0].content
