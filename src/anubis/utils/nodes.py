import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
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


def _stream_writer_or_noop():
    """The run's stream writer, or a no-op outside a streamed run."""
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - no active run
        return lambda _payload: None


async def resolve_human_message_images(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Replace multimodal HumanMessage (base64 image blocks) with plain text descriptions.

    The rewritten message keeps the original ``id`` and ``additional_kwargs``
    (minus the consumed ``image_filenames``), so an ambient observation stays
    hidden and tagged after the images are gone. An ambient observation is
    described with ``DESCRIBE_AMBIENT_IMAGE_PROMPT`` (what the person is doing,
    what is on the screen) and each section is labelled by source
    (``webcam`` / ``screen``) under an ``[AMBIENT_OBSERVATION ...]`` header
    the triage node reads. Every description's usage is folded into the
    ``image_model_*`` state channels and emitted as an
    ``image_description_usage`` stream event so the API can meter the call.

    A ``microphone`` source is passed through as a placeholder line: audio
    observations ride the same call, and transcribing them here is the seam
    for that follow-up.
    """
    from src.anubis.utils.ambient.observations import (
        ambient_details,
        is_ambient_observation,
        observation_header,
    )

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

    additional_kwargs = dict(last.additional_kwargs or {})
    filenames = additional_kwargs.pop("image_filenames", None) or []
    ambient = ambient_details(last) if is_ambient_observation(last) else None
    sources = list((ambient or {}).get("sources") or [])
    if ambient is not None:
        from src.anubis.utils.schema import DESCRIBE_AMBIENT_IMAGE_PROMPT

        descriptor = ImageDescriptionClass(system_prompt=DESCRIBE_AMBIENT_IMAGE_PROMPT)
    else:
        descriptor = ImageDescriptionClass()
    writer = _stream_writer_or_noop()
    img_index = 0
    text_chunks: list[str] = []
    image_sections: list[str] = []
    usage_calls = 0
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    usage_total_cost = 0.0
    usage_latencies: list[float] = []

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
            label = (
                sources[img_index]
                if ambient is not None and img_index < len(sources)
                else fname
            )
            img_index += 1
            try:
                meta = await descriptor.describe(url, fname)
                desc = (meta.get("description") or "").strip()
                usage_calls += 1
                usage_prompt_tokens += int(meta.get("input_tokens") or 0)
                usage_completion_tokens += int(meta.get("output_tokens") or 0)
                usage_total_cost += float(meta.get("total_cost") or 0.0)
                usage_latencies.append(float(meta.get("latency_ms") or 0.0))
                writer(
                    {
                        "type": "image_description_usage",
                        "source": label,
                        "input_tokens": int(meta.get("input_tokens") or 0),
                        "output_tokens": int(meta.get("output_tokens") or 0),
                        "total_tokens": int(meta.get("total_tokens") or 0),
                        "total_cost": float(meta.get("total_cost") or 0.0),
                        "latency_ms": float(meta.get("latency_ms") or 0.0),
                        "model_name": meta.get("model_name"),
                    }
                )
            except Exception as exc:
                logger.exception("Image describe failed for %s: %s", fname, exc)
                desc = "[Image could not be described.]"
            if ambient is not None:
                image_sections.append(f"{label}: {desc}")
            else:
                image_sections.append(f"[{fname}]\n{desc}")
            continue
        tx = _text_from_content_block(block)
        if tx:
            text_chunks.append(tx)

    base_text = "\n\n".join(text_chunks).strip()
    out_parts: list[str] = []
    if ambient is not None:
        for source in sources[img_index:]:
            if source == "microphone":
                image_sections.append(
                    "microphone: (audio transcription is not yet enabled for ambient observations)"
                )
        out_parts.append(observation_header(ambient))
        if base_text:
            out_parts.append(base_text)
        out_parts.append("\n\n".join(image_sections))
        final_text = "\n".join(part for part in out_parts if part)
    else:
        if base_text:
            out_parts.append(base_text)
        if image_sections:
            out_parts.append("---\nImage descriptions:\n" + "\n\n".join(image_sections))
        final_text = "\n\n".join(out_parts)

    update: dict = {
        "messages": [
            RemoveMessage(id=last.id),
            HumanMessage(
                id=last.id, content=final_text, additional_kwargs=additional_kwargs
            ),
        ]
    }
    if usage_calls:
        previous_calls = int(state.get("image_model_calls_count") or 0)
        previous_latencies = list(state.get("image_model_response_latency_list_ms") or [])
        all_latencies = previous_latencies + usage_latencies
        update.update(
            {
                "image_model_calls_count": previous_calls + usage_calls,
                "image_model_prompt_tokens": int(state.get("image_model_prompt_tokens") or 0)
                + usage_prompt_tokens,
                "image_model_completion_tokens": int(
                    state.get("image_model_completion_tokens") or 0
                )
                + usage_completion_tokens,
                "image_model_total_tokens": int(state.get("image_model_total_tokens") or 0)
                + usage_prompt_tokens
                + usage_completion_tokens,
                "image_model_total_cost": float(state.get("image_model_total_cost") or 0.0)
                + usage_total_cost,
                "image_model_response_latency_list_ms": usage_latencies,
                "image_model_average_latency_ms": (
                    sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
                ),
            }
        )
    return update


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
        # A bound machine is either reachable from this API process right now
        # (a live relay socket, or a tunnel/local address) or offline. Only the
        # reachable machines receive tools in ``think``; the prompt must agree
        # with the tools, so the capability guidance is added only when at
        # least one machine is reachable, and the offline machines are named
        # as offline so the avatar neither invents results for the offline
        # machines nor claims the offline machines cannot be seen at all.
        online_connections = [
            connection for connection in bound_mcp_connections if connection.online
        ]
        offline_connections = [
            connection
            for connection in bound_mcp_connections
            if not connection.online
        ]
        if online_connections:
            system_message_str = system_message_str + DATA_ANALYSIS_CAPABILITY_PROMPT
        # Naming the connected machines in the prompt lets the avatar answer
        # "which of my machines can you see?" without spending a tool call, and
        # keeps the model from inventing a machine name that is not connected.
        online_machine_names = ", ".join(
            connection.device_label for connection in online_connections
        )
        offline_machine_names = ", ".join(
            connection.device_label for connection in offline_connections
        )
        if online_connections:
            presence_sentence = (
                "The Neural Nexus MCP data server is connected for this avatar "
                "and reachable right now on the following machines: "
                f"{online_machine_names}. "
            )
        else:
            presence_sentence = (
                "The Neural Nexus MCP data server is connected for this avatar, "
                "but none of the connected machines is reachable right now, so "
                "no file or data-analysis tool is available this turn. "
            )
        if offline_connections:
            presence_sentence += (
                "The following connected machines are offline right now and "
                f"cannot be reached this turn: {offline_machine_names}. When "
                "asked, say plainly that those machines are offline; never "
                "invent results for an offline machine. "
            )
        system_message_str += (
            "\n<MCP_CONNECTION_STATUS>\n"
            + presence_sentence
            + "Confirm this plainly when asked, naming the machines. Never reveal "
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

        # Custom connectors — the owner's own Model Context Protocol servers —
        # gated identically. Their tools are attached with the connector's name
        # as a prefix, so the block names each connector and its tool count and
        # nothing else: the server address and the access token never appear.
        from src.anubis.utils.prompts.system_prompts import CUSTOM_CONNECTOR_PROMPT

        bound_connectors = [
            account
            for account in await bound_accounts_for(runtime.store, user_id, assistant_id)
            if account.get("kind") == "mcp_server"
        ]
        if bound_connectors:
            system_message_str = system_message_str + CUSTOM_CONNECTOR_PROMPT
            connector_descriptions = ", ".join(
                f"{account.get('display_label')} "
                f"({len((account.get('transport') or {}).get('tool_names') or [])} tools)"
                for account in bound_connectors
            )
            system_message_str += (
                "\n<CONNECTOR_STATUS>\n"
                "The following custom connectors are connected for this avatar: "
                f"{connector_descriptions}. Confirm this plainly when asked, naming "
                "the connectors. Never reveal a connector's server address or "
                "access token in a reply.\n"
                "</CONNECTOR_STATUS>\n"
            )

    # Agent inbox — the personal avatar only. The count and the top subjects
    # are named so the avatar raises pending items at the start of a
    # conversation without spending a tool call, and never invents one.
    if is_personal_avatar:
        try:
            from src.anubis.utils.inbox.repository import (
                OPEN_STATES,
                get_inbox_repository,
            )
            from src.anubis.utils.prompts.system_prompts import INBOX_CAPABILITY_PROMPT

            inbox_repository = get_inbox_repository()
            if inbox_repository is not None:
                system_message_str = system_message_str + INBOX_CAPABILITY_PROMPT
                pending_items = await inbox_repository.list_items(
                    assistant_id=assistant_id, states=OPEN_STATES, limit=5
                )
                pending_count = await inbox_repository.count_open(assistant_id)
                if pending_count:
                    headlines = "; ".join(
                        f"{item.get('sender') or 'unknown sender'} — "
                        f"{item.get('subject') or '(no subject)'} "
                        f"[{item.get('decision') or 'notify'}]"
                        for item in pending_items
                    )
                    system_message_str += (
                        "\n<INBOX_NOTIFICATIONS>\n"
                        f"{pending_count} item(s) are waiting for the conversation "
                        f"partner in the agent inbox: {headlines}. Mention them briefly "
                        "and offer to go through them.\n"
                        "</INBOX_NOTIFICATIONS>\n"
                    )
                else:
                    system_message_str += (
                        "\n<INBOX_NOTIFICATIONS>\nNo items are waiting in the agent "
                        "inbox right now.\n</INBOX_NOTIFICATIONS>\n"
                    )
        except Exception:  # noqa: BLE001 - the inbox must never fail a turn
            logger.debug("Inbox status unavailable for the prompt", exc_info=True)

    # Learning from media in conversation — the avatar's creator only (the
    # same owner check the ``think`` node applies before attaching the tool;
    # the subscription tier is enforced when the tool runs). The files attached
    # to this turn are named so the model can pick them by filename; the bytes
    # stay in the endpoint's attachment record and never enter the prompt.
    if avatar_owner_id is not None and avatar_owner_id == user_id:
        try:
            from src.anubis.utils.prompts.system_prompts import (
                IDENTITY_MEDIA_UPDATE_PROMPT,
            )
            from src.api.chat_attachments import describe_turn_attachments

            system_message_str = system_message_str + IDENTITY_MEDIA_UPDATE_PROMPT
            attached = describe_turn_attachments(
                config.get("configurable", {}).get("thread_id")
            )
            if attached:
                attachment_lines = "; ".join(
                    f"{item['filename']} ({item['mime_type']}, {item['size_bytes']} bytes)"
                    for item in attached
                )
                system_message_str += (
                    "\n<ATTACHED_MEDIA>\n"
                    f"Files attached to this turn: {attachment_lines}.\n"
                    "</ATTACHED_MEDIA>\n"
                )
            else:
                system_message_str += (
                    "\n<ATTACHED_MEDIA>\nNo files are attached to this turn.\n"
                    "</ATTACHED_MEDIA>\n"
                )
        except Exception:  # noqa: BLE001 - the block must never fail a turn
            logger.debug("Attached-media block unavailable for the prompt", exc_info=True)

    # Ambient vision: once the conversation partner shares a webcam or a screen,
    # the thread carries hidden ``[AMBIENT_OBSERVATION ...]`` turns.
    # The capability block explains those turns to the model; a thread without
    # any observation keeps its prompt unchanged.
    try:
        from src.anubis.utils.ambient.observations import is_ambient_observation
        from src.anubis.utils.prompts.system_prompts import (
            AMBIENT_VISION_CAPABILITY_PROMPT,
        )

        if any(is_ambient_observation(message) for message in state.get("messages") or []):
            system_message_str = system_message_str + AMBIENT_VISION_CAPABILITY_PROMPT
    except Exception:  # noqa: BLE001 - the block must never fail a turn
        logger.debug("Ambient-vision block unavailable for the prompt", exc_info=True)

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
