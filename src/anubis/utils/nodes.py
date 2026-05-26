from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.context import UserContext, AssistantContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import format_docs, reduce_docs

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
import logging
from pathlib import Path

from src.anubis.utils.classes.ImageDescriptionClass import ImageDescriptionClass

logger = logging.getLogger(__name__)


def _resolve_user_timezone(tz_name: str | None):
    """Resolve a client-supplied IANA timezone name to a ``tzinfo``.

    The frontend sends the browser's IANA zone (e.g. ``"America/New_York"`` from
    ``st.context.timezone``) so the system clock injected into the prompt reflects
    the user's local time regardless of where the server runs. Falls back to UTC
    when the value is missing or not a recognized zone.
    """
    if not tz_name:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown user timezone %r; falling back to UTC", tz_name)
        return timezone.utc


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


async def load_consciousness(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    user_id = state["user_state"]["user_id"]
    assistant_id = state["assistant_state"]["assistant_id"]

    # Update Name and Description of User and Assistant if provided in the context
    logger.info(f"conscioussness breakpoint")
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

    if assistant_name is not None:
        state["assistant_state"].update({"assistant_name": assistant_name})
    else:
        assistant_possible_name = await runtime.store.asearch(
            (user_id, assistant_id, "identity"), query="name"
        )
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

    if assistant_description is not None:
        state["assistant_state"].update(
            {"assistant_description": assistant_description}
        )

    if user_name is not None:
        state["user_state"].update({"user_name": user_name})
    else:
        user_possible_name = await runtime.store.asearch(
            (assistant_id, user_id, "identity"), query="name"
        )
        if len(user_possible_name) > 0:
            user_name = (
                getattr(user_possible_name[0], "value")
                .get("document", {})
                .get("kwargs", {})
                .get("metadata", {})
                .get("fact", "")
            )
        else:
            user_name = ""

    if user_description is not None:
        state["user_state"].update({"user_description": user_description})

    """ Load User Identity documents """

    if len(state["user_identity_documents"]) == 0:
        user_identity_namespace = (assistant_id, user_id, "identity")

        user_identity_document_items = await runtime.store.asearch(
            user_identity_namespace
        )

        # Coerce into document objects from Search Items
        user_identity = reduce_docs([], user_identity_document_items)
    else:
        user_identity = state["user_identity_documents"]

    """ Load Assistant Identity documents """
    creator_id = config["configurable"]["assistant_ctx"]["metadata"]["user_id"]

    if len(state["assistant_identity_documents"]) == 0:
        assistant_identity_namespace = (creator_id, assistant_id, "identity")

        assistant_identity_document_items = await runtime.store.asearch(
            assistant_identity_namespace
        )

        # Coerce into document objects from Search Items
        assistant_identity = reduce_docs([], assistant_identity_document_items)

        assistant_identity_memory_namespace = (
            creator_id,
            assistant_id,
            "identity_memory",
        )
        retrieved_identity_memories_items = await runtime.store.asearch(
            assistant_identity_memory_namespace, limit=1000
        )
        retrieved_identity_memories = reduce_docs([], retrieved_identity_memories_items)
        assistant_identity.extend(retrieved_identity_memories)

    else:
        assistant_identity = state["assistant_identity_documents"]

    """ Always merge assistant reference image (creator namespace), including when identity docs are cached """
    assistent_reference_image_identity_namespace = (
        creator_id,
        assistant_id,
        "reference_image",
    )
    reference_image_items = await runtime.store.aget(
        assistent_reference_image_identity_namespace, assistant_id
    )

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

    # retrieved_memories = state['recalled_memory_documents']

    # if len(retrieved_memories) == 0:
    #     retrieved_memories = None

    """ Retrieve memories """

    query = state["messages"][-1].content
    if isinstance(query, list):
        query = query[0]["text"]

    assistant_memory_namespace = (user_id, assistant_id, "memory")
    retrieved_memories_items = await runtime.store.asearch(
        assistant_memory_namespace, query=query, limit=1000
    )

    # Coerce into document objects from Search Items
    retrieved_memories = reduce_docs([], retrieved_memories_items)

    # retrieved_memories.extend(retrieved_identity_memories)

    # if state['recalled_memory_documents'] is None or len(state['recalled_memory_documents']) == 0:
    #     assistant_identity_namespace = (user_id, assistant_id, "memory")
    #     query = state['messages'][-1].content

    #     retrieved_memories_items = await runtime.store.asearch(assistant_identity_namespace, query=query)

    #     # Coerce into document objects from Search Items
    #     retrieved_memories = reduce_docs([], retrieved_memories_items)
    # else:
    #     retrieved_memories = state['recalled_memory_documents']

    logger.info("breakpoint")

    """ Retrieve Direct Quotes """

    # Few Shot Example of Quotes and Writing style directly from the real-world assistant
    # The QUOTE namespace holds direct quotes from the real-world assistant

    direct_quote_items = await runtime.store.asearch(
        (creator_id, assistant_id, "quote"), query=query
    )
    logger.info(f"direct_quote_items: {direct_quote_items}")

    direct_quotes = reduce_docs([], direct_quote_items)

    """ Retrieve Documents """

    # document namespace is reserved for non-quotes that the assistant has access to (bible, menu, etc.)
    retrieved_knowledge_items = await runtime.store.asearch(
        (creator_id, assistant_id, "document"), query=query
    )
    logger.info(f"retrieved_knowledge_items: {retrieved_knowledge_items}")
    retrieved_knowledge = reduce_docs([], retrieved_knowledge_items)

    """ Retrieve Emotions """

    # from src.anubis.utils.prompts.psycho_analysis import plutchik_emotional_wheel_analysis_prompt
    from src.anubis.utils.state import EmotionSummarization

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
        direct_quotes=direct_quotes,
        user_name=user_name,
        user_description=user_description,
        user_identity=user_identity,
        system_time=system_time,
    )

    logger.info(f"populated_template: {populated_identity_template}")

    # prepend system message
    logger.info(f"state['messages']: {state['messages']}")

    system_message_str = populated_identity_template.messages[0].content


    # Dev-only: dump the created system prompt to a file with real newlines
    # (logger.info shows escaped "\n"; writing the str directly yields newlines).
    if getattr(runtime, "context", None) and runtime.context.dev == "TRUE":
        dev_prompt_path = Path("system_prompt.txt")
        dev_prompt_path.write_text(system_message_str, encoding="utf-8")
        logger.info(f"dev system prompt written to: {dev_prompt_path.resolve()}")


    input_update = {
        "user_identity_documents": user_identity,
        "assistant_identity_documents": assistant_identity,
        "system_message": [
            SystemMessage(
                content=system_message_str, id="00000000-0000-0000-0000-0000000000000"
            )
        ],
    }

    return input_update
