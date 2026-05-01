from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.context import UserContext, AssistantContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import format_docs, reduce_docs

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from datetime import datetime, timezone
from langchain_core.messages import SystemMessage
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
# Repo root: src/anubis/utils/nodes.py -> parents[3]
_AGENT_DEBUG_LOG = Path(__file__).resolve().parents[3] / ".cursor" / "debug-eecee8.log"


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
        # #region agent log
        try:
            import json as _json
            import time as _time

            with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _df:
                _df.write(
                    _json.dumps(
                        {
                            "sessionId": "eecee8",
                            "hypothesisId": "H1",
                            "location": "nodes.py:load_consciousness",
                            "message": "before assistant identity asearch",
                            "data": {"branch": "assistant_name_from_store"},
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
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
        # #region agent log
        try:
            import json as _json
            import time as _time

            with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _df:
                _df.write(
                    _json.dumps(
                        {
                            "sessionId": "eecee8",
                            "hypothesisId": "H1",
                            "location": "nodes.py:load_consciousness",
                            "message": "before user identity asearch",
                            "data": {"branch": "user_name_from_store"},
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
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

        """ search for reference image description and append if existent """
        assistent_reference_image_identity_namespace = (
            creator_id,
            assistant_id,
            "reference_image",
        )
        key = assistant_id
        reference_image_items = await runtime.store.aget(
            assistent_reference_image_identity_namespace, key
        )

        reference_image_items_list: list = []
        if reference_image_items is not None:
            if isinstance(reference_image_items, (list, tuple)):
                reference_image_items_list = list(reference_image_items)
            else:
                reference_image_items_list = [reference_image_items]

        reference_image_doc = reduce_docs([], reference_image_items_list)

        assistant_identity.extend(reference_image_doc)

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

    system_time = datetime.now(tz=timezone.utc).isoformat()

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
