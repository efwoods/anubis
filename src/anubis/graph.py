# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

# NOTE: ``retrieval_graph`` was imported here but never referenced in this module.
# Runtime logs showed ``from ... retrieval_graph import retrieval_graph`` alone took
# ~13 s per cold webapp worker because ``retrieval_graph`` pulls heavy retrieval deps.

from dotenv import load_dotenv

load_dotenv()


from langchain_core.runnables import RunnableConfig, Runnable
from src.anubis.utils.utility import (
    extract_user_id_assistant_id,
    configure_assistant_context,
)

from src.anubis.utils.schema import RouteDecision

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# NOTE: ``langchain.agents.create_agent`` and ``SummarizationMiddleware`` were
# imported here and below but never referenced in this file (the only
# ``create_agent(...)`` site is commented out at ~line 313).  Removed to skip the
# eager ``langchain.agents`` package load on every cold start.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from src.anubis.utils.model import init_model

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import format_docs

from langgraph.store.base import BaseStore

from src.anubis.utils.tokenizer import count_tokens


from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from pydantic import Field

from src.anubis.utils.nodes import load_consciousness, resolve_human_message_images


from src.anubis.utils.utility import (
    reduce_docs,
)

from src.anubis.utils.tools.identity.identity_tools import (
    learn_information_about_the_user,
    update_self_identity_mem_from_user_txt,
    # learn_information_about_yourself_through_images,
    # learn_information_about_yourself_through_tweets,
    # learn_information_about_yourself_through_youtube_videos,
    # learn_new_facts,
    # retrieve_knowledge,
    recall_memories,
    create_episodic_memory,
)

identity_tools = [
    learn_information_about_the_user,
    update_self_identity_mem_from_user_txt,
    # learn_information_about_yourself_through_images,
    # learn_information_about_yourself_through_tweets,
    # learn_information_about_yourself_through_youtube_videos,
    # learn_new_facts,
    # retrieve_knowledge,
    recall_memories,
    create_episodic_memory,
]

from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE, PRIVACY_POLICY

""" NODES """


def _coalesce_ai_message(full: AIMessage | AIMessageChunk) -> AIMessage:
    """Merge streamed chunks into a single AIMessage for graph state."""
    if isinstance(full, AIMessage):
        return full
    return AIMessage(
        content=full.content,
        additional_kwargs=dict(full.additional_kwargs or {}),
        response_metadata=dict(full.response_metadata or {}),
        id=full.id,
        tool_calls=list(full.tool_calls or []),
        invalid_tool_calls=list(full.invalid_tool_calls or []),
    )


async def message_interface(
    state: MessagesState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> GlobalState:
    assistant_state = {}
    user_state = {}

    # Assert the user is loggedin and the assistant has an id from the config:
    # Otherwise use an anonymouse user id

    logger.info("breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        config
    )

    user_state.update(updated_user_state)
    assistant_state.update(updated_assistant_state)

    return {
        "messages": state["messages"],
        "assistant_state": assistant_state,
        "user_state": user_state,
    }


# TODO: COERCE OUTPUT OF MESSAGE ONTO FINAL MESSAGE
async def terms_and_services_content_moderation(
    config: RunnableConfig, runtime: Runtime[GlobalContext]
):

    message = runtime.state["messages"][-1]

    class TermsAndServicesContentModeration(BaseModel):
        violation: bool = Field(
            description="If the user has violated the provided terms and services or the privacy policy then this value is TRUE. If the user has not violated the privacy policy or the terms and services then this value is FALSE"
        )
        reasoning: str = Field(
            description="This is the reason the user has violated the terms of service or the privacy policy. This is a clear reason. The exact terms of service violation or the exact violation of the privacy policy or both MUST be included and unaltered in any way."
        )

    TERMS_AND_SERVICES_CONTENT_MODERATION_SYSTEM_PROMPT = """
    <ROLE>
    You are an expert judge of identifying violations of the terms of service and privacy policy from human messages.
    </ROLE>
    
    <INSTRUCTIONS>
    Determine if the user MESSAGE is a violation of the included TERMS_OF_SERVICE or the PRIVACY_POLICY. 
    Include a reason the user MESSAGE has violated the terms of service or the privacy policy or both. 
    The reason the MESSAGE is a violation must be clear.
    The MESSAGE may not be in violation of the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a TRUE violation if the MESSAGE has violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a FALSE violation if the MESSAGE has NOT violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    MUST include EVERY exact line in the TERMS_OF_SERVICE or the PRIVACY_POLICY that give reason that the MESSAGE is a violation of either the TERMS_OF_SERVICE or the PRIVACY_POLICY whenever there is a violation in the MESSAGE of either or the TERMS_OF_SERVICE or the PRIVACY_POLICY or both.
    </INSTRUCTIONS>

    <TERMS_OF_SERVICE>
    {terms_of_service}
    </TERMS_OF_SERVICE>
    
    <PRIVACY_POLICY>
    {privacy_policy}
    </PRIVACY_POLICY>
    
    <INSTRUCTIONS>
    Determine if the user MESSAGE is a violation of the included TERMS_OF_SERVICE or the PRIVACY_POLICY. 
    Include a reason the user MESSAGE has violated the terms of service or the privacy policy or both. 
    The reason the MESSAGE is a violation must be clear.
    The MESSAGE may not be in violation of the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a TRUE violation if the MESSAGE has violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    Return a FALSE violation if the MESSAGE has NOT violated the TERMS_OF_SERVICE or the PRIVACY_POLICY at all.
    MUST include EVERY exact line in the TERMS_OF_SERVICE or the PRIVACY_POLICY that give reason that the MESSAGE is a violation of either the TERMS_OF_SERVICE or the PRIVACY_POLICY whenever there is a violation in the MESSAGE of either or the TERMS_OF_SERVICE or the PRIVACY_POLICY or both.
    </INSTRUCTIONS>
    
    <ROLE>
    You are an expert judge of identifying violations of the terms of service and privacy policy from human messages.
    </ROLE>
"""

    # TODO: CALCULATE TOKEN USAGE response['response_metadata']
    system_message = SystemMessage(
        content=TERMS_AND_SERVICES_CONTENT_MODERATION_SYSTEM_PROMPT.format(
            terms_of_service=TERMS_OF_SERVICE, privacy_policy=PRIVACY_POLICY
        )
    )
    # TODO: response_metrics_aggregation
    model_with_structured_output = init_model(
        model_without_tools=False, response_format=TermsAndServicesContentModeration
    )

    chat_prompt_template = [system_message] + [message]

    response = await model_with_structured_output.ainvoke(
        input=chat_prompt_template.messages
    )
    moderation_response = {
        "violation": response.violation,
        "reasoning": response.reasoning,
    }
    return {"moderation_response": moderation_response}

async def think(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """

    # model invocation
    # TODO: response_metrics_aggregation
    avatar_model_with_tools = init_model(
        context=runtime.context,
        tools=identity_tools,
    )

    # logger.info(f"breakpoint")
    messages = state["system_message"] + state["messages"] + state["internal_thoughts"]

    # TODO: response_metrics_aggregation
    merged: AIMessageChunk | AIMessage | None = None
    async for chunk in avatar_model_with_tools.astream(messages):
        merged = chunk if merged is None else merged + chunk
    assert merged is not None
    response = _coalesce_ai_message(merged)
    avatar_response_content = getattr(response, "content")
    logger.info(f"Avatar Model Response: {avatar_response_content}")
    return {"internal_thoughts": [response]}


process_thoughts = ToolNode(
    messages_key="internal_thoughts", tools=identity_tools, handle_tool_errors=True
)

async def considering(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> Literal["process_thoughts", "respond"]:
    recent_thought = state["internal_thoughts"][-1]
    if recent_thought.tool_calls:
        for tool_call in recent_thought.tool_calls:
            return "process_thoughts"
    else:
        return "respond"




async def respond(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """

    """ Agent Implementation """
    # avatar_model = init_model(
    #     context = runtime.context,
    # )

    # avatar = create_agent(model=avatar_model, tools=[
    #         ],
    #         state_schema=GlobalState,
    #         )

    messages = state["system_message"] + state["messages"]

    avatar_model = init_model(model_without_tools=False)
    # TODO: response_metrics_aggregation
    writer = get_stream_writer()
    merged: AIMessageChunk | AIMessage | None = None
    async for chunk in avatar_model.astream(messages):
        merged = chunk if merged is None else merged + chunk
        delta = chunk.content
        if isinstance(delta, str) and delta:
            writer({"type": "assistant_token", "text": delta})
    assert merged is not None
    avatar_response = _coalesce_ai_message(merged)

    # TODO: CALCULATE TOKEN USAGE response['response_metadata']

    logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")

    # response = await avatar.ainvoke(input={"messages": messages})
    # avatar_response = response.get("messages", [])[-1]
    # logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")


    from transformers import pipeline
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    sentiment = classifier(avatar_response.content)
    avatar_response.response_metadata.update({"sentiment":{"emotion": sentiment[0]["label"], 'score': sentiment[0]["score"]}})

    result = {"messages": [avatar_response]}

    return result


# async def evaluate_response_quality()

# async def update_response_metadata()

""" GRAPH """


""" RESPONSE ONLY FOR API ENDPOINT """

# Build minimal graph: START -> agent -> END
response_only_subgraph_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=GlobalState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

""" RESPONSE ONLY WORKFLOW NODES """

response_only_subgraph_workflow.add_node("load_consciousness", load_consciousness)
response_only_subgraph_workflow.add_node("respond", respond)

""" RESPONSE ONLY WORKFLOW EDGES """

response_only_subgraph_workflow.add_edge(START, "load_consciousness")
response_only_subgraph_workflow.add_edge("load_consciousness", "respond")
response_only_subgraph_workflow.add_edge("respond", END)

response_graph = response_only_subgraph_workflow.compile()

response_only_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=MessagesState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

# workflow.add_edge("terms_and_services_content_moderation", END)
response_only_workflow.add_node("chat", message_interface)
response_only_workflow.add_node(
    "resolve_human_message_images", resolve_human_message_images
)
response_only_workflow.add_node("anubis", response_graph)

response_only_workflow.add_edge(START, "chat")
response_only_workflow.add_edge("chat", "resolve_human_message_images")
response_only_workflow.add_edge("resolve_human_message_images", "anubis")
response_only_workflow.add_edge("anubis", END)

""" END OF REPSONSE ONLY FOR API ENDPOINT """

# Build minimal graph: START -> agent -> END
anubis_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=GlobalState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

""" ANUBIS WORKFLOW NODES """

# workflow.add_node("terms_and_services_content_moderation", terms_and_services_content_moderation)

anubis_workflow.add_node("load_consciousness", load_consciousness)
anubis_workflow.add_node("think", think)
anubis_workflow.add_node("process_thoughts", process_thoughts)
anubis_workflow.add_node("respond", respond)

# workflow.add_node("evaluate_response_quality", evaluate_response_quality)

# workflow.add_node("update_response_metadata", update_response_metadata)

""" ANUBIS WORKFLOW EDGES """

anubis_workflow.add_edge(START, "load_consciousness")
anubis_workflow.add_edge("load_consciousness", "think")

anubis_workflow.add_conditional_edges(
    "think", considering, {"process_thoughts": "process_thoughts", "respond": "respond"}
)
anubis_workflow.add_edge("process_thoughts", "load_consciousness")
# anubis_workflow.add_edge("avatar_tool_node", "load_consciousness")
anubis_workflow.add_edge("respond", END)

# workflow.add_edge("chat", "terms_and_services_content_moderation")


# COERCION
# workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tools':'avatar_tools', END:"evaluate_response_quality"})
# workflow.add_edge("evaluate_response_quality", "update_response_metadata")
# workflow.add_edge("terms_and_services_content_moderation", "update_response_metadata")
# workflow.add_edge("update_response_metadata", END)


anubis = anubis_workflow.compile()

message_workflow = StateGraph(
    state_schema=GlobalState,
    input_schema=MessagesState,
    output_schema=MessagesState,
    context_schema=GlobalContext,
)

# workflow.add_edge("terms_and_services_content_moderation", END)
message_workflow.add_node("chat", message_interface)
message_workflow.add_node(
    "resolve_human_message_images", resolve_human_message_images
)
message_workflow.add_node("anubis", anubis)

message_workflow.add_edge(START, "chat")
message_workflow.add_edge("chat", "resolve_human_message_images")
message_workflow.add_edge("resolve_human_message_images", "anubis")
message_workflow.add_edge("anubis", END)

graph = message_workflow.compile()

graph.name = "Anubis"

__all__ = ["graph"]
