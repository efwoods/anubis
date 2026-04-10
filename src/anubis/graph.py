# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END


from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

from dotenv import load_dotenv
load_dotenv()


from langchain_core.runnables import RunnableConfig, Runnable
from src.anubis.utils.utility import extract_user_id_assistant_id, configure_assistant_context

from src.anubis.utils.schema import RouteDecision

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.runtime import Runtime   

from src.anubis.utils.model import init_model
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import format_docs

from langgraph.store.base import BaseStore

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from pydantic import Field

from src.anubis.utils.nodes import load_consciousness


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
    create_episodic_memory
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
    create_episodic_memory
]

from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE, PRIVACY_POLICY

""" NODES """

async def message_interface(state:MessagesState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> GlobalState:
    # logger.info(f"state:{state}")
    # logger.info(f"config:{config}")
    # logger.info(f"runtime:{runtime}")
    # logger.info(f"runtime.store:{runtime.store}")
    # logger.info(f"runtime.context: {runtime.context}")

    # logger.info(f"assistant_id:{config['configurable']['assistant_id']}")
    # logger.info(f"configurable:{config['configurable']['langgraph_auth_user_id']}")
    # logger.info(f"configurable:{config['configurable']}")
    # logger.info(f"THIS IS AN UPDATE")
    # logger.info(f"THIS IS ANOTHER UPDATE")

    assistant_state = {}
    user_state = {}

    # Assert the user is loggedin and the assistant has an id from the config:
    # Otherwise use an anonymouse user id

    logger.info("breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(config)

    user_state.update(updated_user_state)
    assistant_state.update(updated_assistant_state)

    return {
            "messages": state['messages'], 
            "assistant_state": assistant_state, 
            "user_state": user_state, 
            }


# TODO: COERCE OUTPUT OF MESSAGE ONTO FINAL MESSAGE
async def terms_and_services_content_moderation(config: RunnableConfig, runtime: Runtime[GlobalContext]):
    
    message = runtime.state['messages'][-1]
    class TermsAndServicesContentModeration(BaseModel):
        violation: bool = Field(description="If the user has violated the provided terms and services or the privacy policy then this value is TRUE. If the user has not violated the privacy policy or the terms and services then this value is FALSE")
        reasoning: str = Field(description="This is the reason the user has violated the terms of service or the privacy policy. This is a clear reason. The exact terms of service violation or the exact violation of the privacy policy or both MUST be included and unaltered in any way.")

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

    system_message = SystemMessage(content = TERMS_AND_SERVICES_CONTENT_MODERATION_SYSTEM_PROMPT.format(terms_of_service=TERMS_OF_SERVICE, privacy_policy = PRIVACY_POLICY))
    model_with_structured_output = init_model(model_without_tools=True, response_format=TermsAndServicesContentModeration)

    chat_prompt_template = [system_message] + [message]

    response = await model_with_structured_output.ainvoke(input=chat_prompt_template.messages)
    moderation_response = {
        "violation": response.violation,
        "reasoning": response.reasoning,
    }
    return {"moderation_response": moderation_response}


from src.anubis.utils.tools.identity.identity_tools import test_update
async def think(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """

    # model invocation
    avatar_model_with_tools = init_model(
        context = runtime.context,
        tools = identity_tools, 
        )

    # logger.info(f"breakpoint")
    messages = state['system_message'] + state['messages'] + state['internal_thoughts']

    response = await avatar_model_with_tools.ainvoke(input=messages)
    avatar_response_content = getattr(response, 'content')
    logger.info(f"Avatar Model Response: {avatar_response_content}")
    return {"internal_thoughts":[response]}

process_thoughts = ToolNode(
    messages_key ="internal_thoughts", 
    tools=identity_tools, 
    handle_tool_errors=True)

from langchain.tools import ToolRuntime

async def considering(state:GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> Literal["process_thoughts", 'respond']:
    recent_thought = state['internal_thoughts'][-1]
    if recent_thought.tool_calls:
        for tool_call in recent_thought.tool_calls:
            return "process_thoughts"
    else:
        return "respond"

# async def process_thoughts(state: GlobalState, config: RunnableConfig, runtime:Runtime
# [GlobalContext]) -> GlobalState:
#     avatar_accessible_tools_dict = {
#         "learn_information_about_the_user": learn_information_about_the_user,
#         "update_self_identity_mem_from_user_txt":update_self_identity_mem_from_user_txt, 
#         "recall_memories":recall_memories,
#         "create_episodic_memory": create_episodic_memory,
#         "test_update": test_update,
#         "test_update_second":test_update_second
#         }
    
#     # avatar_accessible_tool_names = avatar_accessible_tools_dict.keys()
    
#     message = state['internal_thoughts'][-1]
#     logger.info(f"breakpoint")    
#     thoughts = []

#     for tool_call in message.tool_calls:
#             if tool_call['name'] in avatar_accessible_tools_dict:
#                 tool = avatar_accessible_tools_dict[tool_call['name']]
#                 tool_call_id = tool_call.get('id')
#                 tool_runtime = ToolRuntime(
#                     state=state, 
#                     config=config, 
#                     context=runtime.context, 
#                     store=runtime.store,
#                     tool_call_id = tool_call_id,
#                     stream_writer=runtime.stream_writer
#                     )
#                 logger.warning(f"tool_call: {tool_call}")
#                 tool_call["args"].update({"runtime":tool_runtime})

#                 logger.info("process_thoughts breakpoint")
#                 thought = await tool.ainvoke(
#                     tool_call['args'], 
#                 )
#                 return thought
                # thoughts.append(thought)

async def respond(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
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

    messages = state['system_message'] + state['messages']

    avatar_model = init_model(model_without_tools=True)
    avatar_response = await avatar_model.ainvoke(messages)

    logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")

    # response = await avatar.ainvoke(input={"messages": messages})
    # avatar_response = response.get("messages", [])[-1]
    # logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")

    result = {"messages": [avatar_response]}

    return result

# async def evaluate_response_quality()
    
# async def update_response_metadata()

""" GRAPH """
# Build minimal graph: START -> agent -> END
anubis_workflow = StateGraph(
    state_schema = GlobalState,
    input_schema = GlobalState,
    output_schema = MessagesState,
    context_schema = GlobalContext
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

anubis_workflow.add_conditional_edges("think", considering, {'process_thoughts':'process_thoughts', "respond":"respond"})
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
    state_schema = GlobalState,
    input_schema = MessagesState,
    output_schema = MessagesState,
    context_schema = GlobalContext
)

# workflow.add_edge("terms_and_services_content_moderation", END)
message_workflow.add_node("chat", message_interface)
message_workflow.add_node("anubis", anubis)

message_workflow.add_edge(START, "chat")
message_workflow.add_edge("chat", "anubis")
message_workflow.add_edge("anubis", END)

graph = message_workflow.compile()

graph.name = "Anubis"

__all__ = ["graph"]
