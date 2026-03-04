# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext

from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

from dotenv import load_dotenv
load_dotenv()

from src.anubis.utils.context import UserContext, AssistantContext
from langchain_core.runnables import RunnableConfig

from langchain_core.runnables import RunnableConfig
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
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

from src.anubis.utils.utility import format_docs

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder

from langgraph.store.base import BaseStore

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode


from pydantic import Field

from src.anubis.utils.tools import (
    learn_information_about_the_user, 
    learn_information_about_yourself_through_text_from_the_user_as_a_memory,
    # learn_information_about_yourself_through_images,
    # learn_information_about_yourself_through_tweets,
    # learn_information_about_yourself_through_youtube_videos,
    remember_memories,
    # learn_new_facts,
    # retrieve_knowledge,
)

from src.anubis.utils.utility import (
    reduce_docs, 
)

from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE, PRIVACY_POLICY

""" NODES """

async def message_interface(state:MessagesState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> GlobalState:
    logger.info(f"state:{state}")
    logger.info(f"config:{config}")
    logger.info(f"runtime:{runtime}")
    logger.info(f"runtime.store:{runtime.store}")
    logger.info(f"runtime.context: {runtime.context}")

    logger.info(f"assistant_id:{config['configurable']['assistant_id']}")
    logger.info(f"configurable:{config['configurable']['langgraph_auth_user_id']}")
    logger.info(f"configurable:{config['configurable']}")
    logger.info(f"THIS IS AN UPDATE")
    logger.info(f"THIS IS ANOTHER UPDATE")

    assistant_state = {}
    user_state = {}

    # Assert the user is loggedin and the assistant has an id from the config:
    # Otherwise use an anonymouse user id

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(config, runtime)

    user_state.update(updated_user_state)
    assistant_state.update(updated_assistant_state)

    return {"messages": state['messages'], "assistant_state": assistant_state, "user_state": user_state}


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
    model_with_structured_output = init_model(context=GlobalContext, response_format=TermsAndServicesContentModeration)

    chat_prompt_template = [system_message] + [message]

    response = await model_with_structured_output.ainvoke(input=chat_prompt_template.messages)
    moderation_response = {
        "violation": response.violation,
        "reasoning": response.reasoning,
    }
    return {"moderation_response": moderation_response}


async def update_identity_tool_classification(state:GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    """
    Identify and handle identity tool calls.
    """
    
    model_with_identity_tools = init_model(
        context=runtime.context, 
        tools=[
            learn_information_about_the_user, learn_information_about_yourself_through_text_from_the_user_as_a_memory
        ])
    identity_tools_message = model_with_identity_tools.ainvoke(state['messages'])
    return {'messages':identity_tools_message}

async def update_identity_tool_condition(state: GlobalState) -> Literal["update_identity_tools", "load_consciousness"]:
    recent_message = state['messages'][-1]
    if recent_message.tool_calls:
        return "update_identity_tools" 
    else:
        return "load_consciousness"    



async def load_consciousness(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    user_id = state["user_state"]['user_id']
    assistant_id = state['assistant_state']['assistant_id']

    # Update Name and Description of User and Assistant if provided in the context

    if (isinstance(runtime.context.assistant_ctx, AssistantContext)):
        assistant_name = getattr(runtime.context.assistant_ctx, "name", None)
        assistant_description = getattr(runtime.context.assistant_ctx, "description", None)
    else:
        assert(type(runtime.context.assistant_ctx) is dict)
        assistant_name = runtime.context.assistant_ctx.get("name", None)
        assistant_description = runtime.context.assistant_ctx.get("description", None)

    if (isinstance(runtime.context.user_ctx, UserContext)):
        user_name = getattr(runtime.context.user_ctx, "name", None)
        user_description = getattr(runtime.context.user_ctx, "description", None)
    else:
        assert(type(runtime.context.user_ctx) is dict)
        user_name = runtime.context.user_ctx.get("name", None)
        user_description = runtime.context.user_ctx.get("description", None)  
    
    
    if assistant_name is not None:
        state['assistant_state'].update({'assistant_name': assistant_name})        
    else:
        possible_name = await runtime.store.asearch((user_id, assistant_id, "identity"), query="name")
        if len(possible_name) > 0:
            assistant_name = getattr(possible_name[0], "value").get("document", {}).get("metadata", {}).get("fact",'')     
        else:
            assistant_name = ""
        
    if assistant_description is not None:
        state['assistant_state'].update({"assistant_description": assistant_description})        

    if user_name is not None:
        state['user_state'].update({'user_name': user_name})        
    else:
        possible_name = await runtime.store.asearch((assistant_id, user_id, "identity"), query="name")
        if len(possible_name) > 0:
            user_name = getattr(possible_name[0], "value").get("document", {}).get("metadata", {}).get("fact",'')
        else:
            user_name = ""

    if user_description is not None:
        state['user_state'].update({"user_description": user_description})        


    # Load the assistant identity from store:
    # Identity of the assistant
    assistant_identity_namespace = ("user_id", "assistant_id", "identity")

    # Identity of the user according to the assistant
    user_identity_namespace = (assistant_id, user_id, "identity")

    user_identity_document_items = await runtime.store.asearch(user_identity_namespace)

    # Coerce into document objects from Search Items
    user_identity_document_items = reduce_docs([], user_identity_document_items)

    assistant_identity_document_items = await runtime.store.asearch(assistant_identity_namespace)

    # Coerce into document objects from Search Items
    assistant_identity_document_items = reduce_docs([], assistant_identity_document_items)

    state['user_state'].update({'user_identity': {"user_identity_documents": user_identity_document_items}})

    state['assistant_state'].update({'assistant_identity': {"assistant_identity_documents": assistant_identity_document_items}})

    logger.info("breakpoint")

    # retrieved_memories = state['assistant_state'].get("recalled_memories", {}).get("recalled_memory_documents", [])
    retrieved_memories = state['recalled_memory_documents']
    
    if len(retrieved_memories) == 0:
        retrieved_memories = None

    prompt_builder = DynamicPromptBuilder()

    system_time = datetime.now(tz=timezone.utc).isoformat()

    assistant_identity = state['assistant_state'].get('assistant_identity', [])
    assistant_name = state['assistant_state'].get('assistant_name','')

    user_identity = state['user_state'].get('user_identity', [])
    user_name = state['user_state'].get('user_name','')

    populated_identity_template = prompt_builder.build_prompt(
        assistant_name = assistant_name,
        assistant_identity= assistant_identity,
        retrieved_memories=retrieved_memories,
        user_name = user_name,
        user_identity=user_identity, 
        system_time = system_time,
    )

    logger.info(f"populated_template: {populated_identity_template}")

    # prepend system message
    messages = [message for message in state['messages'] if type(message) is not SystemMessage]
    messages = populated_identity_template.messages + state['messages']
    input = {"messages": messages}
    
    logger.info(f"message input: {input}")
    return input

async def invoke_agent(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """


    # model invocation
    avatar_model_with_tools = init_model(
        context = runtime.context,
        tools = [
            remember_memories
        ]
    )

    response = await avatar_model_with_tools.ainvoke(input=state['messages'])
    avatar_response_content = getattr(response, 'content')
    logger.info(f"Avatar Model Response: {avatar_response_content}")
    return {"messages":[response]}

    # agent invocation
    # avatar_model = init_model(
    #     context = runtime.context,
    # )

    # avatar = create_agent(model=avatar_model, tools=[remember_memories])

    # response = await avatar.ainvoke(input={"messages": state['messages']})
    # avatar_response = response.get("messages", [])[-1]

    # logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")
    # result = {"messages": [avatar_response]}
    # return result
 
async def avatar_tools_condition(state:GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> Literal["avatar_tools", '__end__']:
    recent_message = state['messages'][-1]
    if recent_message.tool_calls:
        return "avatar_tools"
    else:
        return "__end__"
    
# async def evaluate_response_quality()
    
# async def update_response_metadata()
    
""" GRAPH """

# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState,
    input_schema = MessagesState,
    output_schema= MessagesState,
    context_schema = GlobalContext
)


update_identity_tools = ToolNode([learn_information_about_the_user, learn_information_about_yourself_through_text_from_the_user_as_a_memory])

avatar_tools = ToolNode([remember_memories])


# TOOL WORKFLOW 

""" TOOL WORKFLOW NODES """

# workflow.add_node("chat", message_interface)

# workflow.add_node("terms_and_services_content_moderation", terms_and_services_content_moderation)

workflow.add_node("update_identity_tool_classification", update_identity_tool_classification)
workflow.add_node("update_identity_tools", update_identity_tools)

# workflow.add_node("load_consciousness", load_consciousness)
# workflow.add_node("respond", invoke_agent)
workflow.add_node("avatar_tools", avatar_tools)

# workflow.add_node("evaluate_response_quality", evaluate_response_quality)

# workflow.add_node("update_response_metadata", update_response_metadata)

# """ TOOL WORKFLOW EDGES """

# workflow.add_edge(START, "chat")
# workflow.add_edge("chat", "terms_and_services_content_moderation")

# workflow.add_edge("chat", "update_identity_tool_classification")
workflow.add_conditional_edges("update_identity_tool_classification", update_identity_tool_condition, {"update_identity_tools": "update_identity_tools", "__end__":"load_consciousness"})
# workflow.add_edge("update_identity_tools", "update_identity_tool_classification")

# workflow.add_edge("load_consciousness", "respond")

# COERCION
# workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tools':'avatar_tools', END:"evaluate_response_quality"})
# workflow.add_edge("evaluate_response_quality", "update_response_metadata")
# workflow.add_edge("terms_and_services_content_moderation", "update_response_metadata")
# workflow.add_edge("update_response_metadata", END)


# TOOL TESTING
workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tools':'avatar_tools', "__end__":"__end__"})
workflow.add_edge("avatar_tools", "load_consciousness")

# workflow.add_edge("terms_and_services_content_moderation", END)


# BASIC CHAT WORKFLOW

""" BASIC CHAT WORKFLOW NODES """

workflow.add_node("chat", message_interface)
workflow.add_node("load_consciousness", load_consciousness)
workflow.add_node("respond", invoke_agent)

""" BASIC CHAT WORKFLOW EDGES """

workflow.add_edge(START, "chat")
workflow.add_edge("chat", "load_consciousness")
workflow.add_edge("load_consciousness", "respond")
# workflow.add_edge("respond", END)

graph = workflow.compile()

graph.name = "Anubis"

__all__ = ["graph"]
