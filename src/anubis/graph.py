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
from src.anubis.utils.configuration import GlobalConfiguration

from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

from dotenv import load_dotenv
load_dotenv()

from src.anubis.utils.context import IdentityContext, AssistantContext
from langchain_core.runnables import RunnableConfig

from langchain_core.runnables import RunnableConfig
from src.anubis.utils.helper_functions import extract_user_id_assistant_id, configure_assistant_context

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
from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.state import GlobalState

from src.anubis.utils.helper_functions import format_docs

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder

from langgraph.store.base import BaseStore

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.graph import MessagesState
""" TOOLS """

""" CONFIGURATION """

configuration = GlobalConfiguration()


""" NODES """

async def invoke_agent(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    
    configuration =  runtime.context.configuration

    if runtime.context.debug == "TRUE":

        logger.info(f"INVOKE AGENT NODE ")

        logger.info(f"state: {state}")
        logger.info(f"config: {config}")
        logger.info(f"runtime: {runtime}")
        logger.info(f"runtime.store: {runtime.store}")
        logger.info(f"runtime.context: {runtime.context}")

        logger.info(f"breakpoint invoke agent")

        logger.warning(f"THERE SHOULD BE ENVIRONMENT VARIABLES; configuration: {configuration}")

        logger.info(f"Testing store access")
        logger.info(f"runtime.store: {runtime.store}")
    
        # Asserting Current Identity:
        # user_id, assistant_id = await extract_user_id_assistant_id(config)
        logger.info(f"user_id: {state['user_state'].get("user_id", "")}")
        logger.info(f"assistant_id: {state['assistant_state'].get("assistant_id", "")}")

        # Loaded Identity:
        logger.info(f"user_identity: {state['user_state'].get("user_identity", [])}")
        logger.info(f"assistant_identity: {state['assistant_state'].get("assistant_identity", [])}")

        # Loaded Names
        logger.info(f"user_identity: {state['user_state'].get("user_name", "")}")
        logger.info(f"assistant_identity: {state['assistant_state'].get("assistant_name", "")}")

        logger.info(f"BREAKPOINT: UPDATE USER AND ASSISTANT CONTEXT")

        # await store.aput("testing", key="testing_key", value={"testing_key":"testing_value"})
        # testing_get = await store.aget("testing", key="testing_key", value={"testing_key":"testing_value"})
        # get_value = await store.aget("evan", key="name")
        # logger.info(f"get_value: {get_value}")
        # logger.info(f"testing_get: {testing_get}")

    """ CREATE MODEL """

    avatar_model = init_model(
        context = runtime.context
    )

    # summarization_model = init_model(
    #     context = runtime.context
    # )

    avatar = create_agent(
        model=avatar_model, 
        tools=[], 
    ) 
    # middleware=SummarizationMiddleware(model=summarization_model, )

    """ VECTORSTORE DOCUMENT RETRIEVAL """

    # logger.info(f"breakpoint")
    # logger.info(f"state['retrieved_docs']: {state['retrieved_docs']}")

    # Vectorstore Retrieved Docments
    # retrieved_docs = format_docs(state.get('retrieved_docs', []))

    # logger.info(f"format_docs(state.get('retrieved_docs', [])): {retrieved_docs}")

    """ RETRIEVE MEMORIES FROM NATURAL LANGUAGE GENERATED QUERY IN VECTORSTORE """
    
    # # TODO: PROMPT INJECT RETRIEVED MEMORIES 

    prompt_builder = DynamicPromptBuilder()

    # TODO: Update the assistant context from the store: details about the AI 

    """ POSTGRES STORE RETRIEVAL (METADATA AI/USER) """

    logger.info(f"async postgres store connection test breakpoint")

    # ai_context_item = await configure_assistant_context(config, store)

    system_time = datetime.now(tz=timezone.utc).isoformat()

    assistant_identity = state['assistant_state'].get("assistant_identity", [])
    assistant_name = state['assistant_state'].get("assistant_name", "")

    user_identity = state['user_state'].get('user_identity', [])
    user_name = state['user_state'].get("user_name", "")

    populated_identity_template = prompt_builder.build_prompt(
        assistant_name = assistant_name,
        assistant_identity= assistant_identity,
        user_name = user_name,
        user_identity=user_identity, 
        system_time = system_time,
    )

    logger.info(f"populated_template: {populated_identity_template}")

    # prepend system message
    system_identity = SystemMessage(content = populated_identity_template.messages)
    chat_prompt_template = ChatPromptTemplate(messages = system_identity + state["messages"])
    input = {"messages": chat_prompt_template.messages}
    
    logger.info(f"message input: {input}")

    response = await avatar.ainvoke(input=input)
    avatar_response = response.get("messages", [])[-1]

    logger.info(f"Avatar RESPONSE: {getattr(avatar_response.content, "content")}")
    result = {"messages": [avatar_response]}

    return result


async def summarize_conversation(state: GlobalState, runtime: Runtime[GlobalContext]):
    
    messages = state['messages']
    configuration = runtime.context.configuration


    # Retrieve the current message summary
    conversation_summary = state.get("conversation_summary", "") 
    if conversation_summary:

        # Extend the current conversation summary:
        summary_message = (
            f"This is the current summary of the conversation to the current message {conversation_summary}\n\n"
            +f"Extend this conversation summary with the included messages:"
        )
    else:
        summary_message = "Please create a summarization of the conversation using the included messages. Do not include this message as part of the summary"
    
    model = init_model(configuration=configuration)

    input_messages = [HumanMessage(context=summary_message)] + state['messages']

    # response = await model.ainvoke(input=HumanMessage(content=summary_message)

async def call_router(state: GlobalState, runtime: Runtime[GlobalContext]) -> Command[Literal["chat"]]:
    """Decides whether to upload new personal information or to respond based upon the chat

    Args:
        state (GlobalState): message state
        runtime (Runtime[GlobalContext]): context of agent and user and configuration

    Returns:
        command: next node location
    """
    logger.info(f"CALL ROUTER")
    
    # Create a model with structured output
    config = runtime.context.configuration # Loads env vars automatically

    model_router_structured_output = init_model(
        configuration=config,
        response_format=RouteDecision
    )

    decision_instructions = """Route to chat.
"""
    system_message = SystemMessage(content=decision_instructions)
    
    assert isinstance(state['messages'][-1], HumanMessage)
    assert hasattr(state['messages'][-1], "content")
    assert getattr(state['messages'][-1], "content") != ""

    human_message = state['messages'][-1]

    decided_route = await model_router_structured_output.ainvoke(
        [system_message, human_message]
    )
    logger.info(f"DECIDED ROUTE {decided_route}")
    
    return {"route_decision": decided_route}

async def route_node_from_decision(state: GlobalState, runtime: GlobalContext) -> str:
    logger.info(f"ROUTING NODE FROM DECISION")
    route_decision = state["route_decision"]
    if route_decision == "chat":
        return "invoke_agent"
    else:
        logger.info(f"NON CHAT")
        return "invoke_agent"

async def message_interface(state:MessagesState, config: RunnableConfig, runtime: Runtime) -> GlobalState:
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

    user_id = config.get("configurable",{}).get("langgraph_api_user_id", '')

    if user_id != '':
        user_state.update("user_id", user_id)
    else:
        """anonymous_user_id is 'str(uuid5(NAMESPACE_URL, 'anonymous_user_id"""
        user_state.update({"user_id":'9977df19-9ceb-5f87-a130-55f6a6282069'})
        
    
    assistant_id = config.get("configurable", {}).get("assistant_id", "")

    if assistant_id != "":
        assistant_state.update({"assistant_id":assistant_id})
    else:
        raise Exception("Assistant does not have an id from the configuration. Provide an assistant_id in config['configurable']['assistant_id'].")

    # Update Name and Description of User and Assistant if provided in the context

    assistant_name = getattr(runtime.context.assistant_ctx, "name", None)
    assistant_description = getattr(runtime.context.assistant_ctx, "description", None)

    user_name = getattr(runtime.context.user_ctx, "name", None)
    user_description = getattr(runtime.context.user_ctx, "description", None)
    
    if assistant_name is not None:
        assistant_state.update({"assistant_name": assistant_name})        
    else:
        runtime.store.asearch("")

    if assistant_description is not None:
        assistant_state.update({"assistant_description": assistant_description})        

    if user_name is not None:
        user_state.update({"user_name": user_name})        

    if user_description is not None:
        user_state.update({"user_description": user_description})        


    # Load the assistant identity from store:
    # Identity of the assistant
    assistant_identity_namespace = ("user_id", "assistant_id", "identity")

    # Identity of the user according to the assistant
    user_identity_namespace = (assistant_id, user_id, "identity")

    user_identity_document_items = await runtime.store.asearch(user_identity_namespace)
    # user_identity_document_items = user_identity_document_items_response.get("items", {})

    assistant_identity_document_items = await runtime.store.asearch(assistant_identity_namespace)

    # assistant_identity_document_items = assistant_identity_document_items_response.get("items", [])

    user_state.update({"user_identity": {"user_identity_documents": user_identity_document_items}})

    assistant_state.update({"assistant_identity": {"assistant_identity_documents": assistant_identity_document_items}})

    logger.info("breakpoint")

    return {"messages": state['messages'], "assistant_state": assistant_state, "user_state": user_state}


""" GRAPH """

# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState,
    input_schema = MessagesState,
    output_schema= MessagesState,
    context_schema = GlobalContext
)
# workflow.add_edge("message_interface", END)

# Add single node (your input/output)
# workflow.add_node("summarize_conversation", summarize_conversation)
# workflow.add_node("call_router", call_router)

# workflow.add_node("message_interface_graph", message_interface_graph)
workflow.add_node("chat", message_interface)
workflow.add_node("avatar", invoke_agent)

workflow.add_edge(START, "chat")
# workflow.add_node("retrieve_documents", retrieval_graph)

# Edges

# workflow.add_edge(START, 'retrieve_documents')
# workflow.add_edge('retrieve_documents', "invoke_agent")
workflow.add_edge('chat', "avatar")
workflow.add_edge("avatar", END)

graph = workflow.compile()

graph.name = "Anubis"

__all__ = ["graph"]
