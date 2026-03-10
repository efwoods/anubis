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

from src.anubis.utils.tools.identity.identity_tools import (
    learn_information_about_the_user, 
    learn_information_about_yourself_through_text_from_the_user_as_a_memory,
    # learn_information_about_yourself_through_images,
    # learn_information_about_yourself_through_tweets,
    # learn_information_about_yourself_through_youtube_videos,
    # learn_new_facts,
    # retrieve_knowledge,
    recall_memories,    
    create_episodic_memory
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

# identity_tools_list = {"learn_information_about_yourself_through_text_from_the_user_as_a_memory": learn_information_about_yourself_through_text_from_the_user_as_a_memory, "learn_information_about_the_user":learn_information_about_the_user}

# identity_tools = [learn_information_about_yourself_through_text_from_the_user_as_a_memory, learn_information_about_the_user]

# async def update_identity_tool_classification(state:GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
#     """
#     Identify and handle identity tool calls.
#     """
#     logger.info("breakpoint")
#     model_with_identity_tools = init_model(
#         context=runtime.context, 
#         tools=identity_tools,
#         tool_choice = {
#             "type":"function", "function":
#             {"name": "learn_information_about_yourself_through_text_from_the_user_as_a_memory"},
#             "type":"function", "function":
#             {"name": "learn_information_about_the_user"}, 
#         }
#     )


#     messages = state['messages']
#     try:
#         identity_tools_message = await model_with_identity_tools.ainvoke(messages)
#     except openai.BadRequestError as e:
#         if "function name not found" in str(e):
#             return 

#     tool_results = []

#     if identity_tools_message.tool_calls:
#         for tool_call in identity_tools_message.tool_calls:
#             if tool_call['name'] in identity_tools_list.keys():
#                 tool_result = await identity_tools_list[tool_call['name']].invoke(messages)
#                 tool_results.append(tool_result)

#     return Command(goto="load_consciousness")

# async def update_identity_tool_condition(state: GlobalState) -> Literal["update_identity_tools", "load_consciousness"]:
#     logger.info("breakpoint")
#     recent_message = state['messages'][-1]
#     if recent_message.tool_calls:
#         for tool_call in recent_message.tool_calls:
#             if tool_call.get("name", "") in update_identity_accessible_tools_list:
#                 return "update_identity_tools" 
#             else:
#                 return "load_consciousness"
    
#     return "load_consciousness"    

async def load_consciousness(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    user_id = state["user_state"]['user_id']
    assistant_id = state['assistant_state']['assistant_id']

    # Update Name and Description of User and Assistant if provided in the context
    logger.info(f"conscioussness breakpoint")

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
        assistant_possible_name = await runtime.store.asearch((user_id, assistant_id, "identity"), query="name")
        if len(assistant_possible_name) > 0:
            assistant_name = getattr(assistant_possible_name[0], "value").get("document", {}).get("kwargs", {}).get("metadata", {}).get("fact",'')     
        else:
            assistant_name = ""
        
    if assistant_description is not None:
        state['assistant_state'].update({"assistant_description": assistant_description})        

    if user_name is not None:
        state['user_state'].update({'user_name': user_name})        
    else:
        user_possible_name = await runtime.store.asearch((assistant_id, user_id, "identity"), query="name")
        if len(user_possible_name) > 0:
            user_name = getattr(user_possible_name[0], "value").get("document", {}).get("kwargs", {}).get("metadata", {}).get("fact",'')
        else:
            user_name = ""

    if user_description is not None:
        state['user_state'].update({"user_description": user_description})        


    if state['user_identity_documents'] is None or len(state['user_identity_documents']) == 0:
        user_identity_namespace = (assistant_id, user_id, "identity")
        
        user_identity_document_items = await runtime.store.asearch(user_identity_namespace)

        # Coerce into document objects from Search Items
        user_identity = reduce_docs([], user_identity_document_items)
    else:
        user_identity = state['user_identity_documents']


    if state['assistant_identity_documents'] is None or len(state['assistant_identity_documents']) == 0:
        assistant_identity_namespace = (user_id, assistant_id, "identity")
        
        assistant_identity_document_items = await runtime.store.asearch(assistant_identity_namespace)

        # Coerce into document objects from Search Items
        assistant_identity = reduce_docs([], assistant_identity_document_items)
    else:
        assistant_identity = state['assistant_identity_documents']

    logger.info("breakpoint")

    # retrieved_memories = state['assistant_state'].get("recalled_memories", {}).get("recalled_memory_documents", [])
    retrieved_memories = state['recalled_memory_documents']
    
    if len(retrieved_memories) == 0:
        retrieved_memories = None

    # Few Shot Example of Quotes and Writing style directly from the real-world assistant
    # The QUOTE namespace holds direct quotes from the real-world assistant
    query = state['messages'][-1].content
    direct_quote_items = await runtime.store.asearch((user_id, assistant_id, 'quote'), query=query)
    logger.info(f"direct_quote_items: {direct_quote_items}")

    direct_quotes = reduce_docs([], direct_quote_items)

    # document namespace is reserved for non-quotes that the assistant has access to (bible, menu, etc.)
    retrieved_knowledge_items = await runtime.store.asearch((user_id, assistant_id, 'document'), query=query)
    logger.info(f"retrieved_knowledge_items: {retrieved_knowledge_items}")
    retrieved_knowledge = reduce_docs([], retrieved_knowledge_items)

    # Search your feelings
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
    assistant_name = state['assistant_state'].get('assistant_name','')

    # user_identity = state['user_state'].get('user_identity', [])
    user_name = state['user_state'].get('user_name','')

    populated_identity_template = prompt_builder.build_prompt(
        assistant_name = assistant_name,
        assistant_description = assistant_description,
        assistant_identity= assistant_identity,
        retrieved_memories=retrieved_memories,
        retrieved_knowledge=retrieved_knowledge,
        direct_quotes = direct_quotes,
        user_name = user_name,
        user_description = user_description,
        user_identity=user_identity, 
        system_time = system_time,
    )

    logger.info(f"populated_template: {populated_identity_template}")

    # prepend system message
    logger.info(f"state['messages']: {state['messages']}")

    system_message_str = populated_identity_template.messages[0].content

    input_update = { 
                    "user_identity_documents": user_identity, 
                    "assistant_identity_documents": assistant_identity, 
                    "system_message": system_message_str
                    }
    

    return  input_update

from src.anubis.utils.dataset.quality import evaluate

async def invoke_agent(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """

    # model invocation
    avatar_model_with_tools = init_model(
        context = runtime.context,
        tools = [
            learn_information_about_the_user, 
            learn_information_about_yourself_through_text_from_the_user_as_a_memory, 
            recall_memories, 
            create_episodic_memory,
            ], 
        )

    logger.info(f"breakpoint")
    messages = state['messages']
    
    if isinstance(messages[0], SystemMessage):
        messages[0].content = state['system_message']
    else:
        messages = [SystemMessage(content = state['system_message'])] + messages

    response = await avatar_model_with_tools.ainvoke(input=messages)
    avatar_response_content = getattr(response, 'content')
    logger.info(f"Avatar Model Response: {avatar_response_content}")
    return {"messages":[response]}

    # agent invocation
    # avatar_model = init_model(
    #     context = runtime.context,
    # )

    # avatar = create_agent(model=avatar_model, system_prompt=state['system_message'], tools=[
    #         learn_information_about_the_user, 
    #         learn_information_about_yourself_through_text_from_the_user_as_a_memory, 
    #         recall_memories, 
    #         create_episodic_memory
    #         ],
    #         state_schema=GlobalState,
    #         )

    # messages = state['messages']
    # response = await avatar.ainvoke(input={"messages": messages})
    # avatar_response = response.get("messages", [])[-1]

    # logger.info(f"Avatar RESPONSE: {getattr(avatar_response, 'content')}")
    # recall_memories()<|python_end|>
    # result = {"messages": [avatar_response]}
    # if len(avatar_response.tool_calls) == 0:
    #     return Command(update = result, goto="__end__")
 
avatar_accessible_tools = [learn_information_about_the_user, learn_information_about_yourself_through_text_from_the_user_as_a_memory, 
                 recall_memories]

from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime

async def avatar_tools_condition(state:GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]) -> Literal["avatar_tool_node", '__end__']:
    recent_message = state['messages'][-1]
    if recent_message.tool_calls:
        for tool_call in recent_message.tool_calls:
            return "avatar_tool_node"
    else:
        return "__end__"
    
from langgraph.types import StreamWriter
async def avatar_tool_node(state: GlobalState, config: RunnableConfig, runtime:Runtime[GlobalContext]) -> Literal["load_consciousness"]:
    avatar_accessible_tools_dict = {
        "learn_information_about_the_user": learn_information_about_the_user, "learn_information_about_yourself_through_text_from_the_user_as_a_memory":learn_information_about_yourself_through_text_from_the_user_as_a_memory, 
        "recall_memories":recall_memories,
        "create_episodic_memory": create_episodic_memory
        }
    
    # avatar_accessible_tool_names = avatar_accessible_tools_dict.keys()
    
    message = state['messages'][-1]
    logger.info(f"breakpoint")    
    for tool_call in message.tool_calls:
            if tool_call['name'] in avatar_accessible_tools_dict:
                tool = avatar_accessible_tools_dict[tool_call['name']]
                tool_runtime = ToolRuntime(
                    state=state, 
                    config=config, 
                    context=runtime.context, 
                    store=runtime.store,
                    tool_call_id = tool_call['id'],
                    stream_writer=runtime.stream_writer
                    )
                logger.warning(f"tool_call: {tool_call}")
                tool_call["args"].update({"runtime":tool_runtime})

                await tool.ainvoke(tool_call['args'], runtime=tool_runtime)

    
# async def evaluate_response_quality()
    
# async def update_response_metadata()

from langgraph.prebuilt import ToolNode

# avatar_tool_node = ToolNode([
#     learn_information_about_the_user, 
#     learn_information_about_yourself_through_text_from_the_user_as_a_memory, 
#     create_episodic_memory, 
#     recall_memories
# ], handle_tool_errors=True)
    
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

# workflow.add_node("update_identity_tool_classification", update_identity_tool_classification)
# workflow.add_node("update_identity_tools", update_identity_tools)

anubis_workflow.add_node("load_consciousness", load_consciousness)
anubis_workflow.add_node("respond", invoke_agent)
anubis_workflow.add_node("avatar_tool_node", avatar_tool_node)

# workflow.add_node("evaluate_response_quality", evaluate_response_quality)

# workflow.add_node("update_response_metadata", update_response_metadata)

""" ANUBIS WORKFLOW EDGES """

anubis_workflow.add_edge(START, "load_consciousness")
anubis_workflow.add_edge("load_consciousness", "respond")

anubis_workflow.add_conditional_edges("respond", avatar_tools_condition, {'avatar_tool_node':'avatar_tool_node', "__end__":"__end__"})

anubis_workflow.add_edge("avatar_tool_node", "load_consciousness")
# anubis_workflow.add_edge("respond", END)

# workflow.add_edge("chat", "terms_and_services_content_moderation")

# workflow.add_conditional_edges("update_identity_tool_classification", update_identity_tool_condition, {"update_identity_tools": "update_identity_tools", "load_consciousness":"load_consciousness"})
# workflow.add_edge("update_identity_tools", "update_identity_tool_classification")


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
