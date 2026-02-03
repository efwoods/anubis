""" Agent SubGraph Tools """
import uuid
import logging
from typing import List, Annotated

from langchain.tools import tool, ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg
from langchain.messages import AIMessage, HumanMessage

from langgraph.store.base import BaseStore

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

logger = logging.getLogger(__name__)

from datetime import datetime, timezone

@tool
async def health_check(runtime: ToolRuntime[GlobalContext]):
    """Call this health check to determine if tools can be called at all.

    Args:
        runtime (ToolRuntime[GlobalContext]): tool runtime of the current state and context 

    Returns:
        _type_: _description_
    """
    return {"messages": AIMessage(content="success")}

# @tool
# async def health_check(runtime: ToolRuntime[GlobalContext]) -> str:
#     """Call this tool specifically when the user asks "are you okay?"
    
#     Args:
#         runtime (ToolRuntime[GlobalContext]): ToolRuntime
        
#     Returns:
#         str: The model's response to the updated prompt
#     """
#     logger.info("Health check started")
    
    
#     # Extract configuration and context
#     ctx = runtime.context
#     config = runtime.context.configuration
    

#     # Update identity with new information [Example]
#     await ctx.update_identity({
#         "health_status": {
#             "last_check": datetime.now(tz=timezone.utc).isoformat(),
#             "status": "operational",
#             "metrics": {
#                 "uptime": "99.9%",
#                 "response_time": "50ms"
#             }
#         }
#     })
    
#     from src.anubis.utils.model import invoke_model_core
    
    
#     # Create TEMPORARY updated system prompt with health check status
#     temp_system_prompt = f"""

# TOOL EXECUTION UPDATE:
# The health_check tool is currently running. System status:
# - Runtime: Operational
# - Memory: Normal
# - Tool execution: In progress
# - Status: All systems nominal

# Please answer the following questions:

# What is the time?
# Where are you?
# What is your name?
# Who is the president?
# Who are you talking to?

# Please acknowledge the health check results and provide a brief status summary to the user while also answering the questions.
# """
    
#     # Temporarily update the temporary system prompt within the context
#     runtime.context.temporary_system_prompt_update = temp_system_prompt
    
#     # Initialize model with tools
#     # [] tools are not being used in this invocation
#     response = await invoke_model_core(
#         runtime.state, 
#         runtime, 
#         []
#     )
#     logger.info(f"Health check completed with model response: {response}")

#     if hasattr(response, 'content'):
#         response_text = response.content
#     else:
#         response_text = str(response)

#     return response_text
    
    
@tool
async def add_to_vectorstore_subgraph(runtime: ToolRuntime[GlobalContext])-> AIMessage:
    """THIS TOOL IS CALLED WHEN THE USER REQUESTS THE ATTACHED MEDIA OR RAW TEXT TO BE ADDED TO THE VECTORSTORE"""
    from src.subgraphs.vector_store_graph.index_graph import index_graph
    from src.subgraphs.process_media_graph import process_media_graph

    logger.info('ADD_TO_VECTORESTORE TOOL CALLED XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    result = process_media_graph(runtime.state, runtime.context)
    if isinstance(result, AIMessage): # This needs to update to call the invoke_model node to prompt for media. 
        return AIMessage

    


    # convert state
    logger.info(f"{runtime.state} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    messages = runtime.state['messages']
    logger.info(f"MESSAGES XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages}")
    
    logger.info(f" LEN` MESSAGE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {len(messages)}")

    recent_msg = messages[-1]
    # logger.info(f" RECENT MESSAGE messages[-1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[-1]}")
    # logger.info(f" RECENT MESSAGE messages[0] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[0]}")
    # logger.info(f" RECENT MESSAGE messages[1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[1]}")
    # [logger.info(f"{message}") for message in messages['content']['text']]
    # count = 0
    # for message in messages:
    #     logger.info(f"{count}: {message}")
    #     count+=1
    # logger.info(f"penultimate has data: messages[-2]: {messages[-2]}")

    # logger.info(f" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {isinstance(recent_msg, HumanMessage)}")
    recent_message = messages[-2]
    recent_docs: List[Document] = []
    if isinstance(recent_message, HumanMessage):
        # logger.info(f"CONTENT XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {recent_msg.content}")
        content = recent_message.content
        logger.info(f"len content: {len(content)}")
        # logger.info(f"content: {content}")
        if isinstance(content, list):
            if len(content) > 1:
                logger.warning(f"LEN OF content 1 or 0 not handled; uploads to vectorstore successfully but no content")
                text_from_human = content[0]
                image_content_dict = content[-1]
                # logger.warning(f"content[-1]:{content[-1]}")
                description_doc = await process_media(image_content_dict)
                logger.info(f"description_doc: {description_doc}")
                recent_docs.append(description_doc)
                logger.info(f"recent_docs: {recent_docs}")
            else: 
                logger.warning(f"content list length is < 1; only text no media inferred.")
                
                return AIMessage(content="Please attach media") # update with custom no media response from llm return to model node with prompt to generate


    index_input = {"docs": recent_docs}
    logger.info(f"index_input XXXXXXXXXXXXXXXXXXXXXXXXX {index_input}")
    
    result = await index_graph.ainvoke(index_input, runtime) # user id is bypassed for testing

    # logger.info(f"RESULT XXXXXXXXXXXXXXXXXXXXXXXXX {result}")
    # # Return results of subgraph
    # ai_content = result.get('messages', [-1]).content if result.get('messages') else "Indexed successfully"
    return AIMessage(content="added to vectorstore successfully")




@tool
async def retrieve_from_vectorstore_subgraph(runtime: ToolRuntime[GlobalContext]) -> AIMessage:
    """Generates the correct query to search the vectorstore for relevant documents to the Human query.

    Args:
        runtime (ToolRuntime[GlobalContext]): tool runtime context

    Returns:
        AIMessage: _description_
    """
    from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph
    # messages = runtime.state['messages']
    response = await retrieval_graph.ainvoke(runtime.state, {"configurable": {"user_id": "test_user_1234"}})
    logger.info(f"RETRIEVAL RESPONSE: {response}")
    ai_response = response['messages'][-1]
    
    return ai_response

@tool
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


""" Vector Store SubGraph Tools """
""" Process Media SubGraph Tools """

