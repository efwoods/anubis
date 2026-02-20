"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from src.subgraphs.vector_store_graph.utils import retrieval
from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.subgraphs.vector_store_graph.utils.utilities import format_docs, get_message_text

import re

import logging
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# Define the function that calls the model
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

from langgraph.runtime import Runtime

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.model import init_model

async def chunk_long_messages(human_message_list, configuration) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=0)
    # Chunk Long Messages
    chunked_message_list = []
    for message in human_message_list:
        message_token_len = count_tokens_approximately([message])
    if message_token_len > configuration.model_token_limit:
        text_chunks = text_splitter.split_text(message.get("content", ""))
        message = [HumanMessage(content=chunk) for chunk in text_chunks]
    if isinstance(message, list):
        chunked_message_list += message
    else:
        chunked_message_list += [message]
    
    human_message_list = chunked_message_list
    return human_message_list 

async def generate_query(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): GlobalConfiguration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    logging.info(f"XXXXX GENERATE QUERY NODE XXXX")

    # update the context from the state

    #  update the configuration if used as an argument

    if config:
        if config.get("metadata", None) != None:
            logger.warning(f"config: {config}")    
            user_id = config['metadata'].get("user_id", "")
            assistant_id = config['metadata'].get("assistant_id", "")

            if (user_id):
                runtime.context.user_ctx.user_id = user_id
                runtime.context.assistant_ctx.user_id = user_id

            if (assistant_id):
                runtime.context.assistant_ctx.assistant_id = assistant_id

    configuration = runtime.context.configuration

    messages = state['messages']

    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])

        # Verify the human input is not too large for the model
        if count_tokens_approximately(human_input) > (.8 * configuration.max_token_limit):
            # chunk the message in to a query
            human_input = await chunk_long_messages(human_input, configuration)
            # Summarize the long message into a single query
            system_message = "<Instruction>Please summarize this message into a query:</Instruction>"
            input = [SystemMessage(content=system_message), HumanMessage(content=human_input)]
            response = model.ainvoke(input=input)
            
            # Verify Successful Response
            if response:
                try:
                    assert(type(response) == AIMessage)
                    human_input = getattr(response, "content", "")
                    assert(human_input is not None)
                    assert(human_input is not "")
                except Exception as e:
                    logger.warning(f"Error extracting content from summarization response of large single message in query generation.")    
            else:
                logger.warning(f"No response while summarizing the chunked large single message. Using original chunked input.")
        return {"queries": [human_input]}
    
    else:
        
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

        # Create a model for invocation
        model_structured_output = init_model(
            configuration = configuration,
            response_format=SearchQuery
        )

        messages = state['messages']
        queries = state['queries']
        system_time = datetime.now(tz=timezone.utc).isoformat(),

        message_value = await prompt.ainvoke(
            {
                "messages": messages,
                "queries": "\n- ".join(queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        
        original_system_message = ""

        future_updated_system_message = "<Instructions>Using the summary and messages, create a brief single-sentence query that identifies the intent of all the messages and message summary and will retrieve documents that match this intent. Treat the intent of the summary and messages as the question to which the retrieved documents are the answer such that the question will match the content of the retrieved documents.</Instructions>\n<Summary>The following is the summary of the current conversation to date.</Summary>"
        
        future_updated_system_message_instruction_length = count_tokens_approximately([SystemMessage(content=future_updated_system_message)])

        if isinstance(message_value.messages[0], SystemMessage):
            original_system_message = message_value.messages[0]
            human_message_list = message_value.messages[1:]
            
            original_system_message_token_length = count_tokens_approximately(message_value.messages[0])
            
            max_tokens_minus_system_message_token_length = configuration.model_token_limit - original_system_message_token_length - future_updated_system_message_instruction_length
        else:
            human_message_list = message_value.messages
            max_tokens_minus_system_message_token_length = configuration.model_token_limit - future_updated_system_message_instruction_length

        

        # Attempt to trim messages to verify that messages need to be summarized
        retained_messages = trim_messages(
            messages=human_message_list, 
            max_tokens=max_tokens_minus_system_message_token_length, 
            token_counter=count_tokens_approximately, 
            strategy="last", 
            end_on=(HumanMessage)
        )
        # Identify if the messages need to be summarized
        if len(retained_messages) == 0:
            human_message_list = await chunk_long_messages(retained_messages, human_message_list)
            # re-attempt to trim messages
            retained_messages = trim_messages(
                messages = human_message_list, 
                max_tokens = max_tokens_minus_system_message_token_length, 
                token_counter=count_tokens_approximately,
                strategy="last",
                end_on=(HumanMessage)
            )
        assert(len(retained_messages != 0)) # messages length will be equal to or less than the length of the original message list because the messages have been chunked below the max token limit of the model;

        if len(retained_messages) == len(human_message_list):
            # no summary required; use the original message structure
            # create a generated query
            generated = cast(SearchQuery, await model_structured_output.ainvoke(message_value))

            return {
                "queries": [generated.query],
            }

        else: # Messages need to be summarized
            assert(len(retained_messages) > 0)
            assert(len(retained_messages) != len(human_message_list))
            # filter the retained messages and summarize the initial messages
            retained_message_id = getattr(retained_messages[0], "id", "")
            assert(retained_message_id is not None)
            assert(retained_message_id is not "")

            if not isinstance(retained_messages, list):
                original_retained_messages_token_length = count_tokens_approximately([retained_messages])
            else:
                original_retained_messages_token_length = count_tokens_approximately(retained_messages)
            
            message_id_list = [message.id for message in human_message_list]
            idx = message_id_list.index(retained_message_id) # find the index in the non-system message list

            summarization_messages = human_message_list[:idx]

            model = init_model(configuration=configuration) 
            
            # Save the optional system message and the retained messages
            if original_system_message is not "":
                if type(retained_messages) is list:
                    master_message_list = [original_system_message] + retained_messages
                else:
                    master_message_list = [original_system_message] + [retained_messages]
            else:
                if type(retained_messages) is list:
                    master_message_list = retained_messages
                else:
                    master_message_list = [retained_messages]
            
            summary_prompt = ""
            while len(summarization_messages) != len(retained_messages):
                # summarize the messages
                if summary_prompt is "":
                    summary_prompt = "<Instructions>Please summarize the following messages:</Instructions>"
                else:
                    summary_prompt = f"<Instructions> This is the current conversation summary to date. Please extend the summary prompt using the included messages. </Instructions>  {summary_prompt}. "
                summary_prompt_token_length = count_tokens_approximately([SystemMessage(content=summary_prompt)])

                if summary_prompt_token_length > configuration.model_token_limit*.8:
                    # summarize the summary prompt
                    input = [SystemMessage(content="<Instructions>Summarize the following message:</Instructions>"), HumanMessage(content=summary_prompt)]
                    response = await model.ainvoke(input=input)
                    if response:
                        assert(type(response) is AIMessage)
                        summary_prompt = getattr(response, "content", "")
                    else:
                        logger.warning(f"No response from the model; summarization prompt is greater than 80% of the length of the max token limit for the current model. Continuing with the current summarization prompt")
                    summary_prompt_token_length = count_tokens_approximately([SystemMessage(content=summary_prompt)])

                max_token_minus_summary_prompt_length_during_message_summarization = configuration.model_token_limit - summary_prompt_token_length

                retained_messages = trim_messages(
                    messages = summarization_messages, 
                    max_tokens=max_token_minus_summary_prompt_length_during_message_summarization, 
                    token_counter=count_tokens_approximately, 
                    strategy="last", 
                    end_on=(HumanMessage)
                )
                
                if len(retained_messages) == len(summarization_messages):
                    # summarize all the remaining messages
                    if type(retained_messages) is list:
                        input = [SystemMessage(content=system_message)] + retained_messages
                    else:
                        input = [SystemMessage(content=system_message)] + [retained_messages]
                        
                    response = model.ainvoke(input=input)
                    # Verify Successful Response
                    if response:
                        try:
                            assert(type(response) == AIMessage)
                            summary_prompt = getattr(response, "content", "")
                            assert(summary_prompt is not None)
                            assert(summary_prompt is not "")
                            
                        except Exception as e:
                            logger.warning(f"Error extracting content from summarization response during base case of message summary.")    
                    else:
                        logger.warning(f"No response while summarizing the chunked large single message. Continuing without message summary.")
                        
                        # continue; while loop will be broken; 
                elif len(retained_messages == 0):
                    # identify large messages and use a text splitter to chunk the messages
                    summarization_messages = await chunk_long_messages(summarization_messages, configuration)
                    # re trim messages
                    retained_messages = trim_messages(
                        messages = summarization_messages, 
                        max_tokens=max_token_minus_summary_prompt_length_during_message_summarization, 
                        token_counter=count_tokens_approximately, 
                        strategy="last", 
                        end_on=(HumanMessage)
                    )
                    if len(retained_messages) == len(summarization_messages):
                        # summarize the messages;  The messages no longer need to be trimmed
                        if type(retained_messages) is list:
                            input = [SystemMessage(content=system_message)] + retained_messages
                        else:
                            input = [SystemMessage(content=system_message)] + [retained_messages]
                            
                        response = model.ainvoke(input=input)

                        # Verify Successful Response
                        if response:
                            try:
                                assert(type(response) == AIMessage)
                                summary_prompt = getattr(response, "content", "")
                                assert(summary_prompt is not None)
                                assert(summary_prompt is not "")
                                
                            except Exception as e:
                                logger.warning(f"Error extracting content from summarization response during base case of message summary.")    
                        else:
                            logger.warning(f"No response while summarizing the chunked large single message. Continuing without message summary.")
                        
                        # continue; while loop will be broken; 

                    else:
                        assert(len(retained_messages) != 0)
                        assert(len(retained_messages) < len(summarization_messages))
                        message_id_list = [message.id for message in summarization_messages]
                        if type(retained_messages, list):
                            retained_message_id = getattr(retained_messages[0], "id", "")
                        else:
                            retained_message_id = getattr(retained_messages, "id", "")
                        assert(retained_message_id is not "")

                        idx = message_id_list.index(retained_message_id) # find the index in the non-system message list

                        # summarize the messages;  The messages no longer need to be trimmed
                        if type(retained_messages) is list:
                            input = [SystemMessage(content=system_message)] + retained_messages
                        else:
                            input = [SystemMessage(content=system_message)] + [retained_messages]
                            
                        response = model.ainvoke(input=input)

                        # Verify Successful Response
                        if response:
                            try:
                                assert(type(response) == AIMessage)
                                summary_prompt = getattr(response, "content", "")
                                assert(summary_prompt is not None)
                                assert(summary_prompt is not "")
                                
                            except Exception as e:
                                logger.warning(f"Error extracting content from summarization response during base case of message summary.")    
                        else:
                            logger.warning(f"No response while summarizing the chunked large single message. Continuing without message summary.")


                        # update the summarization messages
                        summarization_messages = summarization_messages[:idx]

                        # update the retained_messages
                        retained_messages = trim_messages(
                            messages = summarization_messages, 
                            max_tokens=max_token_minus_summary_prompt_length_during_message_summarization, 
                            token_counter=count_tokens_approximately, 
                            strategy="last", 
                            end_on=(HumanMessage)
                        )
                else:
                    logger.warning(f"retained_messasges are non zero and greater than the list of the messages where they initialized. This is a logical error and impossible.")
                
                         
            # message is summarized
            future_updated_system_message_no_summary = "<Instructions>Using the messages, create a brief single-sentence query that identifies the intent of all the messages and will retrieve documents that match this intent. Treat the intent of the summary and messages as the question to which the retrieved documents are the answer such that the question will match the content of the retrieved documents.</Instructions>"
            if summary_prompt is not "":
                # update the system message with the summary_prompt
                summary_prompt = re.sub(r"<Instructions>.*?</Instructions>", future_updated_system_message, summary_prompt)

                if type(master_message_list[0]) == SystemMessage:
                    original_system_message = master_message_list[0].content
                    new_system_message = original_system_message + " \n " + summary_prompt
                    master_message_list[0].content = new_system_message
                else:
                    master_message_list.insert(0, SystemMessage(content=summary_prompt))
            else:
                if type(master_message_list[0]) == SystemMessage:
                    master_message_list[0].content=future_updated_system_message
                else:
                    master_message_list.insert(0, SystemMessage(content=future_updated_system_message))

            final_message_list_token_length = count_tokens_approximately(master_message_list)
            if final_message_list_token_length > configuration.model_token_limit:
                system_message_token_length = count_tokens_approximately(master_message_list[0])
                human_message_token_length = final_message_list_token_length - system_message_token_length
                if system_message_token_length >= human_message_token_length:
                    master_message_list[0].content=future_updated_system_message_no_summary
                else:
                    messages_to_be_trimmed = master_message_list[1:]
                    retained_messages = trim_messages(
                        messages = messages_to_be_trimmed, 
                        max_tokens = configuration.model_token_limit,
                        token_counter=count_tokens_approximately,
                        strategy="last",
                        end_on=(HumanMessage, ToolMessage)
                    )
                    if len(retained_messages) == 0:
                        messages_to_be_trimmed = await chunk_long_messages(messages_to_be_trimmed, configuration)
                        retained_messages = trim_messages(
                            messages = messages_to_be_trimmed, 
                            max_tokens = configuration.model_token_limit,
                            token_counter=count_tokens_approximately,
                            strategy="last",
                            end_on=(HumanMessage, ToolMessage)
                        )
                    
                    system_message = master_message_list[0]
                    if type(retained_messages) == list:
                        master_message_list = [system_message] + retained_messages
                    else:
                        master_message_list = [system_message] = [retained_messages]

            generated = cast(SearchQuery, await model_structured_output.ainvoke(master_message_list))

            return {"queries": [generated.query]}

from src.subgraphs.vector_store_graph.utils.retrieval import (
    make_pg_vector
)

async def retrieve(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): GlobalConfiguration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    from langchain_core.messages import HumanMessage
    logging.info(f"XXXXX RETRIEVE NODE XXXX")
    logger.warning(f"runtime.context.user_ctx.user_id: {runtime.context.user_ctx.user_id}")
    logger.warning(f"runtime.context.assistant_ctx.assistant_id: {runtime.context.assistant_ctx.assistant_id}")
   

    human_message = state['messages'][-1]

    assert(isinstance(human_message, HumanMessage))

    retrieval_message = {"messages" : [human_message]}

    logger.info(f"{retrieval_message}")
    

    configuration = runtime.context.configuration

    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")
        
    if isinstance(runtime.context.assistant_ctx, dict):
        assistant_id = runtime.context.assistant_ctx.get("assistant_id", "")
    else:
        assistant_id = getattr(runtime.context.assistant_ctx, "assistant_id", "")

    memory_search = runtime.context.vector_store_memory_search_only

    if memory_search == "FALSE":
        filter_query = {
                "user_id": {"$eq": user_id},
                "assistant_id": {"$eq": assistant_id}, 
        }
                # "type": {"$ne": "memory"}
    else:
        filter_query = {
                "user_id": {"$eq": user_id},
                "assistant_id": {"$eq": assistant_id}, 
        }
                # "type": {"$eq": "memory"}

    
    vector_store = await make_pg_vector(configuration)

    logger.info(f"breakpoint")
    results = await vector_store.asimilarity_search_with_relevance_scores(
        query = state['queries'][-1],
        filter=filter_query,
    )
        # score_threshold=0.6

    retrieved_docs = [doc[0] for doc in results] # extract documents only

    logger.info(f"breakpoint")

    logger.info(f"Query: {state['queries'][-1]} | Docs: {len(retrieved_docs)}")
    logger.info(f"{retrieved_docs}")
    state['retrieved_docs'] = []
    return {"retrieved_docs": retrieved_docs}

# Define a new graph
builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")

# This compiles it into a graph you can invoke and deploy.
retrieval_graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
retrieval_graph.name = "RetrievalGraph"

__all__ = ["retrieval_graph"]
