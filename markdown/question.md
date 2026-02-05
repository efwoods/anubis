why is there a runtime.store on the first graph but not on the second graph? the first graph is accessible from the chat window of the langgraph dev api server.

FIRST GRAPH:
# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

# from src.subgraphs.conversational_memory_graph.graph import agent_graph

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext
from src.anubis.utils.configuration import GlobalConfiguration

from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_agent

from src.anubis.utils.model import init_model

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.nodes import (
    invoke_agent
)

# Build minimal graph: START -> agent -> END
workflow = StateGraph(state_schema = GlobalState, context_schema = GlobalContext)

# Add single node (your input/output)
# workflow.add_node("call_router", call_router)
workflow.add_node("invoke_agent", invoke_agent)

# Edges
# workflow.add_edge(START, "invoke_agent")
workflow.add_edge(START, "invoke_agent")

workflow.add_edge("invoke_agent", END)

graph = workflow.compile()
graph.name = "Anubis"

__all__ = ["graph"]



async def invoke_agent(state: GlobalState, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    logger.info(f"INVOKE AGENT NODE ")

    # test_store = await runtime.store.alist_namespaces()
    
    config = runtime.context.configuration # Loads env vars automatically

    model = init_model(
        config.provider_model,
        config.llama_api_base_url,
        config.llama_api_key,
        tools,
        config.dev
    )

    # Retrieve documents for the query
    from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

    human_message = state['messages'][-1]

    assert(isinstance(human_message, HumanMessage))
    
    retrieval_message = {"messages" : [human_message]}

    new_state_retrieved_docs = await retrieval_graph.ainvoke(retrieval_message, context=runtime.context)
    
    # populate the relevant documents with a new state
    state['retrieved_docs'] = new_state_retrieved_docs['retrieved_docs']

    # Vectorstore Retrieved Docments
    retrieved_docs = format_docs(state.get('retrieved_docs', []))

    prompt_builder = DynamicPromptBuilder()

    # TODO: Update the assistant context from the state

    # Load the current assistant context for prompt injection
    ai_context = runtime.context.assistant_ctx.to_dict()

    # TODO: Update the user context from the state

    # Load the current user context from prompt injection
    user_ctx = runtime.context.user_ctx.to_dict()

    system_time = datetime.now(tz=timezone.utc).isoformat()

    temporary_system_prompt_update = runtime.context.temporary_system_prompt_update

    populated_template = prompt_builder.build_prompt(
        ai_context=ai_context,
        user_context=user_ctx, 
        retrieved_docs=retrieved_docs,
        system_time = system_time,
        temporary_message=temporary_system_prompt_update,
    )

    logger.info(f"populated_template: {populated_template}")

    # clear the temporary injected prompt
    runtime.context.temporary_system_prompt_update = ""

    # prepend system message
    messages_input = populated_template.messages + state["messages"]
    
    logger.info(f"messages_input: {messages_input}")
    
    # Call the model
    response = await model.ainvoke(input=messages_input)

    logger.info(f"AGENT RESPONSE: {response}")
    result = {"messages": [response]}
    return result

SECOND GRAPH

# src/subgraphs/process_media_graph/graph.py

"""
Given a list of messages: 
Extracts zero or more media from the most recent message.
Determine the media type.
Convert the media into text.
Creates a Document object from the text.
Returns a List of Document Objects for further processing in other subgraphs.
"""

from langgraph.graph import StateGraph, START, END
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext

from src.subgraphs.process_media_graph.utils.nodes import (
    extract_media_from_message, 
    convert_media_list_to_text_document,
)

from src.subgraphs.vector_store_graph.index_graph import index_docs

from src.subgraphs.process_media_graph.utils.nodes import process_uploaded_files

# Define the Graph & Context
workflow = StateGraph(state_schema=GlobalState, context_schema=GlobalContext)

# Add Nodes
workflow.add_node("process_uploaded_files", process_uploaded_files)
workflow.add_node("convert_media_list_to_text_document", convert_media_list_to_text_document)
workflow.add_node("index_docs", index_docs)

# Define Edges
workflow.add_edge(START, "process_uploaded_files")
workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")
workflow.add_edge("convert_media_list_to_text_document", "index_docs")


process_media_graph_api_endpoint = workflow.compile()
process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"

__all__ = ["process_media_graph_api_endpoint"]


async def process_uploaded_files(
    state: GlobalState, 
    runtime: Runtime[GlobalContext], 
    store: BaseStore
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects into standardized media format.
    This is the entry point for direct file uploads (not from messages).
    """
    
    logger.info(f"Process uploaded files NODE")
    

    media_files = state.get('media_files', [])
    
    if not media_files:
        logger.info("No media files to process")
        return {"media_list": []}
    
    logger.info(f"Processing {len(media_files)} uploaded files")
    
    media_list = []
    
    for file_data in media_files:
        try:
            # Extract file info
            filename = file_data.get('filename', 'unknown')
            content_type = file_data.get('content_type', '')
            file_bytes = file_data.get('content')  # Raw bytes
            user_id = file_data.get("user_id")
            assistant_id = file_data.get("assistant_id")
            reference_image = file_data.get("reference_image")
            reference_audio = file_data.get("reference_audio")
            
            logger.info(f"Processing file: {filename} ({content_type})")
            
            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                # Convert image to base64
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "image",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_image": reference_image
                    }
                })
            
            elif content_type.startswith('audio/'):
                # Handle audio files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "audio",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_audio": reference_audio
                    }
                })
            
            elif content_type.startswith('video/'):
                # Handle video files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "video",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id
                    }
                })
            
            elif content_type in ['text/plain', 'application/json', 'text/markdown']:
                # Handle text files
                text_content = file_bytes.decode('utf-8')
                media_list.append({
                    "type": "text",
                    "content": text_content,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id
                    }
                })
            
            elif content_type == 'application/pdf':
                # Handle PDFs
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "pdf",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id
                    }
                })
            
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                continue
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Converted {len(media_list)} files to media format")
    
    return {
        "media_list": media_list,
        "media_files": []  # Clear after processing
    }

API OF THE SECOND GRAPH:
# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import asyncio

from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext

import logging
logger = logging.getLogger(__name__)

# Preload audio to text processor [this needs a startup in a lifecycle call]
 
from contextlib import asynccontextmanager

from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext
from langgraph.store.memory import InMemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    logger.info("Application startup: Preloading Whisper model...")
    global context 
    global in_memory_store

    try:
        # Initialize context / configuration
        context = GlobalContext()
        in_memory_store = InMemoryStore()

        # Create pipeline for audio transcription
        from src.subgraphs.process_media_graph.utils.helper_functions import get_whisper_pipeline
        # Call the function to trigger @lru_cache and load model into memory
        pipe = get_whisper_pipeline()
        
        logger.info("✓ Whisper model preloaded and cached successfully")
        logger.info(f"  - Model: openai/whisper-large-v3")
        logger.info(f"  - Device: {pipe.device}")
        logger.info(f"  - Ready to process audio requests")

        # create a context
        context = GlobalContext()
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ CRITICAL: Failed to preload Whisper model: {e}", exc_info=True)
        logger.error("=" * 60)
        # Decide if you want to fail fast or continue
        raise  # Uncomment to prevent startup if model loading fails
    
    yield  # Application runs here
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down application...")

app = FastAPI(
    title="Media Processing API",
    description="LangGraph-based media processing with Whisper audio transcription",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/hello")
def test_hello_world():
    logger.info(f"HELLO WORLD ENTRY")
    return {"Hello": "World"}

@app.post("/upload-media")
async def upload_media(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="test_user_1234"),
    assistant_id: str = Form(default="default_assistant"),
    reference_audio: bool = False,
    reference_image: bool = False
):
    # Context user_id, assistant_id
    logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
    """
    Upload one or more media files for processing and indexing.
    
    - **files**: One or more files to process
    - **user_id**: User identifier
    - **assistant_id**: Assistant identifier
    """
    try:
        # update the context:
        context.user_ctx.user_id = user_id
        context.assistant_ctx.assistant_id = assistant_id
        context.assistant_ctx.user_id = user_id

        # Read all uploaded files
        media_files = []
        for file in files:
            content = await file.read()
            media_files.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": reference_audio,
                "reference_image": reference_image
            })
        
        # Import graph here to avoid circular imports
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
        # Prepare input state
        initial_state = {
            "media_files": media_files,
        }
        
        

        # Prepare context/config
        # config = {
        #     "configurable": {
        #         "user_id": user_id,
        #         "assistant_id": assistant_id,
        #         # Add other configuration as needed
        #     }
        # }

        runtime_context = {"context": context}
        
        # Invoke the graph
        result = await process_media_graph_api_endpoint.ainvoke(initial_state, context=context)
        
        # Extract indexed documents info
        indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "files_processed": len(files),
                "documents_indexed": len(indexed_docs),
                "filenames": [f.filename for f in files],
                "message": "Media processed and indexed successfully"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )

@app.post("/process-media-json")
async def process_media_json(
    media_list: List[dict],
    user_id: str = "test_user_1234",
    assistant_id: str = "default_assistant", 
    reference_audo: bool = False, 
    reference_image: bool = False
):
    """
    Process media from JSON payload (for pre-encoded base64 data).
    
    Expected format:
    {
        "media_list": [
            {
                "type": "image",
                "data": "base64_encoded_data",
                "metadata": {...}
            }
        ],
        "user_id": "user123",
        "assistant_id": "assistant456"
    }
    """
    try:
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
        initial_state = {
            "media_list": media_list,   
        }
        
        # config = {
        #     "configurable": {
        #         "user_id": user_id,
        #         "assistant_id": assistant_id,
        #     }
        # }
        
        result = await process_media_graph_api_endpoint.ainvoke(initial_state)
        
        indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
        return {
            "status": "success",
            "media_items_processed": len(media_list),
            "documents_indexed": len(indexed_docs)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
