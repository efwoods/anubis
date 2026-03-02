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

# TODO: YOUTUBE IDENTITY UPDATER
# TODO: USE MEMORY RATHER THAN FILE SYSTEM
from src.anubis.utils.helper_functions import download_transcript, parse_vtt

@tool
def get_transcript(url: str, lang: str = "en", save_txt: bool = False) -> str:
    """
    Use this tool when a user suggests there is a youtube link 
    or the link included in the message contains the word 'youtube'. 
    Use this to download the text from the link.


    Args:
        url: YouTube video URL
        lang: Subtitle language code
        save_txt: If True, saves plain text transcript to disk

    Returns:
        Transcript as a plain text string
    """
    print(f"Downloading subtitles for: {url}")
    vtt_path = download_transcript(url, lang=lang)

    print(f"Parsing: {vtt_path}")
    transcript = parse_vtt(vtt_path)

    if save_txt:
        txt_path = vtt_path.rsplit(".", 2)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved to: {txt_path}")

    print(f"TRANSCRIPT: {transcript}")

    return transcript

# TODO: IMAGE URL IDENTITY UPDATER

# TODO: TEXT WEBPAGE URL IDENTITY UPDATER
# TODO: TEST CONFIGURATION & STORE CAPABILITY
from langchain_openai import ChatOpenAI
from src.anubis.utils.prompts.system_prompts import FACT_FORMATTING_STRING_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from uuid import uuid4, uuid5, NAMESPACE_URL
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

@tool
async def update_identity_via_text_content_url(url: str, runtime: ToolRuntime):
    
    # Extract 
    user_id = runtime.config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
    assistant_id = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_id", "")
    assistant_name = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_name", "")
    
    model = ChatOpenAI(
        model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8",
        base_url = "https://api.llama.com/compat/v1/",
        api_key = "LLM|2160921497745669|2M_zeRPt10hKzWlJw39C_P-UIHM",
        temperature = 0.1
    )
    system_message = SystemMessage(content = FACT_FORMATTING_STRING_PROMPT.format(assistant_name=assistant_name))

    filename = url
    filename_uuid5 = uuid5(NAMESPACE_URL, url)

    namespace = (user_id, assistant_id, "identity", filename_uuid5)
    
    loader = UnstructuredLoader(web_url=url)
    docs = loader.load()

    total_number_of_documents = len(docs)
    number_of_documents_to_format = 5
    for index in range(0, total_number_of_documents, number_of_documents_to_format):
        retrieved_data = "\n\n".join([doc.page_content for doc in docs[index:index + number_of_documents_to_format]])
        human_message = HumanMessage(content=retrieved_data)
        messages = [system_message, human_message]
        extracted_response = await model.ainvoke(input=messages)
        extracted_response_document = Document(page_content=extracted_response)
        key = str(uuid4())
        extracted_response_document.metadata.update({"filename":filename, "user_id":user_id, "assistant_id": assistant_id, "filename_uuid5":filename_uuid5, "key":key})
        document_data_json = extracted_response_document.to_json()

        # Upload the document to the store
        await runtime.store.aput(namespace, key=key, value={"document":document_data_json})

# TODO: IN-MESSAGE IDENTITY UPDATER (TEXT)


