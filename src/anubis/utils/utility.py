# src/anubis/utils/helper_functions

# Vectore store helper functions
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.anubis.utils.model import init_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from src.anubis.utils.context import GlobalContext

from langgraph.store.base import BaseStore

import logging
import re

from datetime import datetime
from datetime import timezone

from src.anubis.utils.prompts.system_prompts import TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
from typing import Optional

logger = logging.getLogger(__name__)


from pydantic import BaseModel
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

def add_queries(existing: Sequence[str], new:Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries for the vectorstore.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """

    query_list = list(existing) + list(new)
    query_list = list(set(query_list))
    if len(query_list) > 10:
        return query_list[-10:]
    else:
        return query_list


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    # metadata = doc.metadata or {}
    # meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    # if meta:
        # meta = f" {meta}"
    # f"<document{meta}>\n{doc.page_content}\n</document>"
    return f"<document>\n{doc.page_content}\n</document>"


def format_docs(docs: list[Document] | None) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

############################  Doc Indexing State  #############################
import uuid
from typing import Union, Any, Literal

from langgraph.store.base import SearchItem

def reduce_docs(
    existing: Sequence[Document] | None,
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        Sequence[SearchItem],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            elif isinstance(item, SearchItem):
                logger.info("breakpoint")
                page_content = getattr(item,'value', {}).get("document", {}).get("kwargs", {}).get("page_content", "")
                document_metadata = getattr(item, 'value', {}).get("document", {}).get("kwargs", {}).get("metadata", {})
                document = Document(page_content=page_content, metadata=document_metadata)
                coerced.append(document)
            else:
                coerced.append(item)
        return coerced
    return existing or []


from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from src.anubis.utils.context import GlobalContext


async def extract_user_id_assistant_id(config: RunnableConfig):
    user_state = {}
    assistant_state = {}

    user_id = config.get("configurable",{}).get("user_id", '')

    if user_id != '':
        user_state.update({"user_id": user_id})
    else:
        """anonymous_user_id is 'str(uuid5(NAMESPACE_URL, 'anonymous_user_id"""
        user_state.update({"user_id":'9977df19-9ceb-5f87-a130-55f6a6282069'})
        
    assistant_id = config.get("configurable", {}).get("assistant_id", "")

    if assistant_id != "":
        assistant_state.update({"assistant_id":assistant_id})
    else:
        raise Exception("Assistant does not have an id from the context. Provide an assistant_id in config['configurable']['assistant_id'].")

    return user_state, assistant_state

async def configure_assistant_context(config: RunnableConfig, store: BaseStore):
        user_id, assistant_id = await extract_user_id_assistant_id(config)
        
        namespace=(user_id, assistant_id, "assistant_ctx")
        ai_context_item = await store.aget(namespace, key=assistant_id)
        logger.info(f"ai_context_item: {ai_context_item}")

        # Load/UPDATE AI SELF IDENTITY
        logger.info("item object breakpoint")

        # get the current assistant context as a dict

        configurable_assistant_ctx = config.get("configurable", {}).get("assistant_ctx", None)

        if configurable_assistant_ctx is not None:
            if ai_context_item is not None:
                for key, value in configurable_assistant_ctx:
                    if (value != "" and value != None) and key != "metadata":
                        ai_context_item.value['assistant_ctx'].update({key: value})
                if configurable_assistant_ctx.get("metadata", None) is not None:
                    update_metadata = configurable_assistant_ctx.get("metadata")
                    ai_context_item.value['assistant_ctx']['metadata'].update(update_metadata)                 

                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":ai_context_item.value["assistant_ctx"]})
            else:
                init_assistant_ctx = {
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "name":configurable_assistant_ctx.get("name", ""),
                    "description":configurable_assistant_ctx.get("description", ""),
                    "metadata": configurable_assistant_ctx.get("metadata", {})
                }
                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":init_assistant_ctx})
        else:
            if ai_context_item is None:
                init_assistant_ctx = {
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "name": "",
                    "description": "",
                    "metadata": {}
                }
                await store.aput(namespace, key=assistant_id, value={"assistant_ctx":init_assistant_ctx})

        ai_context_item = await store.aget(namespace, key=assistant_id)

        return ai_context_item


async def image_to_text(target_image_url: str, 
                        reference_image_url: Optional[str] = None, 
                        ):
    """
    Convert an image of a target to text.
    Describe the target individual to the best of your ability. 
    args:
        target_image_url: base64 encoded string or a url to an image to describe.
        reference_image_url (Optional[str]): base64 encoded string or a url to an image. 
            Expected to only have a single individual. Used to identify the target to describe in the target image.
    
    Returns: 
        description (str): This is the description of the target with respect to the individual. The description is of the target from the FIRST PERSON PERSPECTIVE.
    """

    if reference_image_url is not None:
        if "." in reference_image_url:
            # url
            reference_message = {"type": "image_url", "image_url":{"url":reference_image_url}}
        else:
            # base 64 encoding
            reference_message = {"type": "image_url", "image_url":{"url":f"data:image/jpeg;base64,{reference_image_url}"}}

    if "." in target_image_url:
        target_message = {"type": "image_url", "image_url": {"url":target_image_url}}
    else:
        target_message = {"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{target_image_url}"}}

    # Compile the message
    content = [reference_message, target_message] if reference_image_url is not None else [target_message]
    system_message = [SystemMessage(content=TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION)]
    human_message = [{"role": "user", "content": content}]

    messages = system_message + human_message

    # TODO: response_metrics_aggregation
    model = init_model()

    response = await model.ainvoke(input=messages)

    description  = response.content

    return description 


############################  CHUNK LONG MESSAGES  #############################

async def chunk_long_messages(human_message_list, context) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=0)
    # Chunk Long Messages
    chunked_message_list = []
    for message in human_message_list:
        message_token_len = count_tokens_approximately([message])
    if message_token_len > context.model_token_limit:
        text_chunks = text_splitter.split_text(getattr(message, "text", ""))
        message = [HumanMessage(content=[{'type':'text', 'text':chunk}]) for chunk in text_chunks]
    if isinstance(message, list):
        chunked_message_list += message
    else:
        chunked_message_list += [message]
    
    human_message_list = chunked_message_list
    return human_message_list 


############################  Summarize Messages  #############################

from typing import Optional


""" YOUTUBE HELPER FUNCTIONS """
import yt_dlp
import os
import re


async def download_transcript(url: str, lang: str = "en", auto_subs: bool = True, output_dir: str = ".") -> str:
    """
    Download transcript/subtitles from a YouTube video.

    Args:
        url: YouTube video URL
        lang: Language code (e.g., 'en', 'es', 'fr')
        auto_subs: Fall back to auto-generated subtitles if manual not found
        output_dir: Directory to save files

    Returns:
        Path to the downloaded subtitle file
    """
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": auto_subs,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
    }

    async with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")

    # Find the downloaded .vtt file
    for f in os.listdir(output_dir):
        if f.endswith(".vtt"):
            return os.path.join(output_dir, f)

    raise FileNotFoundError(f"No subtitle file found for '{title}'. "
                            "Try listing available languages first.")


def parse_vtt(vtt_path: str) -> str:
    """
    Parse a .vtt subtitle file into clean plain text.
    Removes timestamps, cue settings, and duplicate lines.
    """
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove WEBVTT header and NOTE blocks
    content = re.sub(r"WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    content = re.sub(r"NOTE.*?\n\n", "", content, flags=re.DOTALL)

    lines = content.splitlines()
    clean_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        # Skip timestamps, cue IDs, and empty lines
        if not line or "-->" in line or re.match(r"^\d+$", line):
            continue
        # Remove inline tags like <00:00:01.000>, <c>, </c>
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        # Deduplicate consecutive repeated lines (common in auto-subs)
        if line and line not in seen:
            clean_lines.append(line)
            seen.add(line)
        elif line in seen:
            seen = {line}  # reset window to allow repeated words in new context

    return " ".join(clean_lines)


def list_available_subtitles(url: str) -> dict:
    """List all available subtitle languages for a video."""
    ydl_opts = {"quiet": True, "skip_download": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    manual = info.get("subtitles", {})
    auto = info.get("automatic_captions", {})

    print(f"\nVideo: {info.get('title')}")
    print(f"Manual subtitles:    {list(manual.keys()) or 'None'}")
    print(f"Auto-generated subs: {list(auto.keys()) or 'None'}")

    return {"manual": manual, "automatic": auto}


# This needs to be in-memory rather than using file storage
def get_transcript(url: str, lang: str = "en", save_txt: bool = True) -> str:
    """
    High-level function: download + parse transcript, return plain text.

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

    return transcript


# --- Example usage ---
# if __name__ == "__main__":
#     VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

#     # 1. Check what's available
#     list_available_subtitles(VIDEO_URL)

#     # 2. Download and parse transcript
#     text = get_transcript(VIDEO_URL, lang="en", save_txt=True)

#     print("\n--- Transcript Preview ---")
#     print(text[:500])