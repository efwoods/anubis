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
from src.anubis.utils.configuration import GlobalConfiguration

from langgraph.store.base import BaseStore

import logging
import re

from datetime import datetime
from datetime import timezone

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

from langgraph_sdk.schema import Item

def reduce_docs(
    existing: Sequence[Document] | None,
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
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
            elif isinstance(item, Item):
                document = Document(page_content = item['value'].get("document", {}).get("kwargs", {}).get("page_content", ""))
                document.metadata.update({"metadata": item['value'].get("document", {}).get("kwargs", {}).get("metadata", {})})
                coerced.append(document)
            else:
                coerced.append(item)
        return coerced
    return existing or []


from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from src.anubis.utils.context import GlobalContext


async def extract_user_id_assistant_id(config: RunnableConfig):
       
    user_id = config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
    if user_id == "":
        user_id = config.get("configurable",{}).get("user_id","")


    assistant_id = config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_id", "")
    if assistant_id == "":
        assistant_id = config.get("configurable",{}).get("assistant_id","")

    # Ensure valid user_id and assistant_id
    if user_id == "" and (config.get("metadata", {}).get("from_studio") is True):
        user_id = "Anubis_from_studio_" + assistant_id
    user_id = "".join(user_id.strip())    
    assistant_id = "".join(assistant_id.strip())

    return user_id, assistant_id

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



############################  CHUNK LONG MESSAGES  #############################

async def chunk_long_messages(human_message_list, configuration) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=0)
    # Chunk Long Messages
    chunked_message_list = []
    for message in human_message_list:
        message_token_len = count_tokens_approximately([message])
    if message_token_len > configuration.model_token_limit:
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

async def summarize_messages(
        messages, 
        configuration, 
        future_updated_system_message, 
        future_updated_system_message_failsafe, 
        system_message_instruction_single_message, 
        query_l: Optional[list],
        query_generation_mode: bool=True
        ) -> list:
    """
    This function will accept a list of messages, 
    retain messages according to the model context window length,
    and create a summary in the system message prompt of the current conversation

    Args:
        messages (Sequence[HumanMessage | AIMessage | SystemMessage]): state['messages']
        configuration (GlobalConfiguration): environment variables
        future_updated_system_message (str): system message instructions
        future_updated_system_message_failsafe (str): system message instructions in the case that the system message instructions are too long
        system_message_instruction_single_message (_type_): system message instructions for the case where there is a single message.
    
    Returns: list(SystemMessage, HumanMessage|AIMessage|ToolMessage)

    """
    logger.info("summarize_messages breakpoin")

    if query_generation_mode:

        if len(messages) == 1:
            # It's the first user question. We will use the input directly to search.
            human_input = get_message_text(messages[-1])

            # Verify the human input is not too large for the model
            if count_tokens_approximately(human_input) > (.8 * configuration.model_token_limit):
                # chunk the message into a query
                human_input = await chunk_long_messages(human_input, configuration)
                # Summarize the long message into a single query
                system_message = system_message_instruction_single_message
                input = [SystemMessage(content=[{'type': 'text', 'text': system_message}]), HumanMessage(content=[{'type': 'text', 'text': human_input}])]
                model = init_model(configuration)
                response = model.ainvoke(input=input)


                # Verify Successful Response
                if response:
                    try:
                        assert(type(response) == AIMessage)
                        human_input = getattr(response, "content", "")
                        assert(human_input is not None)
                        assert(human_input != "")
                    except Exception as e:
                        logger.warning(f"Error extracting content from summarization response of large single message in query generation.")    
                else:
                    logger.warning(f"No response while summarizing the chunked large single message. Using original chunked input.")

            # system_message = system_message_instruction_single_message
            # input = [SystemMessage(content=[{'type': 'text', 'text': system_message}]), HumanMessage(content=[{'type': 'text', 'text': human_input}])]
            # model = init_model(configuration)
            # response = await model.ainvoke(input=input)

            logger.info(f"BREAKPOINT")

            return {"queries": [human_input]}

        else:

            # Feel free to customize the prompt, model, and other logic!
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", configuration.query_system_prompt),
                    ("placeholder", "{messages}"),
                ]
            )

            message_value = await prompt.ainvoke(
                {
                    "messages": messages,
                    "queries": "\n- ".join(query_l),
                    "system_time": datetime.now(tz=timezone.utc).isoformat(),
                },
            )

            original_system_message = ""

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
                return message_value

            else: # Messages need to be summarized
                assert(len(retained_messages) > 0)
                assert(len(retained_messages) != len(human_message_list))
                # filter the retained messages and summarize the initial messages
                retained_message_id = getattr(retained_messages[0], "id", "")
                assert(retained_message_id is not None)
                assert(retained_message_id != "")

                # if not isinstance(retained_messages, list):
                #     original_retained_messages_token_length = count_tokens_approximately([retained_messages])
                # else:
                #     original_retained_messages_token_length = count_tokens_approximately(retained_messages)

                message_id_list = [message.id for message in human_message_list]
                idx = message_id_list.index(retained_message_id) # find the index in the non-system message list

                summarization_messages = human_message_list[:idx]

                model = init_model(configuration=configuration) 

                # Save the optional system message and the retained messages
                if original_system_message != "":
                    if type(retained_messages) == list:
                        master_message_list = [original_system_message] + retained_messages
                    else:
                        master_message_list = [original_system_message] + [retained_messages]
                else:
                    if type(retained_messages) == list:
                        master_message_list = retained_messages
                    else:
                        master_message_list = [retained_messages]

                summary_prompt = ""
                while len(summarization_messages) != len(retained_messages):
                    # summarize the messages
                    if summary_prompt == "":
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
                        if type(retained_messages) == list:
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
                                assert(summary_prompt != "")

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
                            if type(retained_messages) == list:
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
                                    assert(summary_prompt != "")

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
                            assert(retained_message_id != "")

                            idx = message_id_list.index(retained_message_id) # find the index in the non-system message list

                            # summarize the messages;  The messages no longer need to be trimmed
                            if type(retained_messages) == list:
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
                                    assert(summary_prompt != "")

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
                if summary_prompt != "":
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
                        master_message_list[0].content=future_updated_system_message_failsafe
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
    
    return master_message_list


""" YOUTUBE HELPER FUNCTIONS """
# import yt_dlp
# import os
# import re


# def download_transcript(url: str, lang: str = "en", auto_subs: bool = True, output_dir: str = ".") -> str:
#     """
#     Download transcript/subtitles from a YouTube video.

#     Args:
#         url: YouTube video URL
#         lang: Language code (e.g., 'en', 'es', 'fr')
#         auto_subs: Fall back to auto-generated subtitles if manual not found
#         output_dir: Directory to save files

#     Returns:
#         Path to the downloaded subtitle file
#     """
#     ydl_opts = {
#         "skip_download": True,
#         "writesubtitles": True,
#         "writeautomaticsub": auto_subs,
#         "subtitleslangs": [lang],
#         "subtitlesformat": "vtt",
#         "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
#         "quiet": True,
#         "no_warnings": False,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         title = info.get("title", "video")

#     # Find the downloaded .vtt file
#     for f in os.listdir(output_dir):
#         if f.endswith(".vtt"):
#             return os.path.join(output_dir, f)

#     raise FileNotFoundError(f"No subtitle file found for '{title}'. "
#                             "Try listing available languages first.")


# def parse_vtt(vtt_path: str) -> str:
#     """
#     Parse a .vtt subtitle file into clean plain text.
#     Removes timestamps, cue settings, and duplicate lines.
#     """
#     with open(vtt_path, "r", encoding="utf-8") as f:
#         content = f.read()

#     # Remove WEBVTT header and NOTE blocks
#     content = re.sub(r"WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
#     content = re.sub(r"NOTE.*?\n\n", "", content, flags=re.DOTALL)

#     lines = content.splitlines()
#     clean_lines = []
#     seen = set()

#     for line in lines:
#         line = line.strip()
#         # Skip timestamps, cue IDs, and empty lines
#         if not line or "-->" in line or re.match(r"^\d+$", line):
#             continue
#         # Remove inline tags like <00:00:01.000>, <c>, </c>
#         line = re.sub(r"<[^>]+>", "", line)
#         line = line.strip()
#         # Deduplicate consecutive repeated lines (common in auto-subs)
#         if line and line not in seen:
#             clean_lines.append(line)
#             seen.add(line)
#         elif line in seen:
#             seen = {line}  # reset window to allow repeated words in new context

#     return " ".join(clean_lines)


# def list_available_subtitles(url: str) -> dict:
#     """List all available subtitle languages for a video."""
#     ydl_opts = {"quiet": True, "skip_download": True}

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=False)

#     manual = info.get("subtitles", {})
#     auto = info.get("automatic_captions", {})

#     print(f"\nVideo: {info.get('title')}")
#     print(f"Manual subtitles:    {list(manual.keys()) or 'None'}")
#     print(f"Auto-generated subs: {list(auto.keys()) or 'None'}")

#     return {"manual": manual, "automatic": auto}


# # This needs to be in-memory rather than using file storage
# def get_transcript(url: str, lang: str = "en", save_txt: bool = True) -> str:
#     """
#     High-level function: download + parse transcript, return plain text.

#     Args:
#         url: YouTube video URL
#         lang: Subtitle language code
#         save_txt: If True, saves plain text transcript to disk

#     Returns:
#         Transcript as a plain text string
#     """
#     print(f"Downloading subtitles for: {url}")
#     vtt_path = download_transcript(url, lang=lang)

#     print(f"Parsing: {vtt_path}")
#     transcript = parse_vtt(vtt_path)

#     if save_txt:
#         txt_path = vtt_path.rsplit(".", 2)[0] + ".txt"
#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(transcript)
#         print(f"Transcript saved to: {txt_path}")

#     return transcript


# --- Example usage ---
# if __name__ == "__main__":
#     VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

#     # 1. Check what's available
#     list_available_subtitles(VIDEO_URL)

#     # 2. Download and parse transcript
#     text = get_transcript(VIDEO_URL, lang="en", save_txt=True)

#     print("\n--- Transcript Preview ---")
#     print(text[:500])