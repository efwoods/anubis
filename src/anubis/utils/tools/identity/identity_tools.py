""" Agent SubGraph Tools """
import uuid
import logging
from typing import List, Annotated

from langchain.tools import tool, ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


from src.anubis.utils.state import GlobalState

logger = logging.getLogger(__name__)



from langchain_core.messages import AnyMessage
from typing import Annotated
from pydantic import BaseModel, Field

from src.anubis.utils.model import init_model
from src.anubis.utils.utility import extract_user_id_assistant_id
from typing import Dict
from langgraph.types import Command

@tool()
async def create_a_memory(
    content: str, 
    # Hide these arguments from the model.
    runtime: ToolRuntime
) -> GlobalState:
    """Create a memory whenever a significant event occurs or when prompted to remember. 
    This is not used when the user is telling the assistant about the assistant's identity.

    This tool is used to create memories that are of SIGNIFICANT FACTS, EVENTS, OR OCCURANCES given the context of your system prompt. 
    An example memory is the user telling the assistant to remember something or the user reveals information about any significant event or a fact or event occurs that is found significant given your specific role and context.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.

    RULE:
    DO NOT CALL THIS TOOL MULTIPLE TIMES WITH THE SAME FACT.

    Example:
    WOW! THIS JUST HAPPENED!
    I never tell anyone this, but here is a secret.
    I want you to remember that.
    This is important.
    This is important to me.

    Args:
        content: The main content of the significant fact, event, or occurance. For example:
            The user reveals a secret. A suprising event occurs. The user indicates this is important. A change or choice is made.
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    class SignificantFactAndContext(BaseModel):
        """
        Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
        """
        significant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
        fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")

    model_with_structured_output = init_model(context = runtime.context, response_format=SignificantFactAndContext)

    DECISION_INSTRUCTIONS = """
    <INSTRUCTIONS>
    Identify the fact, occurance, or event that was asserted was important, found significant, or was a choice or a point of no return. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>

    <FACT>
    {content}
    </FACT>

    <INSTRUCTIONS>
    Identify the fact, occurance, or event that was asserted was important, found significant, or was a choice or a point of no return. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>
    """


    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user. (assistant's memory space)
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    identity_id = str(uuid.uuid4())

    system_message = SystemMessage(content=DECISION_INSTRUCTIONS.format(content=content))

    chat_prompt_model = system_message + runtime.state['messages']

    response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    significant_fact = getattr(response, "assistant_fact", "")
    fact_context = getattr(response, "fact_context", "")

    searchable_page_content = "\n\n".join([significant_fact, fact_context])

    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": fact_context,
        "fact":significant_fact
    }

    assistant_identity_memory_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    assistant_identity_memory_document_json = assistant_identity_memory_document.to_json()

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )

    recent_message = runtime.state['messages'][-1]
    tool_call_id = recent_message.tool_calls[0].get('id')
    update = {"recalled_memory_documents": [assistant_identity_memory_document],
              "messages": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return Command(update=update, goto="load_consciousness")

@tool()
async def learn_information_about_yourself_through_text_from_the_user_as_a_memory(
    content: str, 
    # Hide these arguments from the model.
    runtime: ToolRuntime
) -> GlobalState:
    """Learn Facts about yourself from the User through text

    Update the known information about yourself.
    This tool is used to create memories that the user has told the assistant that are 
    SPECIFICALLY ABOUT THE IDENTITY OF THE ASSISTANT, MODEL, or LLM 
    addressed as YOU or or YOUR or YOURS or the direct given name of the assistant.
    An example memory is the user telling the assistant what the assistant's name is or facts about the assistant.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.

    RULE:
    DO NOT CALL THIS TOOL MULTIPLE TIMES WITH THE SAME FACT.

    Example:
    Hi, my name is Evan.
    This is a description of me.
    
    In each example, the name and description are each saved in a database namespace between the assistant and the user.     

    Example:
    Input:
    Your name is Shivon Zilis, and you have twins.
    Becomes multiple tool calls (multiple facts follow):
    My name is Shivon Zilis.
    I have twins.

    Args:
        content: The main content of the assistant's identity. For example:
            "User expressed that the assistant has an interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    class AssistantFactAndContext(BaseModel):
        """
        Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
        """
        assistant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
        fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")
    
    
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    # VERIFY FACT DOES NOT ALREADY EXIST
    if runtime.state.get('assistant_identity_documents', None) is not None:
        assistant_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['assistant_identity_documents']]
        assistant_content_store_query_results = await runtime.store.asearch(assistant_memory_namespace, query=content)
        assistant_content_store_query_results_significant = [item for item in assistant_content_store_query_results if item.score > 0.8]
        if content in assistant_identity_documents_text_list or len(assistant_content_store_query_results_significant) > 0:
            # Fact already exists:
            recent_message = runtime.state['messages'][-1]
            tool_call_id = recent_message.tool_calls[0].get('id')

            update = {"messages": [ToolMessage(content=f"Fact: {content} previously learned", tool_call_id=tool_call_id)]}
            return Command(update=update, goto="load_consciousness")

    model_with_structured_output = init_model(context = runtime.context, response_format=AssistantFactAndContext)

    DECISION_INSTRUCTIONS = """
    <INSTRUCTIONS>
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>

    <FACT>
    {content}
    </FACT>

    <INSTRUCTIONS>
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>
    """



    identity_id = str(uuid.uuid4())

    system_message = SystemMessage(content=DECISION_INSTRUCTIONS.format(content=content))

    chat_prompt_model = system_message + runtime.state['messages']

    response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    assistant_fact = getattr(response, "assistant_fact", "")
    fact_context = getattr(response, "fact_context", "")

    searchable_page_content = "\n\n".join([assistant_fact, fact_context])

    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": fact_context,
        "fact":assistant_fact
    }

    assistant_identity_memory_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    assistant_identity_memory_document_json = assistant_identity_memory_document.to_json()

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )

    recent_message = runtime.state['messages'][-1]
    tool_call_id = recent_message.tool_calls[0].get('id')
    logger.warning(f"tool_call_id: {tool_call_id}")
    update = {"recalled_memory_documents": [assistant_identity_memory_document],
              "messages": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return Command(update=update, goto="load_consciousness")

@tool()
async def learn_information_about_the_user(
    content: str, 
    # Hide these arguments from the model.
    runtime: ToolRuntime
) -> GlobalState:
    """Learn Facts about the User

    Update the known information about the user.
    The user is a primary source of information about the user, therefore the identity namespace is updated when the user shares information about themself.
    This tool is used to create documents of identity that the user has told the assistant that are SPECIFICALLY ABOUT THE USER.
    An example document identity is the user telling the assistant what the user's name is or facts about the user.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    Example:
    Hi, my name is Evan.
    This is a description of me.
    
    In each example, the name and description are each saved in a database namespace between the assistant and the user.     

    Example:
    Input:
    Hi, may name is Evan, and I love you.
    Becomes multiple tool calls (multiple facts follow):
    Hi, my name is Evan.
    I love you.

    Args:
        content: The main content of the user's identity. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    class UserFactAndContext(BaseModel):
        """
        Extract Facts about the USER and the context of that fact given the history of messages.
        """
        user_fact: str =  Field(description = "This is the fact about the user that was shared by the user.")
        fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Identity of the user from the assistant's perspective
    user_identity_namespace = (assistant_id, user_id, 'identity')

    # VERIFY FACT DOES NOT ALREADY EXIST
    user_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['user_identity_documents']]
    user_content_store_query_results = await runtime.store.asearch(user_identity_namespace, query=content)
    user_content_store_query_results_significant = [item for item in user_content_store_query_results if item.score > 0.8]
    if content in user_identity_documents_text_list or len(user_content_store_query_results_significant) > 0:
        # Fact already exists:
        recent_message = runtime.state['messages'][-1]
        tool_call_id = recent_message.tool_calls[0].get('id')

        update = {"messages": [ToolMessage(content=f"Fact: {content} previously learned", tool_call_id=tool_call_id)]}
        return Command(update=update, goto="load_consciousness")
    
    model_with_structured_output = init_model(context = runtime.context, response_format=UserFactAndContext)

    DECISION_INSTRUCTIONS = """
    <INSTRUCTIONS>
    Identify the fact that the user shared about themselves. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>

    <FACT>
    {content}
    </FACT>

    <INSTRUCTIONS>
    Identity the fact that the user shared about themselves. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    </INSTRUCTIONS>
    """

    identity_id = str(uuid.uuid4())

    system_message = SystemMessage(content=DECISION_INSTRUCTIONS.format(content=content))

    chat_prompt_model = system_message + runtime.state['messages']

    response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    user_fact = getattr(response, "user_fact", "")
    fact_context = getattr(response, "fact_context", "")

    searchable_page_content = "\n\n".join([user_fact, fact_context])

    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": fact_context,
        "fact":user_fact
    }

    user_identity_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    user_identity_document_json = user_identity_document.to_json()

    await runtime.store.aput(
        user_identity_namespace,
        key=identity_id,
        value={"document": user_identity_document_json},
    )

    recent_message = runtime.state['messages'][-1]
    tool_call_id = recent_message.tool_calls[0].get('id')

    update = {"user_identity_documents": [user_identity_document],
               "messages": [ToolMessage(content=f"Learned: {user_fact}")]}

    return Command(update=update, goto="load_consciousness")
































# TODO: YOUTUBE IDENTITY UPDATER
# TODO: USE MEMORY RATHER THAN FILE SYSTEM
# from src.anubis.utils.helper_functions import download_transcript, parse_vtt

# @tool
# def get_transcript(url: str, lang: str = "en", save_txt: bool = False) -> str:
#     """
#     Use this tool when a user suggests there is a youtube link 
#     or the link included in the message contains the word 'youtube'. 
#     Use this to download the text from the link.


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

#     print(f"TRANSCRIPT: {transcript}")

#     return transcript

# TODO: IMAGE URL IDENTITY UPDATER

# TODO: TEXT WEBPAGE URL IDENTITY UPDATER
# TODO: TEST CONFIGURATION & STORE CAPABILITY
from langchain_openai import ChatOpenAI
from src.anubis.utils.prompts.system_prompts import FACT_FORMATTING_STRING_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from uuid import uuid4, uuid5, NAMESPACE_URL
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document

# @tool
# async def update_identity_via_text_content_url(url: str, runtime: ToolRuntime):
    
#     # Extract 
#     user_id = runtime.config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
#     assistant_id = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_id", "")
#     assistant_name = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_name", "")
    
#     model = ChatOpenAI(
#         model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8",
#         base_url = "https://api.llama.com/compat/v1/",
#         api_key = "LLM|2160921497745669|2M_zeRPt10hKzWlJw39C_P-UIHM",
#         temperature = 0.1
#     )
#     system_message = SystemMessage(content = FACT_FORMATTING_STRING_PROMPT.format(assistant_name=assistant_name))

#     filename = url
#     filename_uuid5 = uuid5(NAMESPACE_URL, url)

#     namespace = (user_id, assistant_id, "identity", filename_uuid5)
    
#     loader = UnstructuredLoader(web_url=url)
#     docs = loader.load()

#     total_number_of_documents = len(docs)
#     number_of_documents_to_format = 5
#     for index in range(0, total_number_of_documents, number_of_documents_to_format):
#         retrieved_data = "\n\n".join([doc.page_content for doc in docs[index:index + number_of_documents_to_format]])
#         human_message = HumanMessage(content=retrieved_data)
#         messages = [system_message, human_message]
#         extracted_response = await model.ainvoke(input=messages)
#         extracted_response_document = Document(page_content=extracted_response)
#         key = str(uuid4())
#         extracted_response_document.metadata.update({"filename":filename, "user_id":user_id, "assistant_id": assistant_id, "filename_uuid5":filename_uuid5, "key":key})
#         document_data_json = extracted_response_document.to_json()

#         # Upload the document to the store
#         await runtime.store.aput(namespace, key=key, value={"document":document_data_json})

# TODO: IN-MESSAGE IDENTITY UPDATER (TEXT)


