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

from src.anubis.utils.state import UserState, AssistantState, UserIdentityState, AssistantIdentityState

from datetime import datetime, timezone


from langchain_core.messages import AnyMessage
from typing import Annotated
from pydantic import BaseModel, Field

from src.anubis.utils.model import init_model
from src.anubis.utils.utility import extract_user_id_assistant_id
from typing import Dict

# @tool()
# async def create_memory(
#     content: str, 
#     # Hide these arguments from the model.
#     assistant_state: Annotated[AssistantState, InjectedToolArg],
#     user_state: Annotated[UserState, InjectedToolArg],
#     runtime: ToolRuntime
# ):
#     """Create a new memory.

#     Create a new memory between a user and an assistant. 
#     This tool is used to create memories that the user has told the assistant.
#     An example memory is the user telling the assistant the assistant's name.

     

#     Args:
#         content: The main content of the memory. For example:
#             "User expressed interest in learning about French."
#         context: Additional context for the memory. For example:
#             "This was mentioned while discussing career options in Europe."
#         memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
#         The memory to overwrite.
#     """

#     mem_id = uuid.uuid4()
#     await runtime.store.aput(
#         (user_state.user_id, assistant_state.assistant_id, "memories"),
#         key=str(mem_id),
#         value={"content": content, "context": context},
#     )
#     return f"Stored memory {mem_id}"

from langgraph.types import Command
from langchain_core.messages import ToolMessage
@tool()
async def remember_memories(runtime: ToolRuntime):
    """This is used to retrieve memories or 'REMEMBER MEMORIES' from the store.
     These memories have been shared from the user and contain information about the assistant's identity.
     These memories have been shared from the user and contain information that was important that assistant decided the assistant needed to remember.
     This is used when the assistant needs to RECALL or REMEMBER important information that the assistant chose to MEMORIZE or REMEMBER. 

    Args:
        runtime (ToolRuntime): contains the store that holds the memories.

    Returns:
        GlobalState: Contains the AssistantState class 
        and the RememberedMemories class that contains 
        the list of memories listed as 
        remembered_memory_documents 
        with a description of 
        'This is a list of memories for reference during chat created as 
            Document objects for search by user_id and assistant_id or with 
            relevance scores in vectorstore 
            (Document type is the standard for the vector store).'
    """
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    class ConversationSummaryToProvokeMemories(BaseModel):
        """
        Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
        These document memories have been shared from the user and contain information about the assistant's identity.
        These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
        """
        evoked_memory_query: str =  Field(description = """
                                          This is the summary of the converation. This summary is posed succinctly such that the summary is similar to the content to which a retrived list of documents is the response. The document memories have been shared from the user and contain information about the assistant's identity or contain information that was important that the assistant decided the assistant needed to remember. It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.""")

    model_with_structured_output = init_model(context = runtime.context, response_format=ConversationSummaryToProvokeMemories)

    MEMORY_EVOCATION_INSTRUCTIONS = """
    <INSTRUCTIONS>
        Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
        These document memories have been shared from the user and contain information about the assistant's identity.
        These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
        It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.
        The summary must be at most one sentence. 
        The summary can be a single word or phrase that identifies the topic of conversation. 
    </INSTRUCTIONS>

    <INSTRUCTIONS>
        Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
        These document memories have been shared from the user and contain information about the assistant's identity.
        These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
        It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.
        The summary must be at most one sentence. 
        The summary can be a single word or phrase that identifies the topic of conversation.
    </INSTRUCTIONS>
    """

    system_message = SystemMessage(content=MEMORY_EVOCATION_INSTRUCTIONS)

    
    messages = [message for message in runtime.state['messages'] if type(message) is not SystemMessage]
    
    tool_call_id = messages[-1].tool_calls[0]['id']
    chat_prompt_model = system_message + messages

    response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    evoked_memory_query = getattr(response, "evoked_memory_query", "")

    evoked_memories_response = await runtime.store.asearch(
        assistant_memory_namespace,
        query=evoked_memory_query
    )
    update = {"recalled_memory_documents": evoked_memories_response, 
              "messages":[ToolMessage(content=f"Evoked {len(evoked_memories_response)} memories.", tool_call_id=tool_call_id)]}
    goto = "load_consciousness"
    tool_call_id = runtime.tool_call_id
    return Command(update=update, goto=goto)

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
        
    # class NewUserFactJudgeLLM(BaseModel):
    #     non_new_fact: bool = Field(description = "This value is TRUE when there is a matching fact. " \
    #     "This value is FALSE when there is not a matching fact in the list of known facts." \
    #     "This information is already exists in the learned facts")
    #     key_id: str = Field(description = "This is the id as a string of uuid of the fact that is exactly the same as the fact that is trying to be learned." \
    #     "The key is located in the same dictionary as the matching_fact in the known_facts list under 'document.kwargs.metadata.id'" \
    #     "the key may be None if the user fact is a new fact.")
    #     known_facts: List[Dict[str, any]] = Field(description = "These are facts that are already known about the user. " \
    #     "The fact is in a dictionary located under 'document.kwargs.metadata.fact' and the key is located in the same dictionary under 'document.kwargs.metadata.id'" )
    #     matching_fact: str = Field("This is the fact that is the same as the user_fact that is trying to be learned. " \
    #     "This may not exist if the user fact is a new fact that is not already in the list of known facts." \
    #     "The fact is in a dictionary located under 'document.kwargs.metadata.fact' in the list of known facts")
    #     reasoning: str = Field("This is the reasoning why the user fact is a new fact or is not a new fact. " \
    #     "This is a clear reason. " \
    #     "The user_fact may be a matching fact or the user_fact may be a new fact if the user_fact is not in the list of known facts. ")

    # NEW_USER_FACT_LLM_AS_A_JUDGE_INSTRUCTIONS = """
    # <INSTRUCTIONS>
    # Identify if there is already an exact fact as the included fact in the list of known facts.
    # Given the list of known facts, identify if there is a matching fact.
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # <RULES>
    # Do not change the information of the fact.
    # Do not change any of the keys.
    # Do not change any information at all ever.
    # </RULES>
    # <FACT>
    # {content}
    # </FACT>
    # <INSTRUCTIONS>
    # Identify the fact that the user shared about themselves. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # """

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


    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

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

    tool_call_id = runtime.state['messages'][-1]['tool_calls'][0]['id']

    update =  {"assistant_identity_state": {"recalled_memory_documents": assistant_identity_memory_document},
               "messages":[ToolMessage(content="{assistant_fact}", tool_call_id = tool_call_id)]}

    goto="update_identity_tool_classification"
    return Command(update=update, goto=goto)

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
        
    # class NewUserFactJudgeLLM(BaseModel):
    #     non_new_fact: bool = Field(description = "This value is TRUE when there is a matching fact. " \
    #     "This value is FALSE when there is not a matching fact in the list of known facts." \
    #     "This information is already exists in the learned facts")
    #     key_id: str = Field(description = "This is the id as a string of uuid of the fact that is exactly the same as the fact that is trying to be learned." \
    #     "The key is located in the same dictionary as the matching_fact in the known_facts list under 'document.kwargs.metadata.id'" \
    #     "the key may be None if the user fact is a new fact.")
    #     known_facts: List[Dict[str, any]] = Field(description = "These are facts that are already known about the user. " \
    #     "The fact is in a dictionary located under 'document.kwargs.metadata.fact' and the key is located in the same dictionary under 'document.kwargs.metadata.id'" )
    #     matching_fact: str = Field("This is the fact that is the same as the user_fact that is trying to be learned. " \
    #     "This may not exist if the user fact is a new fact that is not already in the list of known facts." \
    #     "The fact is in a dictionary located under 'document.kwargs.metadata.fact' in the list of known facts")
    #     reasoning: str = Field("This is the reasoning why the user fact is a new fact or is not a new fact. " \
    #     "This is a clear reason. " \
    #     "The user_fact may be a matching fact or the user_fact may be a new fact if the user_fact is not in the list of known facts. ")

    # NEW_USER_FACT_LLM_AS_A_JUDGE_INSTRUCTIONS = """
    # <INSTRUCTIONS>
    # Identify if there is already an exact fact as the included fact in the list of known facts.
    # Given the list of known facts, identify if there is a matching fact.
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # <RULES>
    # Do not change the information of the fact.
    # Do not change any of the keys.
    # Do not change any information at all ever.
    # </RULES>
    # <FACT>
    # {content}
    # </FACT>
    # <INSTRUCTIONS>
    # Identify the fact that the user shared about themselves. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # """

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


    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Identity of the user from the assistant's perspective
    user_identity_namespace = (assistant_id, user_id, 'identity')

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

    update = {"user_identity_state": 
               {"user_identity_documents": user_identity_document},
               "messages": [ToolMessage(content=f"Learned: {user_fact}", tool_call_id=tool_call_id)]}
    goto = "update_identity_tool_classification"

    return Command(update=update, goto=goto)

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


