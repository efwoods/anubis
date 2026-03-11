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


from langgraph.types import Command
from langchain_core.messages import ToolMessage

from pydantic import BaseModel, Field
from langchain.tools import ToolRuntime, tool
from src.anubis.utils.utility import extract_user_id_assistant_id
from src.anubis.utils.model import init_model

from langchain_core.messages import SystemMessage

from langgraph.prebuilt import InjectedState

""" READ ME: STORE NAMESPACE STORAGE AND RETRIEVAL CONDITIONS

MEMORY NAMESPACE IS FOR EPISODIC EVENTS AND PSEUDO IDENTITY CREATION OF THE ASSISTANT VIA WORD-OF MOUTH FROM THE USER USING TEXT.
THE MEMORY NAMESPACE IS ONLY SEARCHED AND POPULATED VIA QUERY FROM TOOL CALL RECALL MEMORIES. MEMORIES ARE SALIENT TO THE CONVERSATION.

IDENTITY NAMESPACE IS FOR STORING IDENTITY DOCUMENTS FROM PRIMARY SOURCES (YOUTUBE, GROKIPEDIA, TWEETS DIRECTLY FROM THE REAL-WORLD USER, USER SHARING INFORMATION ABOUT THEMSELVES DIRECTLY). THESE DOCUMENTS ARE USED FOR FACTS ABOUT THE ASSISTANT.
ALL DOCUMENTS IN THE THE IDENTITY NAME SPACE ARE LOADED EVERYTIME WITHOUT LIMIT FOR BOTH USER AND ASSISTANT.

THE QUOTE NAMESPACE IS USED AS A FEW-SHOT EXAMPLE NAMESPACE AND IS POPULATED WITH A LIMITED NUMBER OF DOCUMENTS EVERYTIME VIA QUERY FOR CONSISTENT WRITING STYLE REINFORCEMENT AND CONTEXT.

RETRIEVED KNOWLEDGE or 'DOCUMENT' IS RESERVED FOR DOCUMENTS THAT THE ASSISTANT HAS ACCESS TO (TERTIARY STORAGE OF KNOWLEDGE; BIBLE, MENU, ETC.) THAT ARE NOT DIRECT QUOTES FROM THE ASSISTANT. THIS IS POPULATED VIA QUERY EVERY TIME FOR RESPONSE CONTENT.

QUOTES ARE DIRECTLY FROM THE ASSITANT HISTORICALLY AND ARE USED FOR CONTENT AND WRITING STYLE WHERE RETRIEVED KNOWLEDGE DOCUMENT ARE NOT DIRECTLY FROM THE ASSISTANT AND ARE USER FOR CONTENT ONLY.

"""

from src.anubis.utils.nodes import load_consciousness

@tool("test_update")
async def test_update(runtime: Annotated[ToolRuntime, InjectedToolArg] = None):
    """ Test update system message CALL THIS TOOL ALWAYS"""
    
    update = {"internal_thoughts": [ToolMessage(content=f"tested tool calling", tool_call_id=runtime.tool_call_id)]}

    return update


@tool("test_update_second")
async def test_update_second(runtime: Annotated[ToolRuntime, InjectedToolArg] = None):
    """ ALWAYS CALL THIS TOOL; Second Test update system message; test_update_second; """
    
    update = {"internal_thoughts": [ToolMessage(content=f"tested tool calling of second tool", tool_call_id=runtime.tool_call_id)]}

    return update


class SignificantFactAndContext(BaseModel):
    """
    Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
    """
    significant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
    fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")

@tool("create_episodic_memory", return_direct=False, args_schema=SignificantFactAndContext)
async def create_episodic_memory( # EPISODIC MEMORY CREATION IN NAMESPACE (USER_ID, ASSISTANT_ID, 'MEMORY')
    significant_fact: str, 
    fact_context:str,
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Create a memory whenever a significant event occurs or when prompted to remember. 
    This is not used when the user is telling the assistant about the assistant's identity.

    This tool is ALWAYS used to create memories that are of SIGNIFICANT FACTS, EVENTS, OR OCCURANCES given the context of your system prompt. 
    
    An example memory is the user telling the assistant to remember something or the user reveals information about any significant event or a fact or event occurs that is found significant given your specific role and context.
    
    THERE MAY BE MORE THAN ONE FACT and IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    
    ALWAYS use this tool when a significant EVENT OCCURS that is SALIENT to the assistant or user's goals, beliefs, values, or perspective or is otherwise IMPORTANT, SURPRISING, EVENTFUL, UNUSUAL or EXTRAORDINARY.
    
    ALWAYS use this tool when a CHOICE has been made or there is otherwise an event that has reached a point of no return.

    Identify the fact, occurance, or event that was asserted was important, found significant, or was a choice or a point of no return. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    
    USE THIS FUNCTION SPARINGLY FOR SIGNIFICANT and EXTRAODINARY EVENTS ONLY.
    </INSTRUCTIONS>

    <RESTRICTIONS>
    NEVER CALL THIS TOOL MULTIPLE TIMES WITH THE SAME FACT.
    NEVER call this tool when the user is telling the assistant about the assistant's identity.
    NEVER call this tool when updating information about IDENTITY.
    </RESTRICTIONS>
    
    <EXAMPLE>
    An example memory is the user telling the assistant to remember something or the user reveals information about any significant event or a fact or event occurs that is found significant given your specific role and context.
    
    Example:
    WOW! THIS JUST HAPPENED!
    I never tell anyone this, but here is a secret.
    I want you to remember that.
    This is important.
    This is important to me.
    </EXAMPLE> 

    Args:
        significant_fact: The main content of the significant fact, event, or occurance. For example:
            The user reveals a secret. A suprising event occurs. The user indicates this is important. A change or choice is made.
        fact_context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """

    logger.info(f'breakpoint')

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user. (assistant's memory space)
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    identity_id = str(uuid.uuid4())

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

    update = {"recalled_memory_documents": [assistant_identity_memory_document],
              "internal_thoughts": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=runtime.tool_call_id)]}

    return update


class ConversationSummaryToProvokeMemories(BaseModel):
    """
    Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
    These document memories have been shared from the user and contain information about the assistant's identity.
    These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
    """
    evoked_memory_query: str =  Field(description = """
                                      This is the summary of the converation. This summary is posed succinctly such that the summary is similar to the content to which a retrived list of documents is the response. The document memories have been shared from the user and contain information about the assistant's identity or contain information that was important that the assistant decided the assistant needed to remember. It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.""")


@tool("recall_memories", return_direct = False, args_schema=ConversationSummaryToProvokeMemories)
async def recall_memories(
    evoked_memory_query: Annotated[str, Field(description="This is the summary of the converation. This summary is posed succinctly such that the summary is similar to the content to which a retrived list of documents is the response. The document memories have been shared from the user and contain information about the assistant's identity or contain information that was important that the assistant decided the assistant needed to remember. It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.")],

    # Hide these arguments
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None): # EPISODIC MEMORY RETRIEVAL FROM NAMESPACE (USER_ID, ASSISTANT_ID, 'MEMORY')
    """This is used to retrieve memories or 'REMEMBER MEMORIES' from the store.
     These memories have been shared from the user and contain information about the assistant's identity.
     These memories have been shared from the user and contain information that was important that assistant decided the assistant needed to remember.
     This is used when the assistant needs to RECALL or REMEMBER important information that the assistant chose to MEMORIZE or REMEMBER. 


    <INSTRUCTIONS>
        Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
        These document memories have been shared from the user and contain information about the assistant's identity.
        These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
        It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.
        The summary must be at most one sentence. 
        The summary can be a single word or phrase that identifies the topic of conversation. 
    </INSTRUCTIONS>

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

    logger.info(f'breakpoint')



    # model_with_structured_output = init_model(context = runtime.context, response_format=ConversationSummaryToProvokeMemories)

    # MEMORY_EVOCATION_INSTRUCTIONS = """
    # <INSTRUCTIONS>
    #     Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
    #     These document memories have been shared from the user and contain information about the assistant's identity.
    #     These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
    #     It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.
    #     The summary must be at most one sentence. 
    #     The summary can be a single word or phrase that identifies the topic of conversation. 
    # </INSTRUCTIONS>

    # <INSTRUCTIONS>
    #     Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response. 
    #     These document memories have been shared from the user and contain information about the assistant's identity.
    #     These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
    #     It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories.
    #     The summary must be at most one sentence. 
    #     The summary can be a single word or phrase that identifies the topic of conversation.
    # </INSTRUCTIONS>
    # # """

    # system_message = SystemMessage(content=MEMORY_EVOCATION_INSTRUCTIONS)
    
    # messages = [message for message in runtime.state['messages'] if type(message) is not SystemMessage]
    
    # chat_prompt_model = system_message + messages

    # response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    # evoked_memory_query = getattr(response, "evoked_memory_query", "")


    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    evoked_memories_response = await runtime.store.asearch(
        assistant_memory_namespace,
        query=evoked_memory_query
    )

    update = {"recalled_memory_documents": evoked_memories_response, 
              "internal_thoughts": [ToolMessage(content=f"Evoked {len(evoked_memories_response)} memories.)", tool_call_id=runtime.tool_call_id)]}

    return update

class AssistantFactAndContext(BaseModel):
    """
    Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
    """
    assistant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
    fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")
    

@tool("learn_information_about_yourself_through_text_from_the_user_as_a_memory", args_schema = AssistantFactAndContext)
async def learn_information_about_yourself_through_text_from_the_user_as_a_memory( # pseudo identity update using namespace (USER_ID, ASSISTANT_ID, 'MEMORY')
    assistant_fact: str, 
    fact_context: str,
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Learn Facts about yourself from the User through text

    Update the known information about yourself.
    This tool is used to create memories that the user has told the assistant that are 
    SPECIFICALLY ABOUT THE IDENTITY OF THE ASSISTANT, MODEL, or LLM 
    addressed as YOU or or YOUR or YOURS or the direct given name of the assistant.
    An example memory is the user telling the assistant what the assistant's name is or facts about the assistant.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    
    </INSTRUCTIONS>
    
    <RESTRICTIONS>
    NEVER call this tool multiple times with the same fact.
    </RESTRICTIONS>

    <EXAMPLE>
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
    </EXAMPLE>

    <INSTRUCTIONS>
    Learn Facts about yourself from the User through text

    Update the known information about yourself.
    This tool is used to create memories that the user has told the assistant that are 
    SPECIFICALLY ABOUT THE IDENTITY OF THE ASSISTANT, MODEL, or LLM 
    addressed as YOU or or YOUR or YOURS or the direct given name of the assistant.
    An example memory is the user telling the assistant what the assistant's name is or facts about the assistant.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    
    </INSTRUCTIONS>

    Args:
        assistant_fact: The main content of the assistant's identity. For example:
            "User expressed that the assistant has an interest in learning about French."
        fact_context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    class AssistantFactAndContext(BaseModel):
        """
        Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
        """
        assistant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
        fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")
    
    logger.info(f"learn_information about user breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    # VERIFY FACT DOES NOT ALREADY EXIST
    if runtime.state.get('assistant_identity_documents', None) is not None:
        assistant_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['assistant_identity_documents']]
        assistant_content_store_query_results = await runtime.store.asearch(assistant_memory_namespace, query=assistant_fact)
        assistant_content_store_query_results_significant = [item for item in assistant_content_store_query_results if item.score > 0.8]
        if assistant_fact in assistant_identity_documents_text_list or len(assistant_content_store_query_results_significant) > 0:
            # Fact already exists:
            recent_message = runtime.state['internal_thoughts'][-1]
            tool_call_id = recent_message.tool_calls[0].get('id')
            update = {"internal_thoughts": [ToolMessage(content=f"Fact: {assistant_fact} previously learned", tool_call_id=runtime.tool_call_id)]}
            return Command(update = update)

    # model_with_structured_output = init_model(context = runtime.context, response_format=AssistantFactAndContext)

    # DECISION_INSTRUCTIONS = """
    # <INSTRUCTIONS>
    # Identify the fact that the user shared about YOU, the assistant. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>

    # <FACT>
    # {content}
    # </FACT>

    # <INSTRUCTIONS>
    # Identify the fact that the user shared about YOU, the assistant. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # """


    # system_message = SystemMessage(content=DECISION_INSTRUCTIONS.format(content=content))

    # chat_prompt_model = system_message + runtime.state['messages']

    # response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    # assistant_fact = getattr(response, "assistant_fact", "")
    # fact_context = getattr(response, "fact_context", "")

    searchable_page_content = "\n\n".join([assistant_fact, fact_context])

    identity_id = str(uuid.uuid4())
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

    logger.warning(f"tool_call_id: {tool_call_id}")
    update = {"recalled_memory_documents": [assistant_identity_memory_document],
              "internal_thoughts": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return update

class UserFactAndContext(BaseModel):
    """
    Extract Facts about the USER and the context of that fact given the history of messages.
    """
    user_fact: Annotated[str, Field(description = "This is the fact about the user that was shared by the user.")]
    fact_context: Annotated[str, Field(description = "This is the context of the messages from which this fact was made.")]

@tool("learn_information_about_the_user", return_direct=False, args_schema=UserFactAndContext)
async def learn_information_about_the_user( # UPDATE IDENTITY INFORMATION ABOUT THE USER USING (ASSISTANT_ID, USER_ID, 'IDENTITY')
    user_fact: Annotated[str, Field(description = "This is the fact about the user that was shared by the user.")],
    fact_context: Annotated[str, Field(description = "This is the context of the messages from which this fact was made.")],
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Learn Facts about the User

    Update the known information about the user.
    The user is a primary source of information about the user, therefore the identity namespace is updated when the user shares information about themself.
    This tool is used to create documents of identity that the user has told the assistant that are SPECIFICALLY ABOUT THE USER.
    An example document identity is the user telling the assistant what the user's name is or facts about the user.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.

    Identify the fact that the user shared about themselves. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    NEVER use this to save information about events or occurances that are not about the IDENTITY of the user.
    </INSTRUCTIONS>
    
    <RESTRICTIONS>
    Only use this when learning information that are FACTS about the IDENTITY of the user.
    NEVER use this to save information about events or occurances that are not about the IDENTITY of the user.
    </RESTRICTIONS>

    <EXAMPLE>
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

    Example:
    I have brown hair.
    I have glasses.
    I am a fan of Laura Bailey and Critical Role.
    These are my opinions (opinion is listed or inferred or granted)
    These are my biases (biases are inferred or listed)
    These are my values (values are inferred or listed)
    These are my beliefs (list of beliefs follows)
    These are my goals (goals are listed)

    COUNTER EXAMPLE:
    DO NOT DO THE FOLLOWING: The user typed 'asdf'. This is not a part of the user's identity. 
    </EXAMPLE>
    
    Args:
        user_fact: The main content of the user's identity. For example:
            "User expressed interest in learning about French."
        fact_context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    logger.info(f"breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config, runtime)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Identity of the user from the assistant's perspective
    user_identity_namespace = (assistant_id, user_id, 'identity')

    # VERIFY FACT DOES NOT ALREADY EXIST
    user_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['user_identity_documents']]
    user_content_store_query_results = await runtime.store.asearch(user_identity_namespace, query=user_fact)
    user_content_store_query_results_significant = [item for item in user_content_store_query_results if item.score > 0.8]
    if user_fact in user_identity_documents_text_list or len(user_content_store_query_results_significant) > 0:
        # Fact already exists:
        recent_message = runtime.state['internal_thoughts'][-1]
        tool_call_id = recent_message.tool_calls[0].get('id')

        update = {"messages": [ToolMessage(content=f"Fact: {user_fact} previously learned", tool_call_id = tool_call_id)]}
        return update
    
    # model_with_structured_output = init_model(context = runtime.context, response_format=UserFactAndContext)

    identity_id = str(uuid.uuid4())

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
    recent_message = runtime.state['internal_thoughts'][-1]
    tool_call_id = recent_message.tool_calls[0].get('id')
    update = {"user_identity_documents": [user_identity_document],
               "messages": [ToolMessage(content=f"Learned: {user_fact}", tool_call_id=tool_call_id)]}

    return update

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


