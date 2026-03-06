
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from pydantic import BaseModel, Field
from langchain.tools import ToolRuntime, tool
from src.anubis.utils.utility import extract_user_id_assistant_id
from src.anubis.utils.model import init_model

from langchain_core.messages import SystemMessage

@tool()
async def recall_memories(runtime: ToolRuntime):
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
    update = {"state_update_data": {"recalled_memory_documents": evoked_memories_response},
              "tool_message":f"Evoked {len(evoked_memories_response)} memories."}

    return update