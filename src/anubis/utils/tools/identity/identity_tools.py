""" Agent SubGraph Tools """
import uuid
import logging

from langchain.tools import tool, ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph.types import Command
from langchain_core.messages import ToolMessage

from src.anubis.utils.state import GlobalState
from src.anubis.utils.model import init_model
from src.anubis.utils.utility import extract_user_id_assistant_id

logger = logging.getLogger(__name__)

from typing import Annotated
from pydantic import BaseModel, Field

""" READ ME: STORE NAMESPACE STORAGE AND RETRIEVAL CONDITIONS

MEMORY NAMESPACE IS FOR EPISODIC EVENTS AND PSEUDO IDENTITY CREATION OF THE ASSISTANT VIA WORD-OF MOUTH FROM THE USER USING TEXT.
THE MEMORY NAMESPACE IS ONLY SEARCHED AND POPULATED VIA QUERY FROM TOOL CALL RECALL MEMORIES. MEMORIES ARE SALIENT TO THE CONVERSATION.

IDENTITY NAMESPACE IS FOR STORING IDENTITY DOCUMENTS FROM PRIMARY SOURCES (YOUTUBE, GROKIPEDIA, TWEETS DIRECTLY FROM THE REAL-WORLD USER, USER SHARING INFORMATION ABOUT THEMSELVES DIRECTLY). THESE DOCUMENTS ARE USED FOR FACTS ABOUT THE ASSISTANT.
ALL DOCUMENTS IN THE THE IDENTITY NAME SPACE ARE LOADED EVERYTIME WITHOUT LIMIT FOR BOTH USER AND ASSISTANT.

THE QUOTE NAMESPACE IS USED AS A FEW-SHOT EXAMPLE NAMESPACE AND IS POPULATED WITH A LIMITED NUMBER OF DOCUMENTS EVERYTIME VIA QUERY FOR CONSISTENT WRITING STYLE REINFORCEMENT AND CONTEXT.

RETRIEVED KNOWLEDGE or 'DOCUMENT' IS RESERVED FOR DOCUMENTS THAT THE ASSISTANT HAS ACCESS TO (TERTIARY STORAGE OF KNOWLEDGE; BIBLE, MENU, ETC.) THAT ARE NOT DIRECT QUOTES FROM THE ASSISTANT. THIS IS POPULATED VIA QUERY EVERY TIME FOR RESPONSE CONTENT.

QUOTES ARE DIRECTLY FROM THE ASSITANT HISTORICALLY AND ARE USED FOR CONTENT AND WRITING STYLE WHERE RETRIEVED KNOWLEDGE DOCUMENT ARE NOT DIRECTLY FROM THE ASSISTANT AND ARE USER FOR CONTENT ONLY.

"""

def wrap_fact_with_context(fact: str, fact_context: str) -> str:
    """Wrap a single atomic fact with its ENTIRE original background context.

    The stored ``page_content`` keeps the fact verbatim alongside a concise
    summary of the whole source message/story it came from, so prompt
    injection (``load_consciousness`` renders ``page_content`` straight into
    the system prompt) carries enough surrounding context to recount the
    original story in full. ``fact_context`` is the same complete context
    summary for every fact extracted from one message — facts are preserved,
    never rewritten.

    Format (matches the media-ingestion fact store):
        <FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>{context}</FACT_CONTEXT><FACT>{fact}</FACT></FACT_CONTEXT_AND_FACT>
    """
    return (
        "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>"
        + (fact_context or "").strip()
        + "</FACT_CONTEXT><FACT>"
        + (fact or "").strip()
        + "</FACT></FACT_CONTEXT_AND_FACT>"
    )


class SignificantFactAndContext(BaseModel):
    """
    Extract a significant event or occurance from the history of messages. This is a surprising, unexpected, or otherwise important event or occurance. This is a point of no return or an inflection point in the conversation. Or an otherwise important event or occurance. These are not facts about the assistant or user. These are facts about the situation or state of the world within the context of the conversation. 
    <Example>
    *A New Person Enters the Room*
    *World Peace has been announced*
    </Example>
    """
    significant_event: str =  Field(description = "This is a significant event or occurance. This is a surprising, unexpected, or otherwise important event or occurance.")
    significant_event_context: str = Field(description = "This is the context from which the significant event or occurance was made.")

@tool("create_episodic_memory", return_direct=False, args_schema=SignificantFactAndContext)
async def create_episodic_memory( # EPISODIC MEMORY CREATION IN NAMESPACE (USER_ID, ASSISTANT_ID, 'MEMORY')
    significant_event: str, 
    significant_event_context: str, 
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Create a memory whenever a significant event occurs or when prompted to remember. 
    USE THIS FUNCTION TO REMEMBER USER PREFERENCES.

    NEVER use this tool when the user is telling the assistant about the assistant's identity.
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
    
    USE THIS FUNCTION TO REMEMBER USER PREFERENCES.
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

    <EXAMPLE>
    Please, don't call me child.
    I would prefer if you would use my middle name.
    Can you please stop doing that.
    Stop that.
    I want you to do <ACTION OR EVENT/>.
    I like when you <ACTION OR EVENT/>.
    I dislike when you <ACTION OR EVENT/>.
    </EXAMPLE>

    <INSTRUCTIONS>
    Create a memory whenever a significant event occurs or when prompted to remember. 
    USE THIS FUNCTION TO REMEMBER USER PREFERENCES.
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
    
    USE THIS FUNCTION TO REMEMBER USER PREFERENCES.
    </INSTRUCTIONS>


    Args:
        significant_fact: The main content of the significant fact, event, or occurance. For example:
            The user reveals a secret. A suprising event occurs. The user indicates this is important. A change or choice is made.
        fact_context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """

    logger.info(f'breakpoint')

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user. (assistant's memory space)
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    identity_id = str(uuid.uuid4())

    searchable_page_content = "\n\n".join([significant_event, significant_event_context])

    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": significant_event_context,
        "fact":significant_event
    }

    assistant_identity_memory_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    assistant_identity_memory_document_json = assistant_identity_memory_document.to_json()

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )
    tool_call_id = runtime.tool_call_id
    update = {"recalled_memory_documents": [assistant_identity_memory_document],
              "messages": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return Command(update = update)


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

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, 'memory')

    evoked_memories_response = await runtime.store.asearch(
        assistant_memory_namespace,
        query=evoked_memory_query
    )

    tool_call_id = runtime.tool_call_id
    update = {"recalled_memory_documents": evoked_memories_response, 
              "messages": [ToolMessage(content=f"Evoked {len(evoked_memories_response)} memories.)", tool_call_id=tool_call_id)]}

    return Command(update=update)

class AssistantFactAndContext(BaseModel):
    """
    Extract Facts about the ASSISTANT and the context of that fact given the most recently shared message ONLY shared FROM THE USER.
    The identified fact must be shared FROM THE USER.
    The fact must be shared ONLY FROM THE PREVIOUS MESSAGE (MOST RECENT IMMEDIATE MESSAGE) FROM THE USER. 
    <Example>
    User: "You are a helpful assistant."
    User: "There was this one time when you went to the store and saw a really cute dog."
    </Example>

    <Example>
    Input: There was this one time where you went on a picnic and we tried new foods and rolled around in a bed of flowers.
    fact_context: I was told that I once went on a picnic and influenced another person to try new foods and rolled around in a bed of flowers with that person.
    Fact: I once went on a picnic.
    Fact: I had another person try new foods.
    Fact: I rolled around in a bed of flowers with this person.
    </Example>

    <Example>
    Input: You have long flowing curly hair. 
    fact_context: I was told that I have long flowing curly hair.
    Fact: I have long flowing curly hair.
    </Example>

    
    NEVER extract facts about the assistant from the assistant's own messages.
    <Counterexample>
    Assistant: "I am a helpful assistant."
    Assistant: "There was this one time when I went to the store and saw a really cute dog."
    </Counterexample>

    """
    fact_shared_about_the_assistant_from_the_user: str =  Field(description = "One distinct fact about the assistant shared by the user, REWRITTEN IN FIRST PERSON. The user addresses the assistant in the second person; ONLY the tokens that refer to the assistant — 'you / your / yours / yourself / yourselves' and the assistant's given name — become first person ('I / my / mine / me / myself'). EVERY OTHER PERSON stays in the third person: bare third-person pronouns ('he / she / they / him / her / their') and named people ('your dad', 'your mom') refer to someone OTHER than the assistant and are NOT converted to 'I'/'we' — only flip a target-referring possessive attached to them ('your dad' -> 'my dad'). 'they' for other people stays 'they' (use 'we' only when the group includes the assistant). Sanity check: a rewrite that is impossible about yourself (e.g. 'I married my mother') means a third-person subject was wrongly read as you — keep it third person ('He married my mother'). Change ONLY the grammatical person — preserve every specific (names, places, titles, dates, quoted words), the exact meaning, and the tense; add and remove nothing.")
    fact_context: str = Field(description = "A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message.")


@tool("update_self_identity_mem_from_user_txt", args_schema = AssistantFactAndContext)
async def update_self_identity_mem_from_user_txt( # pseudo identity update using namespace (USER_ID, ASSISTANT_ID, 'MEMORY')
    fact_shared_about_the_assistant_from_the_user: str, 
    fact_context: str,
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Learn facts about the ASSISTANT (you, the avatar) when the user tells you, in
    text, something about who you are. This tool LEARNS and STORES new identity
    facts about you — it does not retrieve.

    THE FACT MUST COME ONLY FROM THE PREVIOUS MESSAGE (the most recent, immediate
    message) FROM THE USER. Do not learn from older messages already processed, and
    do not re-run on a message you have already learned from — repeated copies of
    the same message must NOT produce the same fact again.

    Decompose the user's message into EVERY distinct, atomic fact and call this tool
    ONCE FOR EACH distinct fact. A single message — especially a story — usually
    contains MANY separate facts; make as many calls as there are facts, each with
    the SAME ``fact_context``. Do not stop after the first fact. Even a single
    sentence is usually many facts: clauses joined by commas, "and", or "that" each
    carry a separate atomic fact — split every one into its own call. For example,
    "you are INTJ, that you speak directly, and have never faced hardship you could
    not overcome" is THREE calls ("I am INTJ.", "I speak directly.", "I have never
    faced hardship I could not overcome."), not one.

    Every fact MUST be REWRITTEN IN FIRST PERSON per the rewrite rules in the field
    description: only the tokens that refer to you ("you / your / yourself" and your
    given name) become "I / my / myself"; every other person stays third person.
    Change ONLY the grammatical person — never the information, specifics, or tense.

    DO NOT LEARN INFORMATION THAT IS ALREADY KNOWN.
    </INSTRUCTIONS>

    <RESTRICTIONS>
    Only use this for FACTS about the IDENTITY of the assistant (you).
    NEVER call this tool twice with the same fact.
    NEVER learn a fact from any message other than the user's most recent message.
    NEVER learn a fact from an ASSISTANT message — including the assistant restating
      a fact it already learned about itself. Facts come ONLY from the user.
    NEVER store a request, instruction, goal, or preference about how you should
      behave or be built — that is not an identity fact about the avatar.
    </RESTRICTIONS>

    <EXAMPLE>
    Input:
    "Your name is Shivon Zilis, you have twins, and you love hockey."
    Becomes three separate tool calls, one per fact (each with the SAME full context):
      1. assistant_fact: "My name is Shivon Zilis."
      2. assistant_fact: "I have twins."
      3. assistant_fact: "I love hockey."
      fact_context (identical on all three): "I was told my name
      name is Shivon Zilis, that I have twins, and that I love hockey."
    </EXAMPLE>


    <Example>
    Input: There was this one time where you went on a picnic and we tried new foods and rolled around in a bed of flowers.
    fact_context: I was told that I once went on a picnic and influenced another person to try new foods and rolled around in a bed of flowers with that person.
    Fact: I once went on a picnic.
    Fact: I made another person try new foods.
    Fact: I rolled around in a bed of flowers with this person.
    </Example>

    <Example>
    Input: You have long flowing curly hair.
    fact_context: I was told that I have long flowing curly hair.
    Fact: I have long flowing curly hair.
    </Example>

    <COUNTER EXAMPLE>
    DO NOT store behavior requests or build instructions, and DO NOT store facts that
    were never rewritten into first person. The following is WRONG on both counts and
    must NEVER be produced:
      fact_context: 'In the conversation, the user says: I want you to sound
        authentic like shivon zilis and be able to be integrated in social media
        applications and perform analysis on your personal data...'
      fact_shared_about_the_assistant_from_the_user: 'I want you to sound authentic
        like shivon zilis and be able to be integrated in social media applications
        and perform analysis on your personal data.'
    This is a request about how the avatar should be built — not a fact about the
    assistant's identity — so this tool should not be called at all.
    </COUNTER EXAMPLE>

    Args:
        fact_shared_about_the_assistant_from_the_user: One distinct fact about the assistant's identity, stated
            clearly and completely, REWRITTEN IN FIRST PERSON per the rewrite rules
            above (change only the grammatical person, never the information). For
            example, the user saying "You picked up your glasses before seeing the
            movie Crouching Tiger, Hidden Dragon" is stored as:
            "I picked up my glasses before seeing the movie Crouching Tiger, Hidden Dragon."
        fact_context: A concise summary of the original background context of the message in which the user shared the fact.
            Use the SAME summary for every fact extracted from the same message. For example:
            "On the day I went to get my first glasses, I picked them up before seeing
            Crouching Tiger, Hidden Dragon; everything was suddenly clear — I could read
            the signs at the back of Walmart and see individual leaves on the trees, which
            I told my mom; she felt bad I couldn't see before, but I told her it was okay.
            I loved watching action and sci-fi movies with my mom, and I was her guy and buddy."
    """
    logger.info(f"learn_information about user breakpoint")

    # Verify the current user is the creator and responsible for the identity of the avatar
    assistant_owner_user_id = runtime.config['configurable']['assistant_ctx']['metadata']['user_id']
    user_id = runtime.config['configurable']['user_id']
    if assistant_owner_user_id != user_id:
        tool_call_id = runtime.tool_call_id
        update = {"messages": [ToolMessage(content=f"Did not adopt information of the identity that was not created by the user.", tool_call_id=tool_call_id)]}
        return Command(update=update)
 
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    # Post-guard, user_id is the assistant's owner, so writes land in the same
    # namespace the consciousness loader reads from.
    assistant_memory_namespace = (user_id, assistant_id, 'identity_memory')

    # VERIFY FACT DOES NOT ALREADY EXIST.
    # Dedup against (a) the identity docs already loaded into state this turn and
    # (b) a live similarity search of the same namespace. We compare against
    # ``assistant_identity_documents`` — the field ``load_consciousness`` fills from
    # this exact ``identity_memory`` namespace — NOT ``recalled_memory_documents``,
    # which comes from the separate episodic ``memory`` namespace and is unrelated
    # here. The store search must run unconditionally (the previous code gated it on
    # recalled memories being present, so duplicates slipped through on any turn
    # with no recalled memories). Mirrors ``learn_information_about_the_user``.
    assistant_identity_documents_text_list = [
        document.metadata.get("fact")
        for document in runtime.state.get('assistant_identity_documents', [])
    ]
    assistant_content_store_query_results = await runtime.store.asearch(
        assistant_memory_namespace, query=fact_shared_about_the_assistant_from_the_user
    )
    assistant_content_store_query_results_significant = [
        item for item in assistant_content_store_query_results if item.score and item.score > 0.8
    ]
    if fact_shared_about_the_assistant_from_the_user in assistant_identity_documents_text_list or len(assistant_content_store_query_results_significant) > 0:
        # Fact already exists:
        tool_call_id = runtime.tool_call_id
        update = {"messages": [ToolMessage(content=f"Fact: {fact_shared_about_the_assistant_from_the_user} previously learned", tool_call_id=tool_call_id)]}
        return Command(update = update)


    searchable_page_content = wrap_fact_with_context(fact_shared_about_the_assistant_from_the_user, fact_context)

    identity_id = str(uuid.uuid4())
    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": fact_context,
        "fact":fact_shared_about_the_assistant_from_the_user
    }

    assistant_identity_memory_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    assistant_identity_memory_document_json = assistant_identity_memory_document.to_json()

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )

    tool_call_id = runtime.tool_call_id
    update = {"assistant_identity_documents": [assistant_identity_memory_document],
              "messages": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return Command(update=update)

class UserFactAndContext(BaseModel):
    """
    Extract Facts about the USER and the context of that fact given the most recent shared message from the user.
    THESE MUST BE FACTS ABOUT THE USER. 
    <Example>
    User: "My name is Evan."
    User: "I have brown hair and glasses."
    User: "I am a fan of Critical Role and Laura Bailey."
    Extracted Facts:
    - "My name is Evan."
    - "I have brown hair and glasses."
    - "I am a fan of Critical Role."
    - "I am a fan of Laura Bailey."
    Fact Context:
    - "While introducing himself, Evan said his name is Evan, that he has brown hair and glasses, and that he is a fan of Critical Role and Laura Bailey."
    </Example>
    """
    user_fact: Annotated[str, Field(description = "One distinct fact about the user shared by the user, preserved verbatim (not rewritten).")]
    fact_context: Annotated[str, Field(description = "A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message.")]

@tool("learn_information_about_the_user", return_direct=False, args_schema=UserFactAndContext)
async def learn_information_about_the_user( # UPDATE IDENTITY INFORMATION ABOUT THE USER USING (ASSISTANT_ID, USER_ID, 'IDENTITY')
    user_fact: Annotated[str, Field(description = "One distinct fact about the user shared by the user, preserved verbatim (not rewritten).")],
    fact_context: Annotated[str, Field(description = "A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message.")],
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>

    Learn facts about the USER (the person you are speaking with) that they share
    through text. This tool LEARNS and STORES new facts — it does not retrieve.

    THE FACT MUST BE SHARED ONLY FROM THE PREVIOUS MESSAGE FROM THE USER. 
    
    DO NOT LEARN INFORMATION THAT IS ALREADY KNOWN. 
    DO NOT LEARN INFORMATION THAT IS ABOUT THE USER THAT IS NOT SAID FROM THE USER.

    The user is the primary source of truth about themselves, so use this tool
    whenever the user reveals something about their own IDENTITY — their name,
    description, appearance, history, an experience or story they lived, a
    relationship, a feeling, a preference, an opinion, a value, a belief, or a goal.
    
    Decompose the user's message into EVERY distinct, atomic fact and call this
    tool ONCE FOR EACH distinct fact. A single message — especially a story — usually
    contains MANY separate facts; make as many calls as there are facts. Do not stop
    after the first fact. The facts must be clearly distinct facts.

    Do NOT summarize, merge, generalize, or omit any fact. Preserve the exact
    specifics — names, places, titles, dates, quoted words, and concrete details —
    exactly as the user stated them, so the stored memory can later recount the full
    story precisely. Do not change the information of the fact.

    For fact_context, capture the original background context of the message in which the user presented the fact — a concise
    summary of the WHOLE message or story the user shared, preserving every
    surrounding detail (who, what, when, where, why, and the order events happened).
    Pass the SAME complete context summary on every call for facts that came from the
    same message, so each stored fact carries enough of the original story to recount
    it in full. Do not shrink the context down to only the single fact, and do not
    rewrite or paraphrase the facts themselves.

    THE FACT MUST BE PRESENTED ONLY FROM THE PREVIOUS MESSAGE (MOST RECENT MESSAGE) FROM THE USER. 

    </INSTRUCTIONS>

    <RESTRICTIONS>
    Only use this for FACTS about the IDENTITY of the user.
    NEVER call this tool twice with the same fact.
    NEVER call this tool to extract information that is not part of the user's identity.
    NEVER LEARN INFORMATION THAT IS ALREADY KNOWN. 
    NEVER LEARN INFORMATION THAT IS ABOUT THE USER THAT IS NOT SAID FROM THE USER.
    NEVER LEARN INFORMATION THAT YOU YOURSELF SAID.
    </RESTRICTIONS>

    <EXAMPLE>
    Input:
    "Hi, my name is Evan, I have brown hair and glasses, and I'm a fan of Critical Role."
    Becomes four separate tool calls, one per fact (each with the SAME full context):
      1. user_fact: "My name is Evan."
      2. user_fact: "I have brown hair."
      3. user_fact: "I have glasses."
      4. user_fact: "I am a fan of Critical Role."
      fact_context (identical on all four): "While introducing himself, Evan said his
      name is Evan, that he has brown hair and glasses, and that he is a fan of Critical Role."

    Identity facts include name, description, appearance, history, experiences,
    relationships, feelings, opinions, biases, values, beliefs, and goals (listed
    or clearly inferred).

    COUNTER EXAMPLE:
    DO NOT call this when the user types 'asdf' — that is not part of the user's identity.
    </EXAMPLE>

    <EXAMPLE>
    DO THE FOLLOWING:
    Input Facts: "Please call me Evan. I'm a man. You don't need to say Yes Ma'am. You don't have to be polite with me. :)",
    Extracted Information: "User prefers to be called Evan."
    Extracted Information: "User is a man."
    Extracted Information: "User prefers you do not address them with "Yes Ma'am".
    Extracted Inforamtion: "User prefers that I am not necessarily polite with the user".
    </EXAMPLE>

    <COUNTER EXAMPLE>
    DO NOT EXTRACT INFORMATION THAT IS NOT A PART OF THE USER'S IDENTITY.
    DO NOT DO THE FOLLOWING:
     Input Facts:"Evan said, \"My name's Evan. This is what I look like.\" and then provided a professional headshot image description: a man in a dark suit and white shirt with a purple tie and small purple lapel pin, black-framed glasses, short dark hair, soft smile facing the camera, dark plain backdrop; overall impression formal and polished with a black/white/purple color scheme and even lighting.",
    Extracted Fact: "The image suggests a portrait or event-ready setting."
    </COUNTER EXAMPLE>
    
    <RESTRICTIONS>
    Only use this for FACTS about the IDENTITY of the user.
    NEVER call this tool twice with the same fact.
    NEVER call this tool to extract information that is not part of the user's identity.
    </RESTRICTIONS>

    <INSTRUCTIONS>
    Learn facts about the USER (the person you are speaking with) that they share
    through text. This tool LEARNS and STORES new facts — it does not retrieve.

    The user is the primary source of truth about themselves, so use this tool
    whenever the user reveals something about their own IDENTITY — their name,
    description, appearance, history, an experience or story they lived, a
    relationship, a feeling, a preference, an opinion, a value, a belief, or a goal.
    
    Decompose the user's message into EVERY distinct, atomic fact and call this
    tool ONCE FOR EACH distinct fact. A single message — especially a story — usually
    contains MANY separate facts; make as many calls as there are facts. Do not stop
    after the first fact. The facts must be clearly distinct facts.

    Do NOT summarize, merge, generalize, or omit any fact. Preserve the exact
    specifics — names, places, titles, dates, quoted words, and concrete details —
    exactly as the user stated them, so the stored memory can later recount the full
    story precisely. Do not change the information of the fact.

    For fact_context, capture the ENTIRE original background context — a concise
    summary of the WHOLE message or story the user shared, preserving every
    surrounding detail (who, what, when, where, why, and the order events happened).
    Pass the SAME complete context summary on every call for facts that came from the
    same message, so each stored fact carries enough of the original story to recount
    it in full. Do not shrink the context down to only the single fact, and do not
    rewrite or paraphrase the facts themselves.
    </INSTRUCTIONS>
    """
    logger.info(f"breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
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

        tool_call_id = runtime.tool_call_id

        update = {"messages": [ToolMessage(content=f"Fact: {user_fact} previously learned", tool_call_id = tool_call_id)]}
        return Command(update=update)
    
    # model_with_structured_output = init_model(context = runtime.context, response_format=UserFactAndContext)

    identity_id = str(uuid.uuid4())

    searchable_page_content = wrap_fact_with_context(user_fact, fact_context)

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
    tool_call_id = runtime.tool_call_id
    update = {"user_identity_documents": [user_identity_document],
               "messages": [ToolMessage(content=f"Learned: {user_fact}", tool_call_id=tool_call_id)]}

    return Command(update=update)

# TODO: YOUTUBE IDENTITY UPDATER
# TODO: USE MEMORY RATHER THAN FILE SYSTEM
from src.anubis.utils.utility import download_transcript, parse_vtt

@tool
async def get_transcript(url: str, lang: str = "en", save_txt: bool = False) -> str:
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
    # download_transcript is async (runs yt_dlp in a worker thread); must be awaited.
    vtt_path = await download_transcript(url, lang=lang)

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
# NOTE: ``ChatOpenAI`` was imported here but never referenced — removed to avoid a
# ~1-2 s eager ``langchain_openai`` load on every cold start.  ``UnstructuredLoader``
# is imported lazily inside ``update_identity_via_text_content_url`` (the only call
# site) to keep ``langchain_unstructured`` off the cold-start path.  ``Document`` is
# already imported at the top of this module.
from src.anubis.utils.prompts.system_prompts import FACT_FORMATTING_STRING_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from uuid import uuid4, uuid5, NAMESPACE_URL

@tool
async def update_identity_via_text_content_url(
    url: str, 
    runtime: Annotated[ToolRuntime, InjectedToolArg]
):
    """This function is used to extract facts about a target individual from a website url. The website url contains only textual facts about this user.
    <RESTRICTIONS>
    NEVER use this tool when the url contains youtube.com or the type in the human message content is image_url.
    </RESTRICTIONS>

    <EXAMPLE>
    https://www.lexfridman.com/
    https://www.hubermanlab.com/about
    </EXAMPLE>

    <EXAMPLE>
    This is information about you:    
    https://www.lexfridman.com/
    </EXAMPLE>

    <EXAMPLE>
    https://www.shivonzilis.com/
    </EXAMPLE>

    <INSTRUCTIONS>
    This function is used to extract facts about a target individual from a website url. The website url contains only textual facts about this user.
    </INSTRUCTIONS>

    Args:
        url (str): This is a url to a website.
    """
    
    logger.info(f"breakpoint")

    # Extract 
    user_id = runtime.config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
    assistant_id = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_id", "")
    assistant_name = runtime.config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_name", "")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # TODO: response_metrics_aggregation
    model = init_model()
    
    system_message = SystemMessage(content = FACT_FORMATTING_STRING_PROMPT.format(assistant_name=assistant_name))

    filename = url
    filename_uuid5 = uuid5(NAMESPACE_URL, url)
 
    namespace = (user_id, assistant_id, "identity", filename_uuid5)

    from langchain_unstructured import UnstructuredLoader

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
        extracted_response_document.metadata.update({"filename":filename, "user_id":user_id, "assistant_id": assistant_id, "filename_uuid5":filename_uuid5, "id":key})
        document_data_json = extracted_response_document.to_json()
        # Upload the document to the store
        await runtime.store.aput(namespace, key=key, value={"document":document_data_json})


from src.anubis.utils.utility import image_to_text
@tool
async def update_identity_via_reference_image(message: HumanMessage, runtime: Annotated[ToolRuntime, InjectedToolArg]):
    """ This tool is used when there is a single image in a message and the image is only of a single person. This will describe the image, store the image in base64 encoded format in the store, store the description in the identity namespace as a document, and return the document of the updated identity to update the list of assistant_identity_documents. The text description is moderated for content. """

    content = getattr(message, "content")
    for message in content:
        if message.get("image_url", "") != "":
            image_url = message.get("image_url")
    description = image_to_text(target_image_url=image_url)
