"""Agent SubGraph Tools"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg
from langgraph.types import Command, interrupt

from src.anubis.utils.model import init_model
from src.anubis.utils.state import GlobalState
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

    significant_event: str = Field(
        description="This is a significant event or occurance. This is a surprising, unexpected, or otherwise important event or occurance."
    )
    significant_event_context: str = Field(
        description="This is the context from which the significant event or occurance was made."
    )


@tool(
    "create_episodic_memory", return_direct=False, args_schema=SignificantFactAndContext
)
async def create_episodic_memory(  # EPISODIC MEMORY CREATION IN NAMESPACE (USER_ID, ASSISTANT_ID, 'MEMORY')
    significant_event: str,
    significant_event_context: str,
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
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

    logger.info(f"breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user. (assistant's memory space)
    assistant_memory_namespace = (user_id, assistant_id, "memory")

    identity_id = str(uuid.uuid4())

    searchable_page_content = "\n\n".join(
        [significant_event, significant_event_context]
    )

    document_metadata = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": significant_event_context,
        "fact": significant_event,
    }

    assistant_identity_memory_document = Document(
        page_content=searchable_page_content, metadata=document_metadata
    )
    assistant_identity_memory_document_json = (
        assistant_identity_memory_document.to_json()
    )

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )
    tool_call_id = runtime.tool_call_id
    update = {
        "recalled_memory_documents": [assistant_identity_memory_document],
        "messages": [
            ToolMessage(
                content=f"Learned: {document_metadata['fact']}",
                tool_call_id=tool_call_id,
            )
        ],
    }

    return Command(update=update)


class ConversationSummaryToProvokeMemories(BaseModel):
    """
    Summarize the conversation and generate a query that is the similar to the content to which a retrieved list of documents is the response.
    These document memories have been shared from the user and contain information about the assistant's identity.
    These document memories have been shared from the user and contain information that was important that the assistant decided the assistant needed to remember.
    """

    evoked_memory_query: str = Field(
        description="""
                                      This is the summary of the converation. This summary is posed succinctly such that the summary is similar to the content to which a retrived list of documents is the response. The document memories have been shared from the user and contain information about the assistant's identity or contain information that was important that the assistant decided the assistant needed to remember. It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories."""
    )


@tool(
    "recall_memories",
    return_direct=False,
    args_schema=ConversationSummaryToProvokeMemories,
)
async def recall_memories(
    evoked_memory_query: Annotated[
        str,
        Field(
            description="This is the summary of the converation. This summary is posed succinctly such that the summary is similar to the content to which a retrived list of documents is the response. The document memories have been shared from the user and contain information about the assistant's identity or contain information that was important that the assistant decided the assistant needed to remember. It is possible that the summary matches no memories because the intent of the conversation does not match the topics of the stored memories."
        ),
    ],
    # Hide these arguments
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
):  # EPISODIC MEMORY RETRIEVAL FROM NAMESPACE (USER_ID, ASSISTANT_ID, 'MEMORY')
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

    logger.info(f"breakpoint")

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    assistant_memory_namespace = (user_id, assistant_id, "memory")

    evoked_memories_response = await runtime.store.asearch(
        assistant_memory_namespace, query=evoked_memory_query
    )

    tool_call_id = runtime.tool_call_id
    update = {
        "recalled_memory_documents": evoked_memories_response,
        "messages": [
            ToolMessage(
                content=f"Evoked {len(evoked_memories_response)} memories.)",
                tool_call_id=tool_call_id,
            )
        ],
    }

    return Command(update=update)


def _extract_message_text(content: object) -> str:
    """Flatten a message ``content`` (str or list of content blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return str(content or "")


def _latest_user_message_text(messages: object) -> str:
    """Return the text of the most recent ``HumanMessage`` in ``messages`` (else "")."""
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage):
            return _extract_message_text(message.content)
    return ""


class _UserMessageGroundsFact(BaseModel):
    """Whether the user's most recent message is the actual source of a proposed identity fact.

    Guards ``update_self_identity_mem_from_user_txt`` against learning facts the avatar surfaced
    from its own retrieved consciousness (identity/quote transcripts injected into the system
    prompt) or from earlier in the conversation, rather than from something the user just shared.
    """

    user_asserted_the_fact: bool = Field(
        description=(
            "True ONLY if the user's most recent message itself states/shares this fact about "
            "the assistant — i.e. the information the fact was derived from is present in that "
            "message. False if the message merely ASKS about, requests, or mentions the topic "
            "without asserting it, or if the fact could only have come from retrieved documents, "
            "a transcript, the assistant's own words, or an earlier message."
        )
    )
    reason: str = Field(description="One short sentence explaining the decision.")


_FACT_GROUNDING_SYSTEM_PROMPT = """You are a strict gatekeeper deciding whether a proposed \
fact about the ASSISTANT was actually shared by the user in their MOST RECENT message.

You are given:
- USER_MESSAGE: the user's most recent message, verbatim.
- PROPOSED_FACT: a first-person fact about the assistant that another component wants to store.

Set `user_asserted_the_fact` true ONLY when USER_MESSAGE itself contains the information the \
PROPOSED_FACT was derived from — the user is telling the assistant this is true about it.

Set it false when:
- USER_MESSAGE only ASKS about, requests, or mentions the topic (e.g. "tell me about X", \
"spell your name", "how does Y affect you?") without asserting the fact.
- The fact could only have come from retrieved documents, a transcript, the assistant's own \
prior statements, or an earlier message — not from THIS message.
- USER_MESSAGE does not contain the specific information in the fact at all.

A question or request is never an assertion. When in doubt, answer false.
"""


async def _user_message_grounds_fact(fact: str, user_message_text: str) -> bool:
    """Verify the user's most recent message is the source the ``fact`` was derived from.

    Safeguard for ``update_self_identity_mem_from_user_txt``: the model frequently "learns"
    facts it actually surfaced from its own retrieved consciousness (identity/quote documents)
    when the user merely ASKS about those topics. We re-check the proposed fact against the
    latest user message only — not retrieved context, the assistant's own messages, or earlier
    turns. Returns True when that message asserts the fact, False otherwise.

    Fails OPEN (returns True) on an empty message or model error so transient failures never
    silently drop a genuine fact — mirrors ``_suggest_correction``'s graceful fallback.
    """
    if not user_message_text.strip():
        return True
    try:
        model = init_model(response_format=_UserMessageGroundsFact)
        result = await model.ainvoke(
            [
                SystemMessage(content=_FACT_GROUNDING_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"USER_MESSAGE: {user_message_text}\n\nPROPOSED_FACT: {fact}"
                ),
            ]
        )
        return bool(result.user_asserted_the_fact)
    except Exception:
        logger.exception(
            "update_self_identity_mem_from_user_txt: fact-grounding check failed; allowing the fact"
        )
        return True


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

    fact_shared_about_the_assistant_from_the_user: str = Field(
        description="One distinct fact about the assistant shared by the user, REWRITTEN IN FIRST PERSON. The user addresses the assistant in the second person; ONLY the tokens that refer to the assistant — 'you / your / yours / yourself / yourselves' and the assistant's given name — become first person ('I / my / mine / me / myself'). EVERY OTHER PERSON stays in the third person: bare third-person pronouns ('he / she / they / him / her / their') and named people ('your dad', 'your mom') refer to someone OTHER than the assistant and are NOT converted to 'I'/'we' — only flip a target-referring possessive attached to them ('your dad' -> 'my dad'). 'they' for other people stays 'they' (use 'we' only when the group includes the assistant). Sanity check: a rewrite that is impossible about yourself (e.g. 'I married my mother') means a third-person subject was wrongly read as you — keep it third person ('He married my mother'). Change ONLY the grammatical person — preserve every specific (names, places, titles, dates, quoted words), the exact meaning, and the tense; add and remove nothing."
    )
    fact_context: str = Field(
        description="A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message."
    )


@tool("update_self_identity_mem_from_user_txt", args_schema=AssistantFactAndContext)
async def update_self_identity_mem_from_user_txt(  # pseudo identity update using namespace (USER_ID, ASSISTANT_ID, 'MEMORY')
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

    DO NOT CALL THIS TOOL ON INFORMATION THAT WAS PRESENTED FROM THE ASSISTANT IN EARLIER CONTEXT OF THE CONVERSATION (INFORMATION YOU PRESENTED YOURSELF)

    <FACTCONTEXT>
    THE CONTEXT OF THE FACTS ARE SUCH THAT YOU HAVE BEEN INFORMED OF THIS INFORMATION.
    THESE ARE FACTS SHARED IN CONVERSATION FROM THE USER. DO NOT REWRITE THESE FACTS AS IF YOU SAID THESE FACTS YOURSELF.
    INSTEAD, INDICATE THAT YOU HAVE BEEN TOLD OR HAVE ON GOOD AUTHORITY THIS INFORMATION.

    NEVER DO THE FOLLOWING:
    <FACT_CONTEXT>You said: "You have been interested in AI after reading a book in your youth, that you speak directly and openly as if you are not afraid of the world and have never experienced harm or evil or hardship that you could not overcome."</FACT_CONTEXT>
    <FACT>I have been interested in AI after reading a book in my youth.</FACT></FACT_CONTEXT_AND_FACT>

    THE CORRECT FACT CONTEXT IS THE FOLLOWING:
    I was told that I have been interested in AI after reading a book in my youth.

    </FACTCONTEXT>

    </INSTRUCTIONS>

    <RESTRICTIONS>
    DO NOT CALL THIS TOOL ON INFORMATION THAT WAS PRESENTED FROM THE ASSISTANT IN EARLIER CONTEXT OF THE CONVERSATION (INFORMATION YOU PRESENTED YOURSELF)
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
    assistant_owner_user_id = runtime.config["configurable"]["assistant_ctx"][
        "metadata"
    ]["user_id"]
    user_id = runtime.config["configurable"]["user_id"]
    if assistant_owner_user_id != user_id:
        tool_call_id = runtime.tool_call_id
        update = {
            "messages": [
                ToolMessage(
                    content=f"Did not adopt information of the identity that was not created by the user.",
                    tool_call_id=tool_call_id,
                )
            ]
        }
        return Command(update=update)

    # SAFEGUARD: only learn a fact the user actually shared in their MOST RECENT message.
    # The model often surfaces facts from the avatar's own retrieved consciousness
    # (identity/quote transcripts injected into the system prompt) when the user merely ASKS
    # about those topics, then "learns" them as if the user had asserted them. Verify the fact
    # was derived from the latest user message — not retrieved context, the assistant's own
    # words, or an earlier turn — before storing anything.
    latest_user_message_text = _latest_user_message_text(runtime.state.get("messages"))

    _SIMILARITY_THRESHOLD = 0.6

    def _compute_message_fact_similarity() -> float:
        from src.anubis.utils.runtime_handles import get_sentence_embedder

        model = get_sentence_embedder()
        message_embedding, fact_embedding = model.encode(
            [latest_user_message_text, fact_shared_about_the_assistant_from_the_user],
            convert_to_numpy=True,
        )
        scores = model.similarity(message_embedding, fact_embedding)
        return float(scores[0][0])

    similarity = await asyncio.to_thread(_compute_message_fact_similarity)
    if similarity < _SIMILARITY_THRESHOLD:
        """ 
            If the fact is not similar to the most recent user message 
            (the fact should come from the user message), then verify with an llm. 
            If that fails, do not learn the fact.
        """

        if not await _user_message_grounds_fact(
            fact_shared_about_the_assistant_from_the_user, latest_user_message_text
        ):
            tool_call_id = runtime.tool_call_id
            update = {
                "messages": [
                    ToolMessage(
                        content=(
                            f'Not learned: "{fact_shared_about_the_assistant_from_the_user}" was '
                            "not shared by the user in their most recent message."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
            return Command(update=update)

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    # Post-guard, user_id is the assistant's owner, so writes land in the same
    # namespace the consciousness loader reads from.
    assistant_memory_namespace = (user_id, assistant_id, "identity_memory")

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
        for document in runtime.state.get("assistant_identity_documents", [])
    ]
    assistant_content_store_query_results = await runtime.store.asearch(
        assistant_memory_namespace, query=fact_shared_about_the_assistant_from_the_user
    )
    assistant_content_store_query_results_significant = [
        item
        for item in assistant_content_store_query_results
        if item.score and item.score > 0.8
    ]
    if (
        fact_shared_about_the_assistant_from_the_user
        in assistant_identity_documents_text_list
        or len(assistant_content_store_query_results_significant) > 0
    ):
        # Fact already exists:
        tool_call_id = runtime.tool_call_id
        update = {
            "messages": [
                ToolMessage(
                    content=f"Fact: {fact_shared_about_the_assistant_from_the_user} previously learned",
                    tool_call_id=tool_call_id,
                )
            ]
        }
        return Command(update=update)

    searchable_page_content = wrap_fact_with_context(
        fact_shared_about_the_assistant_from_the_user, fact_context
    )

    identity_id = str(uuid.uuid4())
    document_metadata = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "document_id": identity_id,
        "fact_context": fact_context,
        "fact": fact_shared_about_the_assistant_from_the_user,
    }

    assistant_identity_memory_document = Document(
        page_content=searchable_page_content, metadata=document_metadata
    )
    assistant_identity_memory_document_json = (
        assistant_identity_memory_document.to_json()
    )

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )

    tool_call_id = runtime.tool_call_id
    update = {
        "assistant_identity_documents": [assistant_identity_memory_document],
        "messages": [
            ToolMessage(
                content=f"Learned: {document_metadata['fact']}",
                tool_call_id=tool_call_id,
            )
        ],
    }

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

    user_fact: Annotated[
        str,
        Field(
            description="One distinct fact about the user shared by the user, preserved verbatim (not rewritten)."
        ),
    ]
    fact_context: Annotated[
        str,
        Field(
            description="A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message."
        ),
    ]


@tool(
    "learn_information_about_the_user",
    return_direct=False,
    args_schema=UserFactAndContext,
)
async def learn_information_about_the_user(  # UPDATE IDENTITY INFORMATION ABOUT THE USER USING (ASSISTANT_ID, USER_ID, 'IDENTITY')
    user_fact: Annotated[
        str,
        Field(
            description="One distinct fact about the user shared by the user, preserved verbatim (not rewritten)."
        ),
    ],
    fact_context: Annotated[
        str,
        Field(
            description="A concise summary of the ENTIRE original background context (the whole message/story) the fact came from, not just this one fact. Use the SAME summary for every fact extracted from the same message."
        ),
    ],
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

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Identity of the user from the assistant's perspective
    user_identity_namespace = (assistant_id, user_id, "identity")

    # VERIFY FACT DOES NOT ALREADY EXIST
    user_identity_documents_text_list = [
        document.metadata.get("fact")
        for document in runtime.state["user_identity_documents"]
    ]
    user_content_store_query_results = await runtime.store.asearch(
        user_identity_namespace, query=user_fact
    )
    user_content_store_query_results_significant = [
        item for item in user_content_store_query_results if item.score > 0.8
    ]
    if (
        user_fact in user_identity_documents_text_list
        or len(user_content_store_query_results_significant) > 0
    ):
        # Fact already exists:

        tool_call_id = runtime.tool_call_id

        update = {
            "messages": [
                ToolMessage(
                    content=f"Fact: {user_fact} previously learned",
                    tool_call_id=tool_call_id,
                )
            ]
        }
        return Command(update=update)

    # model_with_structured_output = init_model(context = runtime.context, response_format=UserFactAndContext)

    identity_id = str(uuid.uuid4())

    searchable_page_content = wrap_fact_with_context(user_fact, fact_context)

    document_metadata = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "document_id": identity_id,
        "fact_context": fact_context,
        "fact": user_fact,
    }

    user_identity_document = Document(
        page_content=searchable_page_content, metadata=document_metadata
    )
    user_identity_document_json = user_identity_document.to_json()

    await runtime.store.aput(
        user_identity_namespace,
        key=identity_id,
        value={"document": user_identity_document_json},
    )
    tool_call_id = runtime.tool_call_id
    update = {
        "user_identity_documents": [user_identity_document],
        "messages": [
            ToolMessage(content=f"Learned: {user_fact}", tool_call_id=tool_call_id)
        ],
    }

    return Command(update=update)


# ──────────────────────────────────────────────────────────────────────────────
# Conversational correction of inaccurate stored facts (human-in-the-loop)
# ──────────────────────────────────────────────────────────────────────────────
# Score above which a store hit is treated as the same fact the user is correcting.
# Calibrated to the PRODUCTION embedding model (``microsoft/harrier-oss-v1-270m``),
# whose cosine scores sit on the same compressed scale ``load_consciousness`` uses to
# decide a fact is relevant enough to inject every turn (``nodes._FILTER_SCORE = 0.1``).
# An earlier 0.8/0.6 gate — tuned to the unit-test mock embedding, which returns cosine
# 1.0 for any shared keyword — rejected every real match, so the sweep returned nothing
# and the tool reported "No stored fact matched" for facts that were actually stored.
# This path favors recall: the HITL interrupt is the safety net (the owner sees every
# proposed match and approves/edits/rejects; nothing is applied unapproved). Facts
# already retrieved into this turn's prompt are actively shaping the response, so they
# are matched at the loader's own relevance floor; everything else needs a small margin
# above it to avoid sweeping in loosely-related facts.

_CORRECTION_MATCH_THRESHOLD = 0.6  # retrieved documents from the store namespaces
_CORRECTION_RETRIEVED_DOC_THRESHOLD = (
    0.1  # all documents contained in the current state
)

# Sentence-level gate for long-text namespaces (quote/document/analysis). Those store
# raw multi-sentence text, so whole-document cosine of a short claim against a ~2,000-word
# transcript is near zero and clears no document-level threshold — the offending clause
# has to be located sentence-by-sentence instead. A single sentence that restates the
# claim scores much higher than the whole blob, so this gate sits above the document
# floor. Recall-biased and HITL-gated like the thresholds above; tune empirically.
_SENTENCE_MATCH_THRESHOLD = 0.45

# Namespace "kind" (3rd tuple element) whose stored ``page_content`` is raw, multi-
# sentence text rather than a single atomic ``<FACT>``-wrapped statement. These need
# sentence-level matching + in-place sentence redaction; the atomic-fact namespaces
# (identity, identity_memory, memory) are matched and rewritten whole.
_LONG_TEXT_NAMESPACE_KINDS = frozenset({"quote", "document", "analysis"})


def _namespace_is_long_text(namespace: tuple) -> bool:
    """True if ``namespace`` holds raw multi-sentence docs (quote/document/analysis)."""
    return len(namespace) >= 3 and namespace[2] in _LONG_TEXT_NAMESPACE_KINDS


# Cheap sentence splitter (mirrors ``dataset.stylistic_profile._split_sentences``;
# inlined to avoid importing that module's heavy numpy/burrows-delta dependency chain at
# tool-call time). Splits on sentence-final punctuation followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text.strip()]


async def _score_sentences(query: str, sentences: list[str]) -> list[float]:
    """Cosine similarity of ``query`` against each of ``sentences`` (same order).

    Embeds with the process-wide cached ``SentenceTransformer`` (same model as the store
    index, so scores are on the retrieval scale). The blocking ``encode`` runs in a
    worker thread so it does not stall the event loop — same pattern the media graph uses
    (``process_media_graph/utils/nodes.py``). Returns ``[]`` for empty input.
    """
    if not sentences:
        return []

    def _compute() -> list[float]:
        from src.anubis.utils.runtime_handles import get_sentence_embedder

        model = get_sentence_embedder()
        query_embedding = model.encode([query], convert_to_numpy=True)
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
        # Row 0 of the similarity matrix = query vs every sentence.
        similarities = model.similarity(query_embedding, sentence_embeddings)[0]
        return [float(score) for score in similarities]

    return await asyncio.to_thread(_compute)


def _item_document_kwargs(item) -> dict:
    """Pull ``{page_content, metadata}`` out of a store SearchItem value.

    Store values are ``{"document": Document.to_json()}`` and
    ``Document.to_json()`` is ``{"kwargs": {"page_content": ..., "metadata": ...}}``
    (same shape ``reduce_docs`` reads in ``utility.py``).
    """
    value = getattr(item, "value", None) or {}
    document = value.get("document") or {}
    kwargs = document.get("kwargs") or {}
    return kwargs if isinstance(kwargs, dict) else {}


def _item_fact(item) -> str | None:
    return (_item_document_kwargs(item).get("metadata") or {}).get("fact")


def _item_document_id(item) -> str | None:
    metadata = _item_document_kwargs(item).get("metadata") or {}
    # Identity/identity_memory/user-identity write ``document_id``; episodic memory
    # writes ``id`` — accept either so a correction reaches every namespace.
    return metadata.get("document_id") or metadata.get("id")


@dataclass
class FactMatch:
    """One thing a correction will change.

    ``kind == "fact"``  → a whole atomic-fact document (identity/identity_memory/memory);
    the entire stored fact is rewritten or the document deleted.
    ``kind == "sentence"`` → one offending sentence inside a long verbatim document
    (quote/document/analysis); only that sentence is redacted/replaced in place.
    """

    item: object
    namespace: tuple
    key: str
    kind: str  # "fact" | "sentence"
    matched_text: str | None  # the atomic fact, or the specific offending sentence
    score: float


@dataclass
class CorrectionChange:
    """Record of one applied change, for the HITL summary + state update."""

    action: str  # "rewrite" | "redact" | "delete"
    namespace: tuple
    key: str
    old_text: str | None
    new_text: str | None
    document: Document | None  # surviving Document (None when the doc was deleted)


@dataclass
class ResolvedCorrection:
    """A single match paired with the FINAL per-document edit to apply to it.

    Produced after the HITL panel resolves: each matched fact/sentence carries its own
    ``corrected_text``/``corrected_context`` (the owner's edit, or the model's per-document
    suggestion) and its own ``include`` flag. An empty ``corrected_text`` on an included
    match means "remove" (delete the atomic fact / redact the sentence).
    """

    match: FactMatch
    corrected_text: str
    corrected_context: str
    include: bool = True

    @property
    def is_removal(self) -> bool:
        return not (self.corrected_text or "").strip()


class ProposedFactEdit(BaseModel):
    """A per-document suggested minimal edit for ONE matched fact/sentence.

    The correction tool finds many stored texts that match the inaccurate claim; each one
    is a different sentence in a different document, so a single global replacement would
    flatten unrelated facts (see the failure cases in ``features/.../fp_correction``). This
    model is generated PER MATCH so only the offending span is changed and everything else
    in that specific text is preserved verbatim.
    """

    asserts_inaccurate_fact: bool = Field(
        description="True only if THIS specific text actually states the inaccurate fact the "
        "user is correcting. False when the matched text is about something else (it was swept "
        "in by a loose semantic match) and must be left untouched."
    )
    corrected_text: str = Field(
        default="",
        description="The matched text rewritten with ONLY the inaccurate portion fixed (or "
        "removed); every other word — names, places, dates, unrelated clauses — preserved "
        "exactly. First person, like the original. Return an EMPTY string to remove the matched "
        "text entirely (a pure deletion, or when the whole sentence only asserted the wrong fact).",
    )
    corrected_context: str = Field(
        default="",
        description="The text's ORIGINAL background context, minimally edited so it stays "
        "accurate — preserve the parts that are still true, change only what the correction "
        "touches. Do NOT blanket-overwrite the whole context with the new fact.",
    )


_CORRECTION_SUGGESTION_SYSTEM_PROMPT = """You revise ONE stored statement about a person so a \
single inaccurate fact is corrected while everything else is preserved exactly.

You are given:
- INACCURATE_FACT: the claim the user says is wrong.
- USER_CORRECTION: what the user said is actually true (may be empty for a pure deletion).
- MATCHED_TEXT: the specific stored sentence/fact that was matched and may need editing.
- FULL_DOCUMENT: the surrounding text MATCHED_TEXT came from (for context only — do NOT return it).
- ORIGINAL_CONTEXT: the stored background context for MATCHED_TEXT.

Rules:
- Decide `asserts_inaccurate_fact`: is MATCHED_TEXT actually stating INACCURATE_FACT? If it is \
about something else, set it false and leave the text unchanged.
- Edit MATCHED_TEXT minimally: fix or remove ONLY the inaccurate portion; keep every other \
detail (names, places, dates, other clauses) verbatim and in first person.
- If MATCHED_TEXT only asserted the wrong fact and has nothing true left, or the user is \
deleting the fact with no replacement, return an empty `corrected_text`.
- Edit ORIGINAL_CONTEXT the same way: keep the still-true parts, change only what the \
correction affects. Never replace the whole context with just the new fact.
"""


# Per-match suggestion calls fan out concurrently; bound them so a correction touching many
# documents does not open dozens of simultaneous model calls.
_SUGGESTION_CONCURRENCY = asyncio.Semaphore(8)


async def _suggest_correction(
    match: FactMatch,
    *,
    inaccurate_information: str,
    corrected_information: str,
    correction_context: str,
    is_deletion: bool,
) -> tuple[str, str, bool]:
    """Suggest a minimal in-place edit for ``match`` → ``(corrected_text, context, asserts)``.

    Uses a structured-output model so the suggestion only changes the offending span of THIS
    document. On any failure (no model configured, network error) it falls back to the global
    correction the model proposed, so the HITL flow still works — the owner can edit anything
    in the panel regardless. Mockable at module scope for tests.
    """
    fallback = ("" if is_deletion else corrected_information, correction_context, True)
    try:
        kwargs = _item_document_kwargs(match.item)
        full_document = kwargs.get("page_content") or ""
        original_context = (kwargs.get("metadata") or {}).get(
            "fact_context"
        ) or correction_context
        human_message = HumanMessage(
            content=(
                f"INACCURATE_FACT: {inaccurate_information}\n"
                f"USER_CORRECTION: {corrected_information or '(remove — the user says this is simply wrong)'}\n"
                f"MATCHED_TEXT: {match.matched_text}\n"
                f"ORIGINAL_CONTEXT: {original_context}\n"
                f"FULL_DOCUMENT: {full_document[:4000]}"
            )
        )
        async with _SUGGESTION_CONCURRENCY:
            model = init_model(response_format=ProposedFactEdit)
            result = await model.ainvoke(
                [
                    SystemMessage(content=_CORRECTION_SUGGESTION_SYSTEM_PROMPT),
                    human_message,
                ]
            )
        corrected_text = "" if is_deletion else (result.corrected_text or "")
        corrected_context = result.corrected_context or correction_context
        return corrected_text, corrected_context, bool(result.asserts_inaccurate_fact)
    except Exception:
        logger.exception(
            "correct_identity_fact: per-document suggestion failed; using global correction"
        )
        return fallback


def _match_preview(
    index: int,
    match: FactMatch,
    suggestion: tuple[str, str, bool],
    *,
    is_deletion: bool,
) -> dict:
    """Compact, JSON-serializable description of ONE editable pending change for the HITL panel.

    Each match is its own editable item. The owner chooses exactly one ``action`` per item
    (see the resume contract in ``correct_identity_fact``):

    - ``"skip"`` — leave this document completely unchanged (the DEFAULT for every item, so a
      falsely-retrieved doc is never touched unless the owner explicitly acts on it).
    - ``"accept"`` — apply the per-document ``suggested_text`` / ``suggested_context``.
    - ``"edit"`` — apply the owner's own ``corrected_text`` / ``correction_context``.
    - ``"remove"`` — delete the atomic fact / redact the offending sentence.

    ``default_action`` states the safe fallback explicitly; ``recommended_action`` is the
    model's suggestion the UI may pre-select (it is only a hint — nothing applies unless the
    owner's resume payload names an action). ``index`` correlates the reply back to ``matches``.
    """
    suggested_text, suggested_context, asserts_flag = suggestion
    kwargs = _item_document_kwargs(match.item)
    excerpt = kwargs.get("page_content") or ""
    # The stored background context, shown beside ``suggested_context`` so the owner can see
    # the original is preserved and only minimally edited in line with the correction.
    current_context = (kwargs.get("metadata") or {}).get("fact_context") or ""
    if not asserts_flag:
        # The model thinks this text does not actually state the inaccurate fact (a loose
        # semantic match), so the safe recommendation is to leave it alone.
        recommended_action = "skip"
    elif is_deletion or not (suggested_text or "").strip():
        # A pure deletion, or a sentence that only asserted the wrong fact with nothing true
        # left, can only be removed/redacted — there is no surviving text to keep.
        recommended_action = "remove"
    else:
        recommended_action = "accept"
    return {
        "index": index,
        "kind": match.kind,
        "namespace": list(match.namespace or []),
        "key": match.key,
        "document_id": _item_document_id(match.item),
        "current_text": match.matched_text,
        "current_context": current_context,
        "document_excerpt": excerpt[:1000],
        "suggested_text": suggested_text,
        "suggested_context": suggested_context,
        "default_action": "skip",
        "recommended_action": recommended_action,
        "score": match.score,
    }


def _correction_namespaces(
    creator_id: str, assistant_id: str, user_id: str
) -> list[tuple]:
    """Assistant-side namespaces a correction sweeps.

    ``(creator_id, assistant_id, "identity")`` is a *prefix* search — it also covers
    the media/URL sub-namespaces ``(creator_id, assistant_id, "identity", <uuid5>)``.
    ``analysis`` holds derived psycho-analysis traits (beliefs/relationships/OCEAN) that
    can also ground a response. User-identity is intentionally excluded (this tool
    corrects facts about the AVATAR, not the user).
    """
    return [
        (user_id, assistant_id, "memory"),
        (creator_id, assistant_id, "identity_memory"),
        (creator_id, assistant_id, "identity"),
        (creator_id, assistant_id, "document"),
        (creator_id, assistant_id, "quote"),
        (creator_id, assistant_id, "analysis"),
    ]


_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "have",
        "from",
        "they",
        "your",
        "about",
        "into",
        "also",
        "been",
        "were",
        "which",
        "their",
        "would",
        "there",
        "what",
        "when",
        "will",
        "i've",
        "ive",
        "never",
        "incorrect",
        "wrong",
    }
)


def _salient_tokens(text: str) -> set[str]:
    """Content words (len ≥ 4, non-stopword) used as a cheap document prefilter."""
    return {
        token
        for token in re.findall(r"[a-z']+", (text or "").lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }


async def find_fact_matches(
    store,
    *,
    creator_id: str,
    assistant_id: str,
    user_id: str,
    query: str,
    state_docs=None,
) -> list[FactMatch]:
    """Find everything matching ``query`` across the assistant namespaces.

    Two matching modes by namespace kind:

    - **Atomic-fact** (identity / identity_memory / memory): whole-document semantic
      match. Retrieved-docs-first — documents already pulled into this turn's prompt
      (``state_docs``, keyed by ``document_id``/``id``) clear the softer
      ``_CORRECTION_RETRIEVED_DOC_THRESHOLD``; everything else needs
      ``_CORRECTION_MATCH_THRESHOLD``.
    - **Long-text** (quote / document / analysis): the claim is one clause inside a
      multi-sentence blob, so whole-document cosine is meaningless. Retrieve all docs,
      split each into sentences, and keep sentences clearing
      ``_SENTENCE_MATCH_THRESHOLD``. A cheap salient-token prefilter skips docs that
      share no content word with the claim, to bound on-the-fly embedding.

    De-duplicated by ``(namespace, key)`` for facts and ``(namespace, key, sentence)``
    for sentences. Reads only — safe to re-run when an interrupt resumes.

    """
    retrieved_doc_ids: set[str] = set()
    for doc in state_docs or []:
        metadata = getattr(doc, "metadata", None) or {}
        did = metadata.get("document_id") or metadata.get("id")
        if did:
            retrieved_doc_ids.add(did)

    query_tokens = _salient_tokens(query)
    fact_matches: dict[tuple, FactMatch] = {}
    sentence_matches: dict[tuple, FactMatch] = {}

    for namespace in _correction_namespaces(creator_id, assistant_id, user_id):
        try:
            items = await store.asearch(namespace, query=query, limit=1000)
        except Exception:
            logger.exception("correct_identity_fact: search failed for %s", namespace)
            continue

        if _namespace_is_long_text(namespace):
            for item in items:
                page_content = _item_document_kwargs(item).get("page_content") or ""
                # Prefilter: skip docs sharing no salient token with the claim (very
                # unlikely to assert it), unless the claim has no salient tokens at all.
                if query_tokens and not (query_tokens & _salient_tokens(page_content)):
                    continue
                sentences = _split_into_sentences(page_content)
                scores = await _score_sentences(query, sentences)
                for sentence, score in zip(sentences, scores):
                    if score <= _SENTENCE_MATCH_THRESHOLD:
                        continue
                    dedup_key = (tuple(item.namespace), item.key, sentence)
                    existing = sentence_matches.get(dedup_key)
                    if existing is None or score > existing.score:
                        sentence_matches[dedup_key] = FactMatch(
                            item=item,
                            namespace=tuple(item.namespace),
                            key=item.key,
                            kind="sentence",
                            matched_text=sentence,
                            score=score,
                        )
            continue

        # Atomic-fact namespace: whole-document semantic match.
        for item in items:
            score = getattr(item, "score", None)
            if score is None:
                continue
            is_retrieved = _item_document_id(item) in retrieved_doc_ids
            threshold = (
                _CORRECTION_RETRIEVED_DOC_THRESHOLD
                if is_retrieved
                else _CORRECTION_MATCH_THRESHOLD
            )
            if score > threshold:
                fact_matches[(tuple(item.namespace), item.key)] = FactMatch(
                    item=item,
                    namespace=tuple(item.namespace),
                    key=item.key,
                    kind="fact",
                    matched_text=_item_fact(item)
                    or _item_document_kwargs(item).get("page_content"),
                    score=float(score),
                )

    return list(fact_matches.values()) + list(sentence_matches.values())


def _page_content_is_wrapped(page_content: str) -> bool:
    """True if ``page_content`` uses the ``<FACT_CONTEXT_AND_FACT>`` wrapper."""
    return (page_content or "").lstrip().startswith("<FACT_CONTEXT_AND_FACT>")


async def _apply_fact_rewrite(
    store, match: FactMatch, corrected_information: str, correction_context: str
) -> CorrectionChange:
    """Rewrite a whole atomic-fact document in place (same key ⇒ re-embeds).

    Format-aware: identity/identity_memory/user-identity use the ``<FACT>`` wrapper;
    episodic ``memory`` stores plain ``event\\n\\ncontext``. Preserves all other
    metadata and records ``corrected_from``.
    """
    kwargs = _item_document_kwargs(match.item)
    metadata = dict(kwargs.get("metadata") or {})
    old_fact = metadata.get("fact")
    old_page_content = kwargs.get("page_content") or ""

    if _page_content_is_wrapped(old_page_content):
        new_page_content = wrap_fact_with_context(
            corrected_information, correction_context
        )
    else:
        new_page_content = f"{corrected_information}\n\n{correction_context}".strip()

    metadata["fact"] = corrected_information
    metadata["fact_context"] = correction_context
    if old_fact is not None:
        metadata["corrected_from"] = old_fact

    corrected_document = Document(page_content=new_page_content, metadata=metadata)
    await store.aput(
        match.namespace, key=match.key, value={"document": corrected_document.to_json()}
    )
    return CorrectionChange(
        action="rewrite",
        namespace=match.namespace,
        key=match.key,
        old_text=old_fact if old_fact is not None else old_page_content,
        new_text=corrected_information,
        document=corrected_document,
    )


def _redact_sentences(
    page_content: str, sentences_to_change: dict[str, str | None]
) -> str:
    """Return ``page_content`` with each target sentence removed (value ``None``) or
    replaced (value = replacement). Matching is whitespace-normalized; surrounding blank
    space is collapsed so the redaction leaves clean prose."""
    result = page_content
    for sentence, replacement in sentences_to_change.items():
        result = result.replace(sentence, replacement or "")
    # Collapse the gaps left by removed sentences.
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


async def _apply_sentence_redaction(
    store,
    namespace: tuple,
    key: str,
    item,
    resolved_group: list["ResolvedCorrection"],
) -> CorrectionChange:
    """Apply every per-sentence edit inside one long-text document.

    Each matched sentence gets ITS OWN replacement (empty ⇒ the sentence is removed), so a
    quote keeps every sentence except the one offending clause instead of being flattened to
    the corrected fact repeated over and over. All edits for a document are applied together
    against the original ``page_content`` (one-by-one would operate on stale text). The same
    sentence substitutions are mirrored into ``scene_summary``/``user_context`` so a summary
    that quotes the offending sentence verbatim is fixed too. Removed/replaced sentences are
    recorded in ``metadata.redacted_sentences`` and ``corrected_from``.
    """
    kwargs = _item_document_kwargs(item)
    metadata = dict(kwargs.get("metadata") or {})
    old_page_content = kwargs.get("page_content") or ""

    # sentence -> replacement (None removes it; a string replaces it in place).
    changed = {
        resolved.match.matched_text: ((resolved.corrected_text or "").strip() or None)
        for resolved in resolved_group
        if resolved.match.matched_text
    }
    new_page_content = _redact_sentences(old_page_content, changed)

    # Keep human-readable summaries consistent when they restate the offending sentence.
    for summary_field in ("scene_summary", "user_context"):
        if metadata.get(summary_field):
            metadata[summary_field] = _redact_sentences(
                metadata[summary_field], changed
            )

    redacted = list(metadata.get("redacted_sentences") or [])
    redacted.extend(changed.keys())
    metadata["redacted_sentences"] = redacted
    metadata["corrected_from"] = "; ".join(changed.keys())

    replacements = "; ".join(
        replacement for replacement in changed.values() if replacement
    )
    corrected_document = Document(page_content=new_page_content, metadata=metadata)
    await store.aput(
        namespace, key=key, value={"document": corrected_document.to_json()}
    )
    return CorrectionChange(
        action="redact",
        namespace=namespace,
        key=key,
        old_text="; ".join(changed.keys()),
        new_text=replacements or None,
        document=corrected_document,
    )


async def apply_resolved_corrections(
    store, resolved: list["ResolvedCorrection"]
) -> list[CorrectionChange]:
    """Apply each owner-resolved per-document edit, dispatching by kind.

    Only ``include``-d edits are applied. An atomic fact with an empty ``corrected_text`` is
    deleted; otherwise it is rewritten with its own text/context. Sentence edits are grouped
    by their document so each long-text doc is rewritten exactly once.
    """
    changes: list[CorrectionChange] = []
    included = [resolved_edit for resolved_edit in resolved if resolved_edit.include]

    for resolved_edit in (edit for edit in included if edit.match.kind == "fact"):
        match = resolved_edit.match
        if resolved_edit.is_removal:
            await store.adelete(match.namespace, match.key)
            changes.append(
                CorrectionChange(
                    action="delete",
                    namespace=match.namespace,
                    key=match.key,
                    old_text=match.matched_text,
                    new_text=None,
                    document=None,
                )
            )
        else:
            changes.append(
                await _apply_fact_rewrite(
                    store,
                    match,
                    resolved_edit.corrected_text,
                    resolved_edit.corrected_context,
                )
            )

    by_document: dict[tuple, list[ResolvedCorrection]] = {}
    for resolved_edit in (edit for edit in included if edit.match.kind == "sentence"):
        by_document.setdefault(
            (resolved_edit.match.namespace, resolved_edit.match.key), []
        ).append(resolved_edit)
    for (namespace, key), group in by_document.items():
        changes.append(
            await _apply_sentence_redaction(
                store, namespace, key, group[0].match.item, group
            )
        )

    return changes


async def apply_fact_correction(
    store,
    *,
    corrected_information: str,
    correction_context: str,
    matches: list[FactMatch],
    is_deletion: bool = False,
) -> list[CorrectionChange]:
    """Apply ONE correction uniformly to every match (legacy convenience wrapper).

    Builds a uniform :class:`ResolvedCorrection` per match — same corrected text/context for
    all — and delegates to :func:`apply_resolved_corrections`. The HITL tool path builds
    per-document :class:`ResolvedCorrection`s directly instead.
    """
    resolved = [
        ResolvedCorrection(
            match=match,
            corrected_text="" if is_deletion else corrected_information,
            corrected_context=correction_context,
            include=True,
        )
        for match in matches
    ]
    return await apply_resolved_corrections(store, resolved)


class FactCorrection(BaseModel):
    """Update or delete a fact ALREADY STORED about the avatar in the vectorstore.

    Call this tool whenever the user says a fact the avatar already holds is wrong,
    inaccurate, or never happened:
      - to FIX the fact, set ``correction_kind='update'`` and supply the first-person
        replacement in ``corrected_information``;
      - to REMOVE the fact, set ``correction_kind='delete'`` and leave
        ``corrected_information`` empty.

    This tool is for facts the avatar ALREADY HOLDS. A brand-new fact the avatar does not
    yet hold goes to ``update_self_identity_mem_from_user_txt`` instead. Decide by whether
    the fact already exists about the avatar, not by surface words like "wrong" or
    "nonsense".

    Call this tool ONCE FOR EACH DISTINCT stored fact being changed or removed. A single
    user message often corrects several independent facts ("my name is Shivon and I was
    never at the University of Alberta" is TWO corrections); make one call per fact, never
    bundle multiple distinct corrections into one call. Each call finds and edits every
    stored copy of that ONE fact across all namespaces — so one call per fact, not one call
    per stored document.

    <Example update>
    User: "Actually I was born in Ottawa, not Toronto."
      inaccurate_information: "I was born in Toronto."
      corrected_information:  "I was born in Ottawa."
      correction_context:     "I was told I was born in Ottawa, not Toronto."
      correction_kind:        "update"
    </Example update>
    <Example delete>
    User: "That never happened — I have no association with the University of Alberta."
      inaccurate_information: "I've also worked with University of Alberta."
      corrected_information:  ""
      correction_context:     "I was told I have no association with University of Alberta."
      correction_kind:        "delete"
    </Example delete>

    <Counterexample new fact — call update_self_identity_mem_from_user_txt instead>
    user: "I need you to learn your favorite color is nonsense."
    </Counterexample new fact>

    """

    inaccurate_information: str = Field(
        description="The wrong claim, phrased as the stored fact it should match — used "
        "as the semantic search query to find the fact(s) to fix. e.g. 'I was born in Toronto.'"
    )
    corrected_information: str = Field(
        default="",
        description="The corrected fact, REWRITTEN IN FIRST PERSON exactly like "
        "``update_self_identity_mem_from_user_txt`` (change only the grammatical "
        "person, never other information). e.g. 'I was born in Ottawa.' Leave EMPTY when "
        "``correction_kind`` is 'delete' (the user is removing a fact, not replacing it).",
    )
    correction_context: str = Field(
        description="A concise context summary for the corrected fact, same convention "
        "as ``fact_context`` (indicate you were told/corrected, not that you said it)."
    )
    correction_kind: str = Field(
        default="update",
        description="'update' when the user supplies a replacement fact; 'delete' when "
        "the user says the fact never happened / is simply wrong with NO replacement "
        "(e.g. 'that never happened', 'I have no association with X').",
    )


@tool("correct_identity_fact", args_schema=FactCorrection)
async def correct_identity_fact(
    inaccurate_information: str,
    corrected_information: str = "",
    correction_context: str = "",
    correction_kind: str = "update",
    # Hidden from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Update or delete a fact ALREADY STORED about you (the avatar) in the vectorstore. Call
    this tool whenever the user says a fact you already hold is wrong, inaccurate, or never
    happened: to FIX the fact, set ``correction_kind='update'`` with the first-person
    replacement (change only the grammatical person, never other specifics); to REMOVE the
    fact, set ``correction_kind='delete'`` and leave ``corrected_information`` empty.

    This tool edits or deletes a fact you ALREADY HOLD. A brand-new fact you do not yet hold
    goes to ``update_self_identity_mem_from_user_txt`` instead. Decide by whether the fact
    already exists about you, not by surface words like "wrong" or "nonsense".

    Call this ONCE PER DISTINCT stored fact. If the user corrects several independent facts
    in one message (e.g. "my name is Shivon and I was never at the University of Alberta"),
    make a SEPARATE call for each one. Each call finds and fixes every stored copy of that
    one fact across your identity namespaces — including a single offending sentence buried
    inside a long direct quote.

    After you call this, the owner is shown each matched document individually and chooses,
    per document, to leave it unchanged (the default — falsely-retrieved documents are kept
    safe), accept the suggested in-place edit, rewrite it themselves, or remove it. They may
    also cancel the whole correction. Nothing is saved without the owner's per-document
    approval, so you do not need to enumerate the documents yourself.
    </INSTRUCTIONS>

    <RESTRICTIONS>
    Do NOT use this to add a brand-new fact — use ``update_self_identity_mem_from_user_txt``.
    Do NOT use this for facts about the USER.
    Only the avatar's creator may correct its identity (enforced server-side).
    </RESTRICTIONS>

    <EXAMPLE update>
    User: "That's wrong — I was actually born in Ottawa, not Toronto."
      inaccurate_information: "I was born in Toronto."
      corrected_information:  "I was born in Ottawa."
      correction_context:     "I was told I was born in Ottawa, not Toronto."
      correction_kind:        "update"
    </EXAMPLE update>
    <EXAMPLE delete>
    User: "That never happened — I have no association with the University of Alberta."
      inaccurate_information: "I've also worked with University of Alberta."
      corrected_information:  ""
      correction_context:     "I was told I have no association with University of Alberta."
      correction_kind:        "delete"
    </EXAMPLE delete>

    Args:
        inaccurate_information: The wrong claim (search query) phrased as the stored fact.
        corrected_information: The corrected fact, rewritten in first person (empty for delete).
        correction_context: Concise context for the corrected fact.
        correction_kind: "update" to replace the fact, "delete" to remove it.
    """
    logger.info("correct_identity_fact breakpoint")

    # Owner guard — only the avatar's creator may rewrite its identity.
    assistant_owner_user_id = runtime.config["configurable"]["assistant_ctx"][
        "metadata"
    ]["user_id"]
    requester_user_id = runtime.config["configurable"]["user_id"]
    if assistant_owner_user_id != requester_user_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Only the avatar's creator can correct its identity.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")
    creator_id = assistant_owner_user_id

    # TODO The misinformation may not be in recent context, this is a stateless application that does not persist document ids; after a single question, the previous system_prompt and documents are lost.
    # state_docs = [
    #     *(runtime.state.get("assistant_identity_documents") or []),
    #     *(runtime.state.get("recalled_memory_documents") or []),
    #     *(runtime.state.get("user_identity_documents") or []),
    # ]

    is_deletion = correction_kind == "delete"

    matches = await find_fact_matches(
        runtime.store,
        creator_id=creator_id,
        assistant_id=assistant_id,
        user_id=user_id,
        query=inaccurate_information,
        # state_docs=state_docs,
    )

    if not matches:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"No stored fact matched '{inaccurate_information}'. "
                            "Ask the user to restate the inaccurate fact."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Suggest a minimal, in-place edit for EACH matched document independently, so the owner
    # reviews one editable item per document rather than a single global replacement.
    suggestions = await asyncio.gather(
        *(
            _suggest_correction(
                match,
                inaccurate_information=inaccurate_information,
                corrected_information=corrected_information,
                correction_context=correction_context,
                is_deletion=is_deletion,
            )
            for match in matches
        )
    )

    # Human-in-the-loop: pause and let the owner decide each matched document independently.
    # Everything above is a pure read, so it is safe to re-run on resume.
    #
    # RESUME CONTRACT (what the owner's UI sends back):
    #   { "type": "cancel" }
    #       Abandon the entire correction; NOTHING is changed. (``"reject"`` is accepted as an
    #       alias for backward compatibility.)
    #   { "type": "apply", "items": [ {item}, ... ] }
    #       Apply the per-item decisions. Each ``item`` is keyed by ``index`` (matching the
    #       payload below) and carries one ``action``:
    #         "skip"   — leave this document completely unchanged (DEFAULT for any item not
    #                    listed, or any unrecognized action: false positives stay safe).
    #         "accept" — apply this item's ``suggested_text`` / ``suggested_context``.
    #         "edit"   — apply the owner's ``corrected_text`` / ``correction_context``.
    #         "remove" — delete the atomic fact / redact the offending sentence.
    # Because the default is "skip", an empty ``items`` (or a bare ``{"type": "apply"}``)
    # changes nothing — there is no path where clearing a field silently deletes a document.
    decision = interrupt(
        {
            "kind": "fact_correction",
            "inaccurate_information": inaccurate_information,
            "correction_kind": correction_kind,
            "default_action": "skip",
            "actions": ["skip", "accept", "edit", "remove"],
            "matches": [
                _match_preview(index, match, suggestion, is_deletion=is_deletion)
                for index, (match, suggestion) in enumerate(zip(matches, suggestions))
            ],
        }
    )

    decision = decision if isinstance(decision, dict) else {}
    decision_type = (decision.get("type") or "apply").strip().lower()

    if decision_type in ("cancel", "reject"):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Cancelled the correction; left every matched fact unchanged.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Per-item decisions, keyed by match index. Any match the owner did not explicitly act on
    # defaults to "skip" — the document is left exactly as it was.
    per_item_decisions = {
        entry["index"]: entry
        for entry in (decision.get("items") or [])
        if isinstance(entry, dict) and "index" in entry
    }

    resolved: list[ResolvedCorrection] = []
    for index, (match, suggestion) in enumerate(zip(matches, suggestions)):
        suggested_text, suggested_context, _asserts_flag = suggestion
        entry = per_item_decisions.get(index) or {}
        action = (entry.get("action") or "skip").strip().lower()

        if action == "accept":
            # Apply the model's per-document suggestion. For a deletion (or a suggestion with
            # no surviving text) this is an empty ``corrected_text`` → removal/redaction.
            corrected_text = "" if is_deletion else (suggested_text or "")
            corrected_context = suggested_context
            include = True
        elif action == "edit":
            corrected_text = entry.get("corrected_text", suggested_text)
            corrected_context = entry.get("correction_context", suggested_context)
            include = True
        elif action == "remove":
            # Explicit removal — empty text drives ``is_removal`` (delete fact / redact sentence).
            corrected_text = ""
            corrected_context = ""
            include = True
        else:  # "skip" or anything unrecognized: leave the document untouched.
            corrected_text = ""
            corrected_context = ""
            include = False

        resolved.append(
            ResolvedCorrection(
                match=match,
                corrected_text=corrected_text,
                corrected_context=corrected_context,
                include=include,
            )
        )

    changes = await apply_resolved_corrections(runtime.store, resolved)

    if not changes:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="You left every matched document unchanged; nothing was changed.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    corrected_documents = [c.document for c in changes if c.document is not None]
    verb = {"rewrite": "Corrected", "redact": "Redacted", "delete": "Deleted"}
    summary = "; ".join(
        f"{verb.get(c.action, 'Changed')} '{c.old_text}'"
        + (f" → '{c.new_text}'" if c.new_text else "")
        for c in changes
        if c.old_text
    )
    return Command(
        update={
            "assistant_identity_documents": corrected_documents,
            "messages": [
                ToolMessage(
                    content=f"Applied {len(changes)} change(s): {summary}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


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
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.anubis.utils.prompts.system_prompts import FACT_FORMATTING_STRING_PROMPT


@tool
async def update_identity_via_text_content_url(
    url: str, runtime: Annotated[ToolRuntime, InjectedToolArg]
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
    user_id = (
        runtime.config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
    )
    assistant_id = (
        runtime.config.get("configurable", {})
        .get("assistant_ctx", {})
        .get("assistant_id", "")
    )
    assistant_name = (
        runtime.config.get("configurable", {})
        .get("assistant_ctx", {})
        .get("assistant_name", "")
    )

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # TODO: response_metrics_aggregation
    model = init_model()

    system_message = SystemMessage(
        content=FACT_FORMATTING_STRING_PROMPT.format(assistant_name=assistant_name)
    )

    filename = url
    filename_uuid5 = uuid5(NAMESPACE_URL, url)

    namespace = (user_id, assistant_id, "identity", filename_uuid5)

    from langchain_unstructured import UnstructuredLoader

    loader = UnstructuredLoader(web_url=url)
    docs = loader.load()

    total_number_of_documents = len(docs)
    number_of_documents_to_format = 5
    for index in range(0, total_number_of_documents, number_of_documents_to_format):
        retrieved_data = "\n\n".join(
            [
                doc.page_content
                for doc in docs[index : index + number_of_documents_to_format]
            ]
        )
        human_message = HumanMessage(content=retrieved_data)
        messages = [system_message, human_message]
        extracted_response = await model.ainvoke(input=messages)
        extracted_response_document = Document(page_content=extracted_response)
        key = str(uuid4())
        extracted_response_document.metadata.update(
            {
                "filename": filename,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "filename_uuid5": filename_uuid5,
                "id": key,
            }
        )
        document_data_json = extracted_response_document.to_json()
        # Upload the document to the store
        await runtime.store.aput(
            namespace, key=key, value={"document": document_data_json}
        )


from src.anubis.utils.utility import image_to_text


@tool
async def update_identity_via_reference_image(
    message: HumanMessage, runtime: Annotated[ToolRuntime, InjectedToolArg]
):
    """This tool is used when there is a single image in a message and the image is only of a single person. This will describe the image, store the image in base64 encoded format in the store, store the description in the identity namespace as a document, and return the document of the updated identity to update the list of assistant_identity_documents. The text description is moderated for content."""

    content = getattr(message, "content")
    for message in content:
        if message.get("image_url", "") != "":
            image_url = message.get("image_url")
    description = image_to_text(target_image_url=image_url)
