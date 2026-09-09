"""Automatic naming of a conversation thread.

A conversation is named by the messaging service rather than by the browser.
The browser has no model to call and no view of the whole transcript, so
before this module every thread reached the sidebar with no
``conversation_title`` at all and was listed by its creation timestamp.

Two moments produce a name, both driven by ``src/api/webapp.py``:

* **A new conversation is started** — the first turn on a fresh thread names
  the conversation from that opening exchange, so the sidebar entry is
  recognizable the moment the conversation appears in the list.
* **A conversation is left** — the browser tells the messaging service that
  the reader has switched to another conversation or closed the page, and the
  name is regenerated from the whole transcript. An opening exchange about one
  subject that turned into a conversation about another subject is renamed to
  the subject the conversation actually covered.

A name the reader typed is never overwritten. Every stored name carries
``conversation_title_source``: ``"automatic"`` for a name this module wrote,
``"user"`` for a name the reader typed in the sidebar. Only an automatic name
is regenerated.

The name is produced by a structured-output call, which
``src.anubis.utils.model.init_model`` routes to the classification model, not
to the avatar's inference model — naming a conversation is cheap classification
work and must not spend the inference budget.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

AUTOMATIC_TITLE_SOURCE = "automatic"
MANUAL_TITLE_SOURCE = "user"

# The key the source is stored under, beside ``conversation_title`` in the
# thread's nested ``thread_metadata`` object.
CONVERSATION_TITLE_SOURCE_KEY = "conversation_title_source"


class ConversationTitle(BaseModel):
    """The short name a reader would recognize a conversation by."""

    conversation_title: str = Field(
        description=(
            "A title of two to six words naming what this conversation is about. "
            "Write the title in the language the conversation is written in. "
            "Use sentence capitalization. Do not end the title with a period, and "
            "do not wrap the title in quotation marks."
        )
    )


CONVERSATION_TITLE_SYSTEM_PROMPT = """
<ROLE>
You name a conversation the way a person scanning a list of past conversations would want that conversation named.
</ROLE>

<INSTRUCTIONS>
Read the TRANSCRIPT, which is a conversation between a USER and an AVATAR.
Write a title of two to six words that names the subject the USER and the AVATAR actually discussed.
Name the specific subject. A title such as "Conversation" or "Chat with the avatar" names nothing and is wrong.
Prefer the concrete nouns the USER used over a general category: "Broken deployment pipeline" is a better title than "Technical question".
Write the title in the language the TRANSCRIPT is written in.
Use sentence capitalization: capitalize the first word and any proper noun, and leave the remaining words lowercase.
Do not end the title with a period. Do not wrap the title in quotation marks. Do not mention the words "user", "avatar", "conversation", or "transcript" in the title.
If the TRANSCRIPT is too short or too empty to name a subject, title the conversation after whatever the USER opened with.
</INSTRUCTIONS>

<OUTPUT>
Return the title in the `conversation_title` field. Return nothing else.
</OUTPUT>
"""


def _thread_metadata_of(thread: Any) -> dict[str, Any]:
    """Return the nested ``thread_metadata`` object of a thread record, or an empty one."""
    if not isinstance(thread, dict):
        return {}
    metadata = thread.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    thread_metadata = metadata.get("thread_metadata")
    return thread_metadata if isinstance(thread_metadata, dict) else {}


def stored_conversation_title(thread_metadata: dict[str, Any], thread_id: str) -> str:
    """Return the name already stored on a thread, or an empty string when there is none.

    A thread whose stored name is the thread identifier itself counts as
    unnamed: earlier builds of the messaging service wrote the identifier into
    ``conversation_title`` when the browser sent an empty name, and showing a
    reader an opaque identifier is the same as showing that reader nothing.
    """
    title = thread_metadata.get("conversation_title")
    if not isinstance(title, str):
        return ""
    title = title.strip()
    if not title or title == thread_id:
        return ""
    return title


def conversation_title_may_be_replaced(
    thread_metadata: dict[str, Any], thread_id: str
) -> bool:
    """Whether this module is allowed to write a new name onto the thread.

    A thread with no usable name may always be named. A thread that already
    carries a name may be renamed only when that name was written by this
    module; a name the reader typed in the sidebar is the reader's, and a
    regeneration that overwrote the reader's name would look to the reader like
    the sidebar losing the reader's own edit.
    """
    if not stored_conversation_title(thread_metadata, thread_id):
        return True
    return thread_metadata.get(CONVERSATION_TITLE_SOURCE_KEY) != MANUAL_TITLE_SOURCE


def visible_conversation_messages(messages: list[Any]) -> list[Any]:
    """Keep the messages a reader can see, which are the messages the name is drawn from.

    Ambient observations of the room, browser harvest turns, and the avatar's
    replies to those turns are hidden from the transcript. Naming a
    conversation after a scene the camera noticed would title a conversation
    the reader never had.
    """
    from src.anubis.utils.ambient.observations import is_hidden_message

    return [message for message in (messages or []) if not is_hidden_message(message)]


async def invoke_structured(
    response_format: type[BaseModel], system_prompt: str, human_text: str
) -> Any:
    """Run one structured-output model call. Isolated so tests can replace this call."""
    from src.anubis.utils.model import init_model

    model = init_model(response_format=response_format)
    return await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    )


def _title_from_response(response: Any) -> str:
    """Pull the title string out of whatever shape the structured call returned."""
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, BaseModel):
        response = response.model_dump()
    if isinstance(response, dict):
        title = response.get("conversation_title")
        return title.strip() if isinstance(title, str) else ""
    return ""


def _tidy_title(title: str, maximum_characters: int) -> str:
    """Strip the punctuation a model adds around a title and cap the length."""
    title = " ".join(title.split())
    title = title.strip().strip('"').strip("'").strip()
    while title.endswith((".", ",", ";", ":", "!", "?")):
        title = title[:-1].rstrip()
    if maximum_characters > 0 and len(title) > maximum_characters:
        # Cut on a word boundary when there is one, so the reader sees whole
        # words followed by an ellipsis rather than a word torn in half.
        cut = title[:maximum_characters].rstrip()
        last_space = cut.rfind(" ")
        if last_space > maximum_characters // 2:
            cut = cut[:last_space].rstrip()
        title = cut + "…"
    return title


async def generate_conversation_title(
    messages: list[Any], context: GlobalContext | None = None
) -> str:
    """Return a short name for the conversation in ``messages``, or an empty string.

    An empty string is returned rather than an exception raised for every
    failure — an unreachable classification model, a transcript with nothing in
    it, a model that answered with an empty title. Naming a conversation is a
    convenience for the sidebar, and no failure here may cost the reader the
    reply that was just streamed.
    """
    context = context or GlobalContext()
    from src.anubis.utils.learning.sentiment import render_transcript

    tail_messages = int(
        getattr(context, "conversation_title_transcript_tail_messages", 20) or 20
    )
    transcript = render_transcript(
        visible_conversation_messages(messages), tail=tail_messages
    )
    if not transcript.strip():
        return ""
    timeout_seconds = float(
        getattr(context, "conversation_title_timeout_seconds", 20.0) or 20.0
    )
    try:
        response = await asyncio.wait_for(
            invoke_structured(
                ConversationTitle,
                CONVERSATION_TITLE_SYSTEM_PROMPT,
                "<TRANSCRIPT>\n" + transcript + "\n</TRANSCRIPT>",
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "Naming the conversation timed out after %.1f seconds", timeout_seconds
        )
        return ""
    except Exception as title_error:  # noqa: BLE001 - a name is a convenience
        logger.warning("Naming the conversation failed: %s", title_error)
        return ""
    maximum_characters = int(
        getattr(context, "conversation_title_max_characters", 60) or 60
    )
    return _tidy_title(_title_from_response(response), maximum_characters)


def conversation_naming_enabled(context: GlobalContext | None = None) -> bool:
    """Whether the messaging service names conversations at all."""
    context = context or GlobalContext()
    setting = getattr(context, "conversation_title_enabled", "TRUE")
    return str(setting).strip().upper() in ("TRUE", "1", "YES", "ON")


async def name_conversation_thread(
    thread_id: str,
    *,
    langgraph_client_headers: dict,
    only_when_unnamed: bool = False,
    user_id: str | None = None,
    assistant_id: str | None = None,
    context: GlobalContext | None = None,
) -> str:
    """Name ``thread_id`` from its transcript and store the name on the thread.

    Returns the name that was written, or an empty string when the thread was
    left alone: naming is switched off, the reader has already typed a name, the
    transcript is empty, or the classification model could not be reached. When
    ``user_id`` and ``assistant_id`` are given, the thread is verified to belong
    to that reader and that avatar before anything is written.

    ``only_when_unnamed`` is what keeps a message turn from paying for a naming
    call it does not need. Every turn of a conversation runs through the
    messaging service, but only the first turn of a new conversation has a
    conversation to name; with ``only_when_unnamed`` set, a conversation that
    already carries a name is left exactly as it is. The route the browser calls
    when the reader leaves a conversation clears the flag, because a
    conversation that has moved on from the subject of its opening exchange is
    renamed after the subject the whole transcript covers.

    The thread's existing ``thread_metadata`` is read and merged rather than
    replaced. The platform stores that nested object whole, so writing a fresh
    object would drop the flags that live beside the name — ``pinned``,
    ``shared``, ``most_recent_message`` — and a named conversation would quietly
    lose the reader's pin.
    """
    context = context or GlobalContext()
    if not conversation_naming_enabled(context):
        return ""
    if not thread_id:
        return ""

    from langgraph_sdk import get_client

    langgraph_client = get_client(headers=langgraph_client_headers)
    try:
        thread = await langgraph_client.threads.get(thread_id=thread_id)
    except Exception as thread_error:  # noqa: BLE001 - a name is a convenience
        logger.warning(
            "Could not read thread %s to name it: %s", thread_id, thread_error
        )
        return ""

    thread_metadata = _thread_metadata_of(thread)
    owning_user_id = thread_metadata.get("user_id")
    owning_assistant_id = thread_metadata.get("assistant_id")
    if user_id is not None and owning_user_id is not None and owning_user_id != user_id:
        return ""
    if (
        assistant_id is not None
        and owning_assistant_id is not None
        and owning_assistant_id != assistant_id
    ):
        return ""
    if not conversation_title_may_be_replaced(thread_metadata, thread_id):
        return ""
    if only_when_unnamed and stored_conversation_title(thread_metadata, thread_id):
        return ""

    try:
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
        messages = state.get("values", {}).get("messages", []) if state else []
    except Exception as state_error:  # noqa: BLE001 - a name is a convenience
        logger.warning(
            "Could not read the transcript of thread %s to name it: %s",
            thread_id,
            state_error,
        )
        return ""

    title = await generate_conversation_title(messages, context)
    if not title:
        return ""

    existing_metadata = thread.get("metadata") if isinstance(thread, dict) else None
    existing_metadata = existing_metadata if isinstance(existing_metadata, dict) else {}
    next_metadata = {
        **existing_metadata,
        "thread_metadata": {
            **thread_metadata,
            "conversation_title": title,
            CONVERSATION_TITLE_SOURCE_KEY: AUTOMATIC_TITLE_SOURCE,
        },
    }
    try:
        await langgraph_client.threads.update(
            thread_id=thread_id, metadata=next_metadata
        )
    except Exception as update_error:  # noqa: BLE001 - a name is a convenience
        logger.warning(
            "Could not store the name of thread %s: %s", thread_id, update_error
        )
        return ""
    logger.info("Named conversation %s %r", thread_id, title)
    return title
