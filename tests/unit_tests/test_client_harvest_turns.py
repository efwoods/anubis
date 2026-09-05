"""Browser harvest turns stay out of the model's input and the transcript.

A thread that stored a ``[neural-nexus:conversation-suggestions]`` request
from an earlier browser build, plus the JSON lists the avatar answered with,
must not teach the avatar to keep answering in JSON.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.anubis.utils.client_harvest_turns import (
    is_client_harvest_turn,
    is_suggestion_list_reply,
    without_stale_client_harvest_turns,
)

HARVEST = HumanMessage(
    content="[neural-nexus:conversation-suggestions] Given this conversation, "
    "suggest three short messages. Reply with a JSON array of three strings.",
    id="harvest",
)
JSON_LIST = AIMessage(
    content='["Hello there. I’m here with you.", "Hey. How are you?", "Hi."]',
    id="json-reply",
)


def test_marker_and_json_list_detection() -> None:
    assert is_client_harvest_turn(HARVEST)
    assert is_client_harvest_turn({"type": "human", "content": HARVEST.content})
    assert not is_client_harvest_turn(HumanMessage(content="hello there"))
    assert not is_client_harvest_turn(AIMessage(content=HARVEST.content))
    assert is_suggestion_list_reply(JSON_LIST)
    assert is_suggestion_list_reply({"type": "ai", "content": JSON_LIST.content})
    assert not is_suggestion_list_reply(AIMessage(content="I love you, buddy."))
    assert not is_suggestion_list_reply(AIMessage(content="[smiles] I missed you"))
    assert not is_suggestion_list_reply(HumanMessage(content=JSON_LIST.content))


def test_past_harvest_and_its_reply_and_stray_json_lists_are_dropped() -> None:
    hello = HumanMessage(content="hello there", id="hello")
    spoken = AIMessage(content="I love you, buddy.", id="spoken")
    later = HumanMessage(content="please say something", id="later")
    kept = without_stale_client_harvest_turns(
        [HARVEST, JSON_LIST, hello, JSON_LIST, spoken, later]
    )
    assert [message.id for message in kept] == ["hello", "spoken", "later"]


def test_reply_after_a_harvest_is_dropped_even_when_not_a_json_list() -> None:
    description = AIMessage(content="I am a calm and steady friend.", id="description")
    hello = HumanMessage(content="hi", id="hi")
    kept = without_stale_client_harvest_turns(
        [
            HumanMessage(content="[neural-nexus:generate-description] Describe yourself."),
            description,
            hello,
        ]
    )
    assert [message.id for message in kept] == ["hi"]


def test_a_live_harvest_request_at_the_end_is_kept_for_answering() -> None:
    hello = HumanMessage(content="hi", id="hi")
    reply = AIMessage(content="Hey.", id="reply")
    kept = without_stale_client_harvest_turns([HARVEST, JSON_LIST, hello, reply, HARVEST])
    assert [message.id for message in kept] == ["hi", "reply", "harvest"]


def test_clean_threads_pass_through_unchanged() -> None:
    thread = [HumanMessage(content="hi", id="a"), AIMessage(content="Hey.", id="b")]
    assert without_stale_client_harvest_turns(thread) == thread
    assert without_stale_client_harvest_turns([]) == []
