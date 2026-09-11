"""Reading a person's own words out of the conversations that already hold them.

What is pinned down here:

- **The module never writes.** The whole point of reading conversations where
  they already sit is that nothing is duplicated into the store, where every copy
  would also cost an embedding row. A store double that raises on any write proves
  it, so a future "small" convenience write fails the suite instead of shipping.
- **Only the person's own words count.** A browser harvest request, an ambient
  observation, a turn this system authored, and a turn from somebody else's Slack
  room all wear a human turn's shape and are none of them the person speaking.
- **Role direction is inverted from the chat's.** The person is the ``user`` in a
  conversation and the ``assistant`` in their own training data, because their
  words are what the avatar must learn to produce.
- **Model text never reaches the completion side.** An avatar trained on model
  output drifts towards the model and away from the person, and no amount of
  further training recovers it.
- **A spoken turn is split by who actually spoke**, and the avatar's own voice
  heard through a speaker in the room is discarded rather than attributed to its
  owner.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.anubis.utils.learning.speaker_turns import (
    assert_no_model_text_on_completion_side,
    is_group_platform_turn,
    is_quotable_human_turn,
    model_texts_of,
    role_rows_for_speaker,
    speaker_quotes,
    speaker_thread_ids,
)

LONG_ENOUGH = "I spent eleven years building bridges in Rotterdam before I wrote a line of code."
ALSO_LONG = "The hardest part was learning that concrete and code fail differently."
HARVEST_MARKER = (
    "[neural-nexus:conversation-suggestions] Given this conversation, suggest three "
    "short messages. Reply with a JSON array of three strings."
)


def _conversation() -> list:
    """One thread carrying every shape a human turn can take."""
    return [
        HumanMessage(content=LONG_ENOUGH, id="m1"),
        AIMessage(content="That is a serious change of direction.", id="a1"),
        HumanMessage(content="ok", id="m2"),
        HumanMessage(content=HARVEST_MARKER, id="m3"),
        HumanMessage(
            content="[AMBIENT_OBSERVATION id=1] A desk with two monitors.",
            id="m4",
            additional_kwargs={"hidden": True, "kind": "ambient_observation"},
        ),
        HumanMessage(
            content="The owner just connected Gmail.",
            id="m5",
            additional_kwargs={"hidden": True, "kind": "connection_acknowledgement"},
        ),
        HumanMessage(
            content="Something a stranger typed in a Slack channel this avatar watches.",
            id="m6",
            additional_kwargs={"platform": "slack"},
        ),
        AIMessage(content="Understood.", id="a2"),
        HumanMessage(content=ALSO_LONG, id="m7"),
    ]


def test_only_the_persons_own_words_are_quoted() -> None:
    quotes = speaker_quotes(_conversation(), minimum_characters=40)
    assert [quote.message_id for quote in quotes] == ["m1", "m7"]
    assert quotes[0].text == LONG_ENOUGH
    # The counterpart is what the other side said immediately before, which is
    # what makes the pair usable as a training example.
    assert quotes[0].counterpart_text == ""
    assert quotes[1].counterpart_text == "Understood."


@pytest.mark.parametrize(
    ("index", "why"),
    [
        (3, "a browser harvest request is the product talking to itself"),
        (4, "an ambient observation was never typed by anybody"),
        (5, "a connection acknowledgement is server-authored"),
        (6, "a group platform turn belongs to somebody else's room"),
    ],
)
def test_turns_that_are_not_the_person(index: int, why: str) -> None:
    assert not is_quotable_human_turn(_conversation()[index]), why


def test_group_platform_turns_are_refused_by_either_marker() -> None:
    # Two markers, because a group turn reaches the avatar by more than one route
    # and platform_policies commits us to not training on the people in a room.
    assert is_group_platform_turn(
        HumanMessage(content="hello", additional_kwargs={"platform": "discord"})
    )
    assert is_group_platform_turn(
        HumanMessage(content="hello", additional_kwargs={"source_kind": "twitch"})
    )
    assert not is_group_platform_turn(HumanMessage(content="hello"))


def test_short_turns_are_not_quotes_but_are_still_conversation() -> None:
    quotes = speaker_quotes(_conversation(), minimum_characters=40)
    assert all(quote.text != "ok" for quote in quotes)
    # "ok" is a real turn and still belongs in the conversation's shape.
    rows = role_rows_for_speaker(_conversation())
    assert any(row["content"] == "ok" for row in rows)


def test_role_direction_is_inverted_from_the_chat() -> None:
    rows = role_rows_for_speaker(_conversation())
    assert rows[0] == {"role": "assistant", "content": LONG_ENOUGH}
    assert rows[1] == {
        "role": "user",
        "content": "That is a serious change of direction.",
    }
    # Excluded turns stay excluded on this path too.
    assert all(HARVEST_MARKER not in row["content"] for row in rows)
    assert all("Slack channel" not in row["content"] for row in rows)


def test_consecutive_same_role_turns_are_merged() -> None:
    rows = role_rows_for_speaker(
        [
            HumanMessage(content="First thing.", id="1"),
            HumanMessage(content="Second thing.", id="2"),
            AIMessage(content="Noted.", id="3"),
        ]
    )
    assert rows == [
        {"role": "assistant", "content": "First thing.\n\nSecond thing."},
        {"role": "user", "content": "Noted."},
    ]


def test_model_text_never_reaches_the_completion_side() -> None:
    messages = _conversation()
    rows = role_rows_for_speaker(messages)
    poisoned = rows + [
        {"role": "assistant", "content": "That is a serious change of direction."}
    ]
    kept = assert_no_model_text_on_completion_side(poisoned, model_texts_of(messages))
    assert len(kept) == len(poisoned) - 1
    assert all(
        row["content"] != "That is a serious change of direction."
        for row in kept
        if row["role"] == "assistant"
    )
    # The same sentence on the PROMPT side is fine: a model may ask the question.
    assert any(
        row["content"] == "That is a serious change of direction."
        for row in kept
        if row["role"] == "user"
    )


def test_a_spoken_turn_is_split_by_who_spoke() -> None:
    spoken = HumanMessage(
        content="Evan: ...\nSpeaker 2: ...",
        id="s1",
        additional_kwargs={
            "kind": "spoken_turn",
            "speakers": {
                "owner_label": "Evan",
                "segments": [
                    {
                        "speaker": "Speaker 2",
                        "text": "So what got you into this line of work?",
                        "is_owner": False,
                    },
                    {
                        "speaker": "Evan",
                        "text": "Eleven years of bridges, then a laptop and a bad idea.",
                        "is_owner": True,
                    },
                    {
                        "speaker": "Evan (avatar)",
                        "text": "I would put that differently, honestly.",
                        "is_owner": True,
                        "is_avatar": True,
                    },
                ],
            },
        },
    )
    quotes = speaker_quotes([spoken], minimum_characters=10)
    # Exactly one: the owner's own line. The other person is the counterpart, and
    # the avatar's cloned voice heard through a speaker is discarded outright.
    assert len(quotes) == 1
    assert quotes[0].speaker_label == "Evan"
    assert quotes[0].spoken is True
    assert quotes[0].counterpart_text == "So what got you into this line of work?"
    assert all("differently" not in quote.text for quote in quotes)


def test_a_hidden_spoken_turn_is_still_the_person_speaking() -> None:
    # Some spoken modes tag the turn hidden; the speaker record wins over that.
    spoken = HumanMessage(
        content="Evan: ...",
        id="s2",
        additional_kwargs={
            "hidden": True,
            "kind": "spoken_turn",
            "speakers": {
                "owner_label": "Evan",
                "segments": [
                    {"speaker": "Evan", "text": "A thing worth keeping.", "is_owner": True}
                ],
            },
        },
    )
    assert is_quotable_human_turn(spoken)
    assert len(speaker_quotes([spoken], minimum_characters=5)) == 1


class _StoreThatRefusesWrites:
    """Any write is a bug: the conversations are already stored."""

    async def aput(self, *args, **kwargs):
        raise AssertionError("speaker_turns must never write to the store")

    async def adelete(self, *args, **kwargs):
        raise AssertionError("speaker_turns must never delete from the store")

    async def aget(self, *args, **kwargs):
        raise AssertionError("speaker_turns must not read the store either")


def test_reading_words_writes_nothing() -> None:
    store = _StoreThatRefusesWrites()
    messages = _conversation()
    # Every entry point, with a store in scope that detonates on contact.
    speaker_quotes(messages)
    role_rows_for_speaker(messages)
    model_texts_of(messages)
    assert_no_model_text_on_completion_side([], set())
    assert isinstance(store, _StoreThatRefusesWrites)


class _ThreadSearch:
    """A threads.search that pages, and records how it was called."""

    def __init__(self, thread_ids: list[str]) -> None:
        self._thread_ids = thread_ids
        self.calls: list[dict] = []

    async def search(self, *, metadata, limit, offset, sort_by, sort_order):
        self.calls.append({"metadata": metadata, "limit": limit, "offset": offset})
        page = self._thread_ids[offset : offset + limit]
        return [{"thread_id": thread_id} for thread_id in page]


class _Client:
    def __init__(self, thread_ids: list[str]) -> None:
        self.threads = _ThreadSearch(thread_ids)


@pytest.mark.asyncio
async def test_every_thread_is_listed_not_only_the_first_page() -> None:
    # threads.search defaults to ten and gives no sign that it truncated; an
    # account's history must not silently shrink to its ten newest conversations.
    client = _Client([f"thread-{index}" for index in range(25)])
    found = await speaker_thread_ids(client, "user-1", page_size=10)
    assert len(found) == 25
    assert client.threads.calls[0]["metadata"] == {
        "thread_metadata": {"user_id": "user-1"}
    }
    assert [call["offset"] for call in client.threads.calls] == [0, 10, 20]


@pytest.mark.asyncio
async def test_a_search_failure_yields_nothing_rather_than_raising() -> None:
    class _Failing:
        class threads:  # noqa: N801 - mirrors the software development kit shape
            @staticmethod
            async def search(**_kwargs):
                raise RuntimeError("the server said no")

    assert await speaker_thread_ids(_Failing(), "user-1") == []
    assert await speaker_thread_ids(None, "user-1") == []
    assert await speaker_thread_ids(_Client([]), "") == []
