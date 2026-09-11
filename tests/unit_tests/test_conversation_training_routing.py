"""A person's words build that person's own avatar, whoever they were talking to.

What somebody types to Neural Nexus is a direct quote of that person. So when a
visitor holds a conversation with somebody ELSE's public avatar, those words are
the visitor's — they belong in the visitor's own avatar's corpus, and nothing of
them belongs to the avatar that was addressed. What is pinned down:

- the corpus is routed by **who spoke**, not by which avatar was spoken to;
- the words are **read from the threads that already hold them** and handed on in
  memory, so the store gains no copy of any conversation;
- a person whose personal-avatar pointer has not been written yet is skipped
  rather than guessed at;
- a failure anywhere in this is worth zero of the learning records the sweep has
  already written.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.anubis.utils.learning.bulk_learning as bulk_learning
from src.anubis.utils.learning.namespaces import engagement_namespace
from src.anubis.utils.learning.speaker_turns import (
    conversation_quote_documents,
    speaker_quotes,
    speaker_thread_ids_from_engagement,
)
from src.anubis.utils.personal_avatar import personal_avatar_pointer_namespace

VISITOR = "visitor-account"
VISITORS_OWN_AVATAR = "visitor-personal-avatar"
SOMEBODY_ELSES_AVATAR = "a-public-avatar-of-somebody-else"

FIRST = "I spent eleven years building bridges in Rotterdam before I wrote any code."
SECOND = "The hardest part was learning that concrete and code fail differently."


class _Store:
    """An in-memory store that records every write, so copies are visible."""

    def __init__(self) -> None:
        self.rows: dict = {}
        self.writes: list[tuple] = []

    async def aput(self, namespace, key, value):
        self.writes.append((tuple(namespace), key))
        self.rows[(tuple(namespace), key)] = value

    async def aget(self, namespace, key):
        namespace = tuple(namespace)
        if (namespace, key) not in self.rows:
            return None
        return type("Item", (), {"value": self.rows[(namespace, key)], "key": key})()

    async def alist_namespaces(self, prefix=None, limit=1000):
        return [
            namespace
            for (namespace, _key) in self.rows
            if prefix is None or namespace[: len(prefix)] == tuple(prefix)
        ]

    async def asearch(self, namespace, query="*", limit=100):
        return []


class _Graph:
    """A graph whose checkpoints already hold the conversations."""

    def __init__(self, threads: dict[str, list]) -> None:
        self.threads = threads

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return type(
            "Snapshot", (), {"values": {"messages": self.threads.get(thread_id, [])}}
        )()


def _conversation_with_somebody_elses_avatar() -> list:
    return [
        HumanMessage(content=FIRST, id="m1"),
        AIMessage(content="What made you change direction?", id="a1"),
        HumanMessage(content=SECOND, id="m2"),
    ]


async def _seed(store: _Store, *, with_pointer: bool = True) -> None:
    # The visitor's engagement with somebody else's avatar — written by the
    # ordinary per-turn path, and already naming the thread.
    await store.aput(
        engagement_namespace(VISITOR, SOMEBODY_ELSES_AVATAR),
        "engagement",
        {"value": {"message_count": 2, "conversation_thread_ids": ["thread-1"]}},
    )
    if with_pointer:
        await store.aput(
            personal_avatar_pointer_namespace(VISITOR),
            "personal_avatar",
            {"value": {"assistant_id": VISITORS_OWN_AVATAR}},
        )


@pytest.mark.asyncio
async def test_threads_are_found_through_the_index_that_already_exists() -> None:
    store = _Store()
    await _seed(store)
    assert await speaker_thread_ids_from_engagement(store, VISITOR) == ["thread-1"]
    assert await speaker_thread_ids_from_engagement(store, "nobody") == []


@pytest.mark.asyncio
async def test_the_words_build_the_speakers_own_avatar(monkeypatch) -> None:
    store = _Store()
    await _seed(store)
    graph = _Graph({"thread-1": _conversation_with_somebody_elses_avatar()})
    calls: list[dict] = []

    async def _fake_calibrate(*, store, assistant_id, documents, user_id):
        calls.append(
            {"assistant_id": assistant_id, "user_id": user_id, "documents": documents}
        )

    import src.subgraphs.process_media_graph.utils.calibrate_ground_truth as module

    monkeypatch.setattr(module, "calibrate_ground_truth", _fake_calibrate)

    writes_before = len(store.writes)
    assert await bulk_learning.recalibrate_style_from_conversations(
        store, graph, VISITOR
    ) == 1

    (call,) = calls
    # Routed by who spoke, not by which avatar was spoken to.
    assert call["assistant_id"] == VISITORS_OWN_AVATAR
    assert call["assistant_id"] != SOMEBODY_ELSES_AVATAR
    assert call["user_id"] == VISITOR
    assert [document.page_content for document in call["documents"]] == [FIRST, SECOND]
    # The avatar's own question is carried as the prompt the answer answered.
    assert call["documents"][1].metadata["adapter_prompt"] == (
        "What made you change direction?"
    )
    # And the conversation itself was not copied anywhere.
    assert len(store.writes) == writes_before


@pytest.mark.asyncio
async def test_a_person_with_no_pointer_yet_is_skipped_not_guessed_at(
    monkeypatch,
) -> None:
    store = _Store()
    await _seed(store, with_pointer=False)
    graph = _Graph({"thread-1": _conversation_with_somebody_elses_avatar()})

    import src.subgraphs.process_media_graph.utils.calibrate_ground_truth as module

    def _explode(**_kwargs):
        raise AssertionError("must not calibrate without knowing whose avatar it is")

    monkeypatch.setattr(module, "calibrate_ground_truth", _explode)
    assert await bulk_learning.recalibrate_style_from_conversations(
        store, graph, VISITOR
    ) == 0


@pytest.mark.asyncio
async def test_a_calibration_failure_never_costs_the_sweep(monkeypatch) -> None:
    store = _Store()
    await _seed(store)
    graph = _Graph({"thread-1": _conversation_with_somebody_elses_avatar()})

    async def _failing(**_kwargs):
        raise RuntimeError("the corpus could not be read")

    import src.subgraphs.process_media_graph.utils.calibrate_ground_truth as module

    monkeypatch.setattr(module, "calibrate_ground_truth", _failing)
    assert await bulk_learning.recalibrate_style_from_conversations(
        store, graph, VISITOR
    ) == 0


@pytest.mark.asyncio
async def test_a_conversation_with_nothing_quotable_calibrates_nothing(
    monkeypatch,
) -> None:
    store = _Store()
    await _seed(store)
    graph = _Graph({"thread-1": [HumanMessage(content="ok", id="m1")]})

    import src.subgraphs.process_media_graph.utils.calibrate_ground_truth as module

    monkeypatch.setattr(
        module,
        "calibrate_ground_truth",
        lambda **_kwargs: pytest.fail("nothing worth calibrating over"),
    )
    assert await bulk_learning.recalibrate_style_from_conversations(
        store, graph, VISITOR
    ) == 0


def test_a_conversational_quote_is_shaped_like_any_other_quote() -> None:
    quotes = speaker_quotes(_conversation_with_somebody_elses_avatar())
    documents = conversation_quote_documents(quotes, thread_id="thread-1")
    metadata = documents[0].metadata
    # The same metadata the media pipeline puts on a quote from an interview, so
    # nothing downstream has to know which kind of quote it is holding.
    assert metadata["namespace"] == "quote"
    assert metadata["is_target"] is True
    assert metadata["classified_situation"] == "tweets_or_quotes"
    assert metadata["synthetic"] is False
    assert metadata["source"] == "conversation"


def test_the_same_turn_always_gets_the_same_identifier() -> None:
    quotes = speaker_quotes(_conversation_with_somebody_elses_avatar())
    first = conversation_quote_documents(quotes, thread_id="thread-1")
    second = conversation_quote_documents(quotes, thread_id="thread-1")
    # A later pass must not add a second derived entry for a turn already known.
    assert [document.metadata["document_id"] for document in first] == [
        document.metadata["document_id"] for document in second
    ]
    other_thread = conversation_quote_documents(quotes, thread_id="thread-2")
    assert (
        first[0].metadata["document_id"] != other_thread[0].metadata["document_id"]
    )
