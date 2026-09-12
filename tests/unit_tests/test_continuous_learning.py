"""Continuous learning: engagement, sentiment, ratings, feedback, preferences, the sweep.

Covers the pure engagement arithmetic and rendering, rating polarity switching,
the one-batch prompt-section retrieval, the ask-what-feels-real gate, idle
account selection, the background sweep end to end (with the structured model
calls replaced), the ``observe_user`` graph node, and the message-lookup
helpers the feedback endpoint relies on. The store is a real ``InMemoryStore``
with a deterministic hashing embedder so similarity retrieval runs for real.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.learning.bulk_learning as bulk_learning
import src.anubis.utils.learning.sentiment as sentiment_module
import src.anubis.utils.nodes as nodes_module
from src.anubis.utils.learning.bulk_learning import (
    InferredPreference,
    InferredRealnessStatement,
    InferredUserPreferences,
    RatingPreferenceSummary,
    list_pending_accounts,
    mark_thread_pending,
    run_account_learning_sweep,
    select_idle_accounts,
)
from src.anubis.utils.learning.engagement import (
    apply_engagement,
    describe_elapsed,
    load_engagement_record,
    render_engagement_section,
)
from src.anubis.utils.learning.feedback import (
    list_thread_ratings,
    retrieve_learning_sections,
    should_ask_what_feels_real,
    store_feedback_message,
    store_message_rating,
    store_user_preference,
    store_what_feels_real,
)
from src.anubis.utils.learning.namespaces import (
    RATING_NEGATIVE,
    RATING_POSITIVE,
    learning_pending_namespace,
    preference_namespace,
    rating_namespace,
    sentiment_namespace,
)
from src.anubis.utils.learning.sentiment import (
    ConversationSentimentSummary,
    render_immediate_sentiment,
    render_transcript,
    save_current_conversation_sentiment,
)

USER = "user-1"
AVATAR = "avatar-1"
THREAD = "thread-1"
DIMENSIONS = 64


def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedder (cosine over shared words)."""
    vectors = []
    for text in texts:
        raw = [0.0] * DIMENSIONS
        for token in re.findall(r"[a-z']+", (text or "").lower()):
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMENSIONS
            raw[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        vectors.append([v / norm for v in raw])
    return vectors


def _make_store() -> InMemoryStore:
    return InMemoryStore(
        index={"dims": DIMENSIONS, "embed": _embed, "fields": ["document.kwargs.page_content"]}
    )


# ── engagement ──────────────────────────────────────────────────────────────


def test_apply_engagement_counts_messages_and_distinct_conversations():
    now = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    record = apply_engagement(None, "t1", now)
    record = apply_engagement(record, "t1", now + timedelta(minutes=5))
    record = apply_engagement(record, "t2", now + timedelta(days=1))

    assert record["message_count"] == 3
    assert record["conversation_count"] == 2
    assert record["first_engaged_at"] == now.isoformat()
    assert record["last_engaged_at"] == (now + timedelta(days=1)).isoformat()
    assert record["daily_message_counts"] == {"2026-09-03": 2, "2026-09-04": 1}


def test_render_engagement_section_reads_naturally():
    now = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    record = apply_engagement(None, "t1", now - timedelta(hours=3))
    rendered = render_engagement_section(record, now)
    assert "sent 1 message across 1 conversation" in rendered
    assert "3 hours ago" in rendered
    assert render_engagement_section(None) == ""


def test_describe_elapsed_bands():
    assert describe_elapsed(10) == "just now"
    assert describe_elapsed(60 * 20) == "20 minutes ago"
    assert describe_elapsed(3600 * 5) == "5 hours ago"
    assert describe_elapsed(86400 * 3) == "3 days ago"
    assert describe_elapsed(86400 * 21) == "3 weeks ago"


# ── sentiment rendering ─────────────────────────────────────────────────────


def test_render_immediate_sentiment_and_transcript():
    rendered = render_immediate_sentiment(
        {"emotion": "gratitude", "base_emotion": "joy", "score": 0.91}
    )
    assert "gratitude" in rendered and "joy" in rendered and "0.91" in rendered
    assert render_immediate_sentiment(None) == ""

    transcript = render_transcript(
        [
            HumanMessage(content="hi there"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
            AIMessage(content="hello friend"),
        ]
    )
    assert transcript == "USER: hi there\nAVATAR: hello friend"


# ── ratings ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rating_polarity_switch_moves_the_record():
    store = _make_store()
    await store_message_rating(
        store, USER, AVATAR, rating=RATING_POSITIVE, thread_id=THREAD,
        message_id="m1", avatar_message_text="Glad you asked, friend.",
        preceding_user_message_text="How are you?",
    )
    assert await store.aget(rating_namespace(USER, AVATAR, RATING_POSITIVE), "m1")
    await store_message_rating(
        store, USER, AVATAR, rating=RATING_NEGATIVE, thread_id=THREAD,
        message_id="m1", avatar_message_text="Glad you asked, friend.",
    )
    assert await store.aget(rating_namespace(USER, AVATAR, RATING_POSITIVE), "m1") is None
    assert await store.aget(rating_namespace(USER, AVATAR, RATING_NEGATIVE), "m1")

    ratings = await list_thread_ratings(store, USER, AVATAR, THREAD)
    assert ratings == [
        {"message_id": "m1", "rating": "negative", "rating_score": None, "rated_at": ratings[0]["rated_at"]}
    ]


# ── prompt-section retrieval ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_learning_sections_fills_every_section():
    store = _make_store()
    await store_feedback_message(
        store, USER, AVATAR, comment="Please keep your sailing stories shorter."
    )
    await store_message_rating(
        store, USER, AVATAR, rating=RATING_POSITIVE, thread_id="old-thread",
        message_id="m-good", avatar_message_text="Sailing at dawn is the best sailing.",
        preceding_user_message_text="Tell me about sailing.",
    )
    await store_message_rating(
        store, USER, AVATAR, rating=RATING_NEGATIVE, thread_id="old-thread",
        message_id="m-bad", avatar_message_text="Sailing is a nautical activity involving boats.",
    )
    await store_what_feels_real(
        store, USER, AVATAR, statement="The sailing jokes feel real.",
        statement_context="", polarity="feels_real",
    )
    await store_user_preference(
        store, USER, AVATAR, preference="The user wants short sailing replies.",
        preference_context="", category="communication_style",
    )
    await save_current_conversation_sentiment(
        store, USER, AVATAR, THREAD,
        {"sentiment_summary": "The user is cheerful about sailing.", "dominant_emotions": ["joy"],
         "overall_polarity": "positive", "engagement_signal": "rising"},
    )
    history_document = sentiment_module.build_sentiment_history_document(
        {"sentiment_summary": "Last time the user was wistful about sailing.",
         "dominant_emotions": ["sadness"], "overall_polarity": "negative",
         "engagement_signal": "steady", "summarized_at": "2026-09-01T00:00:00+00:00"},
        user_id=USER, assistant_id=AVATAR, thread_id="old-thread",
    )
    await store.aput(
        sentiment_namespace(USER, AVATAR), "history:old-thread",
        {"document": history_document.to_json()},
    )
    for _ in range(3):
        from src.anubis.utils.learning.engagement import record_engagement
        await record_engagement(store, USER, AVATAR, THREAD)

    sections = await retrieve_learning_sections(
        store, USER, AVATAR, thread_id=THREAD, query="tell me about sailing", limit=5
    )
    assert "sent 3 messages" in sections.user_engagement
    assert "sailing stories shorter" in sections.user_feedback_messages
    assert "rated positive" in sections.positively_rated_messages
    assert "rated negative" in sections.negatively_rated_messages
    assert "cheerful about sailing" in sections.current_conversation_sentiment
    assert "Engagement is rising" in sections.current_conversation_sentiment
    assert "wistful about sailing" in sections.conversation_sentiment_history
    assert "sailing jokes feel real" in sections.what_feels_real
    assert "short sailing replies" in sections.user_preferences
    assert sections.what_feels_real_recorded is True
    assert should_ask_what_feels_real(sections, 2) is False


@pytest.mark.asyncio
async def test_should_ask_what_feels_real_after_threshold_when_nothing_recorded():
    store = _make_store()
    from src.anubis.utils.learning.engagement import record_engagement

    for _ in range(5):
        await record_engagement(store, USER, AVATAR, THREAD)
    sections = await retrieve_learning_sections(
        store, USER, AVATAR, thread_id=THREAD, query="hello", limit=5
    )
    assert sections.what_feels_real_recorded is False
    assert should_ask_what_feels_real(sections, 5) is True
    assert should_ask_what_feels_real(sections, 6) is False
    assert should_ask_what_feels_real(sections, 0) is False


@pytest.mark.asyncio
async def test_duplicate_preferences_and_feedback_are_not_stored_twice():
    store = _make_store()
    first = await store_user_preference(
        store, USER, AVATAR, preference="The user prefers to be called Sam.",
        preference_context="", category="address",
    )
    second = await store_user_preference(
        store, USER, AVATAR, preference="the user prefers to be called sam.",
        preference_context="", category="address",
    )
    assert first is not None and second is None
    assert len(await store.asearch(preference_namespace(USER, AVATAR), limit=10)) == 1


# ── the sweep ───────────────────────────────────────────────────────────────


def test_select_idle_accounts_uses_newest_pending_message():
    now = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    pending = {
        "idle": [{"last_message_at": (now - timedelta(minutes=20)).isoformat()}],
        "busy": [
            {"last_message_at": (now - timedelta(minutes=20)).isoformat()},
            {"last_message_at": (now - timedelta(minutes=2)).isoformat()},
        ],
        "unstamped": [{"last_message_at": None}],
    }
    assert select_idle_accounts(pending, now=now, idle_seconds=600) == ["idle", "unstamped"]


class _FakeGraph:
    def __init__(self, messages_by_thread):
        self.messages_by_thread = messages_by_thread

    async def aget_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return SimpleNamespace(values={"messages": self.messages_by_thread.get(thread_id, [])})


@pytest.mark.asyncio
async def test_run_account_learning_sweep_end_to_end(monkeypatch):
    store = _make_store()
    messages = [
        HumanMessage(content="Call me Sam and keep your sailing stories short."),
        AIMessage(content="Sam it is. Sailing at dawn is the best sailing.", id="m1"),
        HumanMessage(content="That felt like the real you."),
    ]
    await mark_thread_pending(store, user_id=USER, assistant_id=AVATAR, thread_id=THREAD)
    await store_message_rating(
        store, USER, AVATAR, rating=RATING_POSITIVE, thread_id=THREAD, message_id="m1",
        avatar_message_text="Sam it is. Sailing at dawn is the best sailing.",
        preceding_user_message_text="Call me Sam and keep your sailing stories short.",
    )

    async def fake_invoke_structured(response_format, system_prompt, human_text):
        if response_format is ConversationSentimentSummary:
            return ConversationSentimentSummary(
                sentiment_summary="The user warmed up over the conversation.",
                dominant_emotions=["joy"], overall_polarity="positive", engagement_signal="rising",
            )
        if response_format is RatingPreferenceSummary:
            return RatingPreferenceSummary(
                positive_patterns=["The user responds well to short, direct replies."],
                negative_patterns=[],
            )
        if response_format is InferredUserPreferences:
            return InferredUserPreferences(
                preferences=[InferredPreference(
                    preference="The user prefers to be called Sam.", category="address",
                    evidence="Call me Sam",
                )],
                communication_style="The user writes short, direct messages.",
                what_feels_real=[InferredRealnessStatement(
                    statement="Short replies about sailing feel like the real avatar.",
                    polarity="feels_real", evidence="That felt like the real you.",
                )],
            )
        raise AssertionError(response_format)

    monkeypatch.setattr(sentiment_module, "invoke_structured", fake_invoke_structured)
    monkeypatch.setattr(bulk_learning, "invoke_structured", fake_invoke_structured)

    pending = await list_pending_accounts(store)
    assert list(pending) == [USER]
    counters = await run_account_learning_sweep(store, _FakeGraph({THREAD: messages}), USER, pending[USER])

    # ``style_recalibrated`` is 0 here because this account has no personal-avatar
    # pointer: the sweep skips folding a person's own words into their avatar's
    # style rather than guessing which avatar is theirs.
    assert counters == {
        "threads": 1,
        "ratings_aggregated": 1,
        "inferred_records": 3,
        "style_recalibrated": 0,
    }
    history = await store.aget(sentiment_namespace(USER, AVATAR), f"history:{THREAD}")
    assert "warmed up" in history.value["document"]["kwargs"]["page_content"]
    preferences = await store.asearch(preference_namespace(USER, AVATAR), limit=10)
    preference_texts = {item.value["document"]["kwargs"]["metadata"]["fact"] for item in preferences}
    assert preference_texts == {
        "The user responds well to short, direct replies.",
        "The user prefers to be called Sam.",
        "The user writes short, direct messages.",
    }
    rating = await store.aget(rating_namespace(USER, AVATAR, RATING_POSITIVE), "m1")
    assert rating.value["document"]["kwargs"]["metadata"]["aggregated"] is True
    assert await store.asearch(learning_pending_namespace(USER), limit=10) == []
    engagement = await load_engagement_record(store, USER, AVATAR)
    assert engagement["last_sweep_at"] is not None


# ── observe_user node ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_observe_user_node_records_signals(monkeypatch):
    store = _make_store()

    async def fake_classify(text):
        return {"emotion": "curiosity", "base_emotion": "surprise", "score": 0.8}

    async def fake_update_sentiment(store_, user_id, assistant_id, thread_id, messages):
        return {"sentiment_summary": "The user is curious.", "dominant_emotions": ["curiosity"],
                "overall_polarity": "positive", "engagement_signal": "steady"}

    monkeypatch.setattr(nodes_module, "classify_user_message_sentiment", fake_classify)
    monkeypatch.setattr(nodes_module, "update_current_conversation_sentiment", fake_update_sentiment)

    state = {
        "messages": [HumanMessage(content="What do you sail?")],
        "user_state": {"user_id": USER},
        "assistant_state": {"assistant_id": AVATAR},
    }
    config = {"configurable": {"thread_id": THREAD, "assistant_ctx": {"metadata": {"user_id": "creator-1"}}}}
    runtime = SimpleNamespace(store=store, context=None)

    update = await nodes_module.observe_user(state, config, runtime)

    assert "curiosity" in update["current_user_emotions"]
    assert "The user is curious." in update["current_conversation_sentiment"]
    engagement = await load_engagement_record(store, USER, AVATAR)
    assert engagement["message_count"] == 1
    pending = await store.aget(learning_pending_namespace(USER), THREAD)
    assert pending.value["value"]["creator_id"] == "creator-1"


@pytest.mark.asyncio
async def test_observe_user_node_ignores_non_human_turns():
    update = await nodes_module.observe_user(
        {"messages": [AIMessage(content="hi")], "user_state": {}, "assistant_state": {}},
        {"configurable": {}},
        SimpleNamespace(store=None, context=None),
    )
    assert update == {}


# ── feedback endpoint helpers ───────────────────────────────────────────────


def test_find_avatar_message_and_prompt_prefers_named_message_and_skips_tool_turns():
    from src.api.webapp import _find_avatar_message_and_prompt

    messages = [
        {"type": "human", "id": "h1", "content": "first question"},
        {"type": "ai", "id": "a1", "content": "", "tool_calls": [{"name": "t"}]},
        {"type": "ai", "id": "a2", "content": "first answer"},
        {"type": "human", "id": "h2", "content": "second question"},
        {"type": "ai", "id": "a3", "content": "second answer"},
    ]
    latest, prompt = _find_avatar_message_and_prompt(messages, None)
    assert latest["id"] == "a3" and prompt["id"] == "h2"
    named, prompt = _find_avatar_message_and_prompt(messages, "a2")
    assert named["id"] == "a2" and prompt["id"] == "h1"
    assert _find_avatar_message_and_prompt(messages, "missing") == (None, None)
    # A reply rated before the browser knew its stored id: the request id the
    # reply streamed under, else the text the browser quoted, finds the row.
    messages[4]["response_metadata"] = {"request_id": "req-3"}
    by_request, prompt = _find_avatar_message_and_prompt(messages, None, "req-3")
    assert by_request["id"] == "a3" and prompt["id"] == "h2"
    by_text, prompt = _find_avatar_message_and_prompt(
        messages, None, "req-unknown", "first answer"
    )
    assert by_text["id"] == "a2" and prompt["id"] == "h1"
    assert _find_avatar_message_and_prompt(messages, None, "req-unknown", "nope") == (
        None,
        None,
    )
