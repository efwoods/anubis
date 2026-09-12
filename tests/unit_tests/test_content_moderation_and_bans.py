"""AI monitoring: the fast screen, the deep judge, the graph, bans, and refusals.

Adapted from the z-anubis suite to the restructured moderation this line builds:
moderation is a compiled subgraph (``src/subgraphs/moderation_graph/graph.py``)
whose first stage is the cheap OpenAI moderation endpoint and whose second stage
is the structured-output terms-of-service judge. The chat path runs only the
cheap screen inline and refuses only on a hard block; the upload path judges
every document before anything is persisted.

Both stages are **fail-open by design**: a screening or judging outage must let
the turn through, never refuse it. The tests below assert that intent directly,
so a later change that "fixes" an outage into a refusal fails here.

Every model call is replaced with a fake, the pool is an in-memory fake that
records SQL, and Stripe is a fake client. Nothing here reaches a network.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

import src.anubis.utils.moderation.content_moderation as moderation
import src.anubis.utils.moderation.fast_screen as fast_screen_module
import src.security.bans as bans
from src.anubis.graph import (
    moderate_content_fast,
    refuse_for_violation,
    route_after_moderation,
)
from src.anubis.utils.moderation.content_moderation import (
    TermsAndServicesContentModeration,
    judge_documents,
    judge_text,
)
from src.anubis.utils.moderation.fast_screen import (
    FAST_SCREEN_BLOCK,
    FAST_SCREEN_CLEAN,
    FAST_SCREEN_SUSPECT,
)
from src.anubis.utils.state import GlobalState
from src.security.auth import refuse_if_banned
from src.security.bans import (
    BanSubject,
    ban_account,
    ban_subject_from_user,
    find_active_ban,
    lift_ban,
    list_bans,
)
from src.subgraphs.moderation_graph.graph import (
    MODERATION_MODE_MESSAGE,
    MODERATION_MODE_UPLOAD,
    moderate_documents_with_graph,
    moderate_text_with_graph,
)
from src.subgraphs.moderation_graph.media_node import (
    MEDIA_MODERATION_CONSUMERS,
    moderate_documents,
    route_media_moderation,
)

# ── fakes ───────────────────────────────────────────────────────────────────


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self._rows: list = []

    async def execute(self, sql, params=None):
        self.pool.executed.append((sql, params))
        self._rows = self.pool.respond(sql, params)

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return list(self._rows)


class _FakePool:
    """Records every statement; ``bans`` is the in-memory table."""

    def __init__(self):
        self.executed: list = []
        self.bans: list[dict] = []

    def _row(self, ban: dict):
        return tuple(
            ban.get(column.strip()) for column in bans._BAN_COLUMNS.split(",")
        )

    def respond(self, sql, params):
        if "INSERT INTO" in sql and "banned_accounts" in sql:
            self.bans.append(
                {
                    **params,
                    "banned_at": "2026-09-09T00:00:00+00:00",
                    "lifted_at": None,
                    "appeal_note": None,
                }
            )
            return []
        if "SET subscription_id" in sql:
            for ban in self.bans:
                if ban["ban_id"] == params["ban_id"]:
                    ban["subscription_id"] = params["subscription_id"]
                    ban["refund_id"] = params["refund_id"]
            return []
        if "SET lifted_at = now()" in sql:
            for ban in self.bans:
                if ban["ban_id"] == params["ban_id"] and ban["lifted_at"] is None:
                    ban["lifted_at"] = "2026-09-10T00:00:00+00:00"
                    ban["appeal_note"] = params["appeal_note"]
                    return [self._row(ban)]
            return []
        if "WHERE lifted_at IS NULL" in sql and "user_id = %(user_id)s" in sql:
            for ban in reversed(self.bans):
                if ban["lifted_at"] is not None:
                    continue
                if params["user_id"] and ban["user_id"] == params["user_id"]:
                    return [self._row(ban)]
                if params["hashed_ip"] and ban["hashed_ip"] == params["hashed_ip"]:
                    return [self._row(ban)]
                if params["email"] and (ban["email"] or "").lower() == params[
                    "email"
                ].lower():
                    return [self._row(ban)]
            return []
        if "ORDER BY banned_at DESC" in sql and "include_lifted" in sql:
            return [
                self._row(ban)
                for ban in self.bans
                if params["include_lifted"] or ban["lifted_at"] is None
            ]
        return []

    @asynccontextmanager
    async def connection(self):
        yield self

    @asynccontextmanager
    async def cursor(self):
        yield _FakeCursor(self)


class _FakeStripe:
    def __init__(self):
        self.deleted: list[str] = []
        self.refunds: list[dict] = []
        parent = self

        class Subscription:
            @staticmethod
            def list(customer, status, limit):
                return {"data": [{"id": "sub_live", "status": "active"}]}

            @staticmethod
            def delete(subscription_id):
                parent.deleted.append(subscription_id)

        class Invoice:
            @staticmethod
            def list(customer, status, limit):
                return {
                    "data": [
                        {
                            "id": "in_1",
                            "amount_paid": 2000,
                            "charge": "ch_1",
                            "payment_intent": None,
                        }
                    ]
                }

        class Refund:
            @staticmethod
            def create(**arguments):
                parent.refunds.append(arguments)
                return {"id": "re_1"}

        self.Subscription, self.Invoice, self.Refund = Subscription, Invoice, Refund


class _FakeScreenResult:
    """One result from the OpenAI moderation endpoint."""

    def __init__(self, flagged: bool, categories: dict, scores: dict):
        self.flagged = flagged
        self.categories = categories
        self.category_scores = scores


def _screen_response(flagged: bool, categories: dict, scores: dict):
    return SimpleNamespace(results=[_FakeScreenResult(flagged, categories, scores)])


def _moderation_context(**overrides):
    """A context with both stages on and a mid-range block threshold."""
    settings = {
        "content_moderation_enabled": "TRUE",
        "content_moderation_fast_screen_enabled": "TRUE",
        "content_moderation_deep_judge_enabled": "TRUE",
        "content_moderation_fast_screen_threshold": 0.9,
        "content_moderation_max_characters": 6000,
        "content_moderation_concurrency": 2,
        "ban_appeal_contact_email": "appeals@example.com",
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


@pytest.fixture(autouse=True)
def _clear_ban_cache():
    bans._clear_ban_cache()
    yield
    bans._clear_ban_cache()


@pytest.fixture
def clean_screen_everywhere(monkeypatch):
    """Screen every text as clean unless a test says otherwise."""

    async def _clean(text, context):
        return _screen_response(False, {}, {"violence": 0.01})

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", _clean)


def _verdict(violation: bool, reasoning: str = "") -> TermsAndServicesContentModeration:
    return TermsAndServicesContentModeration(
        violation=violation,
        reasoning=reasoning,
        violated_clauses=(
            ["Upload or share content that is unlawful"] if violation else []
        ),
    )


# ── stage two: the deep judge ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_judge_text_windows_long_content_and_reports_first_violation(monkeypatch):
    seen: list[str] = []

    async def fake_model(content, platforms=None):
        seen.append(content)
        return _verdict("BAD" in content, "contains BAD")

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_model)
    verdict = await judge_text("a" * 12 + "BAD" + "b" * 20, max_characters=12)
    assert verdict["violation"] is True
    assert verdict["reasoning"] == "contains BAD"
    # Judged window by window, stopping at the first violation.
    assert len(seen) == 2 and "BAD" in seen[1]
    assert (await judge_text("", max_characters=12))["violation"] is False


@pytest.mark.asyncio
async def test_judge_fails_open_on_model_error(monkeypatch):
    """A judge outage must let the turn through. Refusing here would be wrong."""

    async def broken(content, platforms=None):
        raise RuntimeError("model down")

    monkeypatch.setattr(moderation, "invoke_moderation_model", broken)
    verdict = await judge_text("anything")
    assert verdict["violation"] is False and "model down" in verdict["judge_error"]


@pytest.mark.asyncio
async def test_judge_documents_returns_the_violating_source(monkeypatch):
    async def fake_model(content, platforms=None):
        return _verdict("pirated" in content)

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_model)
    verdict = await judge_documents(
        [
            Document(
                page_content="a normal memoir",
                metadata={"source_filename": "memoir.txt"},
            ),
            Document(
                page_content="pirated film script",
                metadata={"source_filename": "film.txt"},
            ),
        ],
        concurrency=2,
    )
    assert verdict["violation"] is True and verdict["source"] == "film.txt"


# ── stage one: the cheap screen ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_screen_outcome_is_decided_by_the_threshold(monkeypatch):
    async def fake_screen(text, context):
        if "hateful" in text:
            return _screen_response(True, {"hate": True}, {"hate": 0.97})
        if "borderline" in text:
            return _screen_response(True, {"harassment": True}, {"harassment": 0.42})
        return _screen_response(False, {}, {"hate": 0.01})

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    context = _moderation_context()

    blocked = await fast_screen_module.fast_screen_text("hateful thing", context)
    assert blocked["outcome"] == FAST_SCREEN_BLOCK and blocked["categories"] == ["hate"]

    suspect = await fast_screen_module.fast_screen_text("borderline thing", context)
    assert suspect["outcome"] == FAST_SCREEN_SUSPECT

    clean = await fast_screen_module.fast_screen_text("an ordinary sentence", context)
    assert clean["outcome"] == FAST_SCREEN_CLEAN
    assert (await fast_screen_module.fast_screen_text("", context))["outcome"] == (
        FAST_SCREEN_CLEAN
    )


@pytest.mark.asyncio
async def test_screen_fails_open_on_endpoint_error(monkeypatch):
    """A screening outage must not refuse a turn either."""

    async def broken(text, context):
        raise RuntimeError("screen down")

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", broken)
    screen = await fast_screen_module.fast_screen_text("anything", _moderation_context())
    assert screen["outcome"] == FAST_SCREEN_CLEAN
    assert "screen down" in screen["screen_error"]


# ── the moderation graph: all four routes out of the screen ─────────────────


@pytest.mark.asyncio
async def test_a_clean_message_ends_without_paying_for_the_judge(monkeypatch):
    judged: list[str] = []

    async def fake_screen(text, context):
        return _screen_response(False, {}, {"hate": 0.01})

    async def fake_judge(content, platforms=None):
        judged.append(content)
        return _verdict(False)

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)

    result = await moderate_text_with_graph(
        "an ordinary sentence",
        mode=MODERATION_MODE_MESSAGE,
        context=_moderation_context(),
    )
    assert result["screen"]["outcome"] == FAST_SCREEN_CLEAN
    assert result["verdict"]["violation"] is False
    assert judged == []  # the chat path does not pay for a judge it does not need


@pytest.mark.asyncio
async def test_a_clean_upload_is_judged_anyway(monkeypatch):
    """Nothing violating may reach the index, and no person waits on an upload."""
    judged: list[str] = []

    async def fake_screen(text, context):
        return _screen_response(False, {}, {"hate": 0.01})

    async def fake_judge(content, platforms=None):
        judged.append(content)
        return _verdict("pirated" in content, "pirated material")

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)

    result = await moderate_documents_with_graph(
        [Document(page_content="pirated film script", metadata={})],
        mode=MODERATION_MODE_UPLOAD,
        context=_moderation_context(),
    )
    assert judged, "a clean screen must still be judged on the upload path"
    assert result["verdict"]["violation"] is True


@pytest.mark.asyncio
async def test_a_suspect_screen_is_settled_by_the_judge(monkeypatch):
    async def fake_screen(text, context):
        return _screen_response(True, {"harassment": True}, {"harassment": 0.42})

    async def fake_judge(content, platforms=None):
        return _verdict(True, "harassment")

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)

    result = await moderate_text_with_graph(
        "borderline thing", mode=MODERATION_MODE_MESSAGE, context=_moderation_context()
    )
    assert result["screen"]["outcome"] == FAST_SCREEN_SUSPECT
    assert result["verdict"]["violation"] is True and result["verdict"]["reasoning"] == (
        "harassment"
    )


@pytest.mark.asyncio
async def test_a_blocking_screen_refuses_without_calling_the_judge(monkeypatch):
    judged: list[str] = []

    async def fake_screen(text, context):
        return _screen_response(True, {"hate": True}, {"hate": 0.97})

    async def fake_judge(content, platforms=None):
        judged.append(content)
        return _verdict(False)

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)

    result = await moderate_text_with_graph(
        "hateful thing", mode=MODERATION_MODE_MESSAGE, context=_moderation_context()
    )
    assert result["screen"]["outcome"] == FAST_SCREEN_BLOCK
    assert result["verdict"]["violation"] is True
    assert judged == []  # the block is decided from the screen alone
    assert "hate" in result["verdict"]["reasoning"]


# ── the chat path ───────────────────────────────────────────────────────────


def test_route_after_moderation_refuses_only_a_confirmed_violation():
    assert (
        route_after_moderation({"moderation_response": {"verdict": {"violation": True}}})
        == "refuse_for_violation"
    )
    # No violation falls through to the ambient routing, which must still work.
    fell_through = route_after_moderation(
        {"moderation_response": {"verdict": {"violation": False}}, "messages": []}
    )
    assert fell_through != "refuse_for_violation"
    assert route_after_moderation({"messages": []}) != "refuse_for_violation"


@pytest.mark.asyncio
async def test_moderate_content_fast_screens_the_latest_human_message(monkeypatch):
    async def fake_screen(text, context):
        if "hateful" in text:
            return _screen_response(True, {"hate": True}, {"hate": 0.97})
        return _screen_response(False, {}, {"hate": 0.01})

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    runtime = SimpleNamespace(context=_moderation_context())

    update = await moderate_content_fast(
        {"messages": [HumanMessage(content="a hateful thing")]}, {}, runtime
    )
    assert update["moderation_response"]["verdict"]["violation"] is True

    update = await moderate_content_fast(
        {"messages": [HumanMessage(content="an ordinary sentence")]}, {}, runtime
    )
    assert update["moderation_response"]["verdict"]["violation"] is False

    # Only a human turn is screened; the avatar's own words are not.
    update = await moderate_content_fast(
        {"messages": [AIMessage(content="a hateful thing")]}, {}, runtime
    )
    assert update["moderation_response"]["verdict"]["violation"] is False


@pytest.mark.asyncio
async def test_moderate_content_fast_respects_the_flag_and_the_skip(monkeypatch):
    async def fake_screen(text, context):
        return _screen_response(True, {"hate": True}, {"hate": 0.97})

    monkeypatch.setattr(fast_screen_module, "invoke_fast_screen_model", fake_screen)
    state = {"messages": [HumanMessage(content="a hateful thing")]}

    disabled = SimpleNamespace(
        context=_moderation_context(content_moderation_enabled="FALSE")
    )
    update = await moderate_content_fast(state, {}, disabled)
    assert update["moderation_response"]["verdict"]["violation"] is False

    # A re-entered graph must not screen the same message twice.
    runtime = SimpleNamespace(context=_moderation_context())
    update = await moderate_content_fast(
        state, {"configurable": {"skip_content_moderation": True}}, runtime
    )
    assert update["moderation_response"]["verdict"]["violation"] is False


@pytest.mark.asyncio
async def test_moderate_content_fast_fails_open_when_the_graph_raises(monkeypatch):
    async def broken(text, *, mode, context):
        raise RuntimeError("graph down")

    monkeypatch.setattr(
        "src.subgraphs.moderation_graph.graph.moderate_text_with_graph", broken
    )
    update = await moderate_content_fast(
        {"messages": [HumanMessage(content="anything")]},
        {},
        SimpleNamespace(context=_moderation_context()),
    )
    assert update["moderation_response"]["verdict"]["violation"] is False


@pytest.mark.asyncio
async def test_refuse_for_violation_streams_the_refusal_and_the_ban_event():
    workflow = StateGraph(GlobalState)
    workflow.add_node("refuse_for_violation", refuse_for_violation)
    workflow.add_edge(START, "refuse_for_violation")
    workflow.add_edge("refuse_for_violation", END)
    compiled = workflow.compile()

    custom_events: list = []
    final_messages: list = []
    async for mode, payload in compiled.astream(
        {
            "messages": [HumanMessage(content="bad")],
            "moderation_response": {
                "verdict": {
                    "violation": True,
                    "reasoning": "r",
                    "violated_clauses": ["c"],
                }
            },
        },
        stream_mode=["custom", "updates"],
        context=None,
    ):
        if mode == "custom":
            custom_events.append(payload)
        elif mode == "updates":
            final_messages.extend(
                payload.get("refuse_for_violation", {}).get("messages", [])
            )

    assert custom_events[0]["type"] == "moderation_violation"
    assert custom_events[0]["reasoning"] == "r"
    assert custom_events[1]["type"] == "assistant_token"
    assert "contact@neuralnexus.site" in custom_events[1]["text"]
    assert (
        final_messages[0].response_metadata["moderation_violation"]["violation"] is True
    )


# ── the upload path ─────────────────────────────────────────────────────────


def test_route_media_moderation_gates_every_consumer():
    assert route_media_moderation({"moderation_violation": {"violation": True}}) == (
        "__end__"
    )
    consumers = route_media_moderation({})
    assert consumers == list(MEDIA_MODERATION_CONSUMERS)
    # The psychological analysis is gated too: a refused upload must not be read
    # for the target's psychology any more than it is indexed.
    assert "psycho_analysis" in consumers


@pytest.mark.asyncio
async def test_moderate_documents_records_the_verdict_on_a_violation(
    monkeypatch, clean_screen_everywhere
):
    async def fake_judge(content, platforms=None):
        return _verdict("pirated" in content)

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)
    runtime = SimpleNamespace(context=_moderation_context())

    clean = await moderate_documents(
        {"vectorstore_documents_to_be_indexed": [Document(page_content="memoir")]},
        {},
        runtime,
    )
    assert clean == {}

    dirty = await moderate_documents(
        {
            "documents_to_be_processed_for_adapter_training": [
                Document(page_content="pirated", metadata={"source_filename": "x"})
            ]
        },
        {},
        runtime,
    )
    assert dirty["moderation_violation"]["violation"] is True


@pytest.mark.asyncio
async def test_moderate_documents_respects_the_flag_and_the_skip(
    monkeypatch, clean_screen_everywhere
):
    async def fake_judge(content, platforms=None):
        return _verdict(True)

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_judge)
    state = {
        "vectorstore_documents_to_be_indexed": [Document(page_content="pirated")]
    }

    disabled = SimpleNamespace(
        context=_moderation_context(content_moderation_enabled="FALSE")
    )
    assert await moderate_documents(state, {}, disabled) == {}

    runtime = SimpleNamespace(context=_moderation_context())
    skipped = await moderate_documents(
        state, {"configurable": {"skip_content_moderation": True}}, runtime
    )
    assert skipped == {}


# ── bans ────────────────────────────────────────────────────────────────────


def test_ban_subject_from_user_authenticated_and_anonymous(monkeypatch):
    # A hash that stands for many callers is never recorded; pin the set so the
    # test does not depend on whether this machine is in development mode.
    monkeypatch.setattr(bans, "shared_hashed_ip_values", lambda: {"shared-hash"})

    user = {
        "identities": [{"user_id": "u1"}],
        "email": "Person@example.com",
        "app_metadata": {
            "stripe_customer_id": "cus_1",
            "subscription_status": {"subscription_id": "sub_1"},
        },
    }
    subject = ban_subject_from_user(user, "hashed-ip")
    assert subject == BanSubject(
        "u1", "hashed-ip", "Person@example.com", "cus_1", "sub_1", False
    )

    anonymous = {
        "identities": [{"user_id": "hashed-visitor"}],
        "is_anonymous": True,
        "app_metadata": {},
    }
    subject = ban_subject_from_user(anonymous, None)
    assert subject.user_id is None
    assert subject.hashed_ip == "hashed-visitor"
    assert subject.is_anonymous

    # A shared hash bans nobody by address: the ban falls back to the account.
    shared = ban_subject_from_user(user, "shared-hash")
    assert shared.hashed_ip is None and shared.user_id == "u1"


@pytest.mark.asyncio
async def test_ban_account_records_refunds_and_is_idempotent():
    pool = _FakePool()
    stripe = _FakeStripe()
    app_state = SimpleNamespace(
        pool=pool, stripe=stripe, context=SimpleNamespace(ban_refund_enabled="TRUE")
    )
    subject = BanSubject("u1", "ip1", "person@example.com", "cus_1", "sub_1", False)

    ban = await ban_account(
        app_state,
        subject,
        reason="unlawful content",
        violated_clauses=["clause"],
        source="message",
        excerpt="bad",
    )
    assert ban["refund_id"] == "re_1" and ban["subscription_id"] == "sub_1"
    assert stripe.deleted == ["sub_1", "sub_live"]
    assert stripe.refunds[0]["charge"] == "ch_1"

    again = await ban_account(
        app_state,
        subject,
        reason="again",
        violated_clauses=None,
        source="media",
        excerpt=None,
    )
    assert again["ban_id"] == ban["ban_id"]
    assert len(pool.bans) == 1  # banning twice does not record a second row

    assert (await find_active_ban(pool, email="PERSON@example.com"))["ban_id"] == (
        ban["ban_id"]
    )
    assert (await find_active_ban(pool, hashed_ip="ip1"))["ban_id"] == ban["ban_id"]
    assert await find_active_ban(pool, user_id="someone-else") is None


@pytest.mark.asyncio
async def test_find_active_ban_is_cached_until_a_write():
    pool = _FakePool()
    assert await find_active_ban(pool, user_id="u1") is None
    statements_before = len(pool.executed)
    assert await find_active_ban(pool, user_id="u1") is None
    assert len(pool.executed) == statements_before  # served from cache

    app_state = SimpleNamespace(
        pool=pool, stripe=None, context=SimpleNamespace(ban_refund_enabled="FALSE")
    )
    await ban_account(
        app_state,
        BanSubject("u1", None, None, None, None, False),
        reason="r",
        violated_clauses=[],
        source="message",
        excerpt=None,
    )
    # The write cleared the cache, so the ban is enforced on the very next request.
    assert await find_active_ban(pool, user_id="u1") is not None


@pytest.mark.asyncio
async def test_lift_ban_and_list_bans():
    pool = _FakePool()
    app_state = SimpleNamespace(
        pool=pool, stripe=None, context=SimpleNamespace(ban_refund_enabled="FALSE")
    )
    ban = await ban_account(
        app_state,
        BanSubject("u1", "ip1", "a@b.c", None, None, False),
        reason="r",
        violated_clauses=[],
        source="message",
        excerpt=None,
    )
    assert len(await list_bans(pool)) == 1

    lifted = await lift_ban(pool, ban["ban_id"], "appeal accepted")
    assert lifted["appeal_note"] == "appeal accepted" and lifted["lifted_at"]
    assert await find_active_ban(pool, user_id="u1") is None
    assert await list_bans(pool) == []
    assert len(await list_bans(pool, include_lifted=True)) == 1
    assert await lift_ban(pool, "missing", None) is None


# ── auth refusals ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refuse_if_banned_raises_403_only_for_active_bans():
    pool = _FakePool()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                pool=pool,
                context=SimpleNamespace(ban_appeal_contact_email="appeals@example.com"),
            )
        )
    )
    await refuse_if_banned(request, user_id="u1", email="a@b.c")  # nothing banned yet

    app_state = SimpleNamespace(
        pool=pool, stripe=None, context=SimpleNamespace(ban_refund_enabled="FALSE")
    )
    await ban_account(
        app_state,
        BanSubject("u1", "ip1", "a@b.c", None, None, False),
        reason="unlawful content",
        violated_clauses=[],
        source="message",
        excerpt=None,
    )

    for identifiers in ({"user_id": "u1"}, {"hashed_ip": "ip1"}, {"email": "A@B.C"}):
        with pytest.raises(HTTPException) as refused:
            await refuse_if_banned(request, **identifiers)
        assert refused.value.status_code == 403
        assert "appeals@example.com" in refused.value.detail
        assert "unlawful content" in refused.value.detail

    # No database means no ban can be known, so nothing is refused.
    no_pool = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    await refuse_if_banned(no_pool, user_id="u1")


# ── media jobs ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_media_job_bans_the_uploader_on_violation(monkeypatch):
    from src.api import media_jobs

    registry: dict = {}
    master = media_jobs.create_master_job(registry, user_id="u1", assistant_id="a1")
    child = media_jobs.create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="film.txt",
        namespace_filename="film",
    )
    master.child_ids.append(child.job_id)

    async def fake_single_item(
        child_job,
        master_job,
        media_file,
        config,
        store,
        context,
        existing_namespaces=None,
    ):
        verdict = {
            "violation": True,
            "reasoning": "pirated",
            "violated_clauses": [],
            "source": "film.txt",
        }
        child_job.moderation_violation = verdict
        master_job.moderation_violation = dict(verdict)
        media_jobs.finish_job(
            child_job, error="Content violates the terms of service"
        )

    async def fake_calibrate(*args, **kwargs):
        return None

    monkeypatch.setattr(media_jobs, "run_single_item_job", fake_single_item)
    monkeypatch.setattr(
        media_jobs, "_calibrate_ground_truth_after_batch", fake_calibrate
    )
    banned: list[dict] = []

    async def on_violation(verdict):
        banned.append(verdict)

    await media_jobs.run_batch_media_job(
        master,
        [{"child": child, "media_file": {"filename": "film.txt"}}],
        {},
        None,
        SimpleNamespace(),
        concurrency=1,
        registry=registry,
        on_moderation_violation=on_violation,
    )

    assert banned == [
        {
            "violation": True,
            "reasoning": "pirated",
            "violated_clauses": [],
            "source": "film.txt",
        }
    ]
    assert master.status == "completed"
    assert master.result["moderation_violation"]["reasoning"] == "pirated"
    assert any(event.get("stage") == "account_banned" for event in master.events)


# ── the judge's prompt actually contains the policies ────────────────────────
#
# These exist because the judge shipped reading its instruction paragraph and two
# file paths. ``src/anubis/utils/prompts/legal/`` had no ``__init__.py``, so
# ``from ...legal import TERMS_OF_SERVICE`` bound the SUBMODULE rather than the
# string inside it, and the prompt interpolated ``<module '...' from '...'>``.
# Every other test in this file patches ``invoke_moderation_model``, so the whole
# suite exercised the judge's plumbing without ever looking at what the model was
# actually asked. A judge with no policy returns "no violation" for everything,
# which is indistinguishable from a working judge on clean content.


def test_the_policies_are_strings_not_modules():
    """The import that broke this bound modules; nothing downstream noticed."""
    from src.anubis.utils.prompts.legal import PRIVACY_POLICY, TERMS_OF_SERVICE

    assert isinstance(TERMS_OF_SERVICE, str)
    assert isinstance(PRIVACY_POLICY, str)


def test_the_judge_prompt_carries_the_real_policy_documents():
    from src.anubis.utils.moderation.content_moderation import (
        build_moderation_system_prompt,
    )
    from src.anubis.utils.prompts.legal import PRIVACY_POLICY, TERMS_OF_SERVICE

    prompt = build_moderation_system_prompt()
    assert TERMS_OF_SERVICE.strip() in prompt
    assert PRIVACY_POLICY.strip() in prompt


def test_the_judge_prompt_never_interpolates_a_module_repr():
    """The exact signature of the original bug, in the prompt the model receives."""
    from src.anubis.utils.moderation.content_moderation import (
        build_moderation_system_prompt,
    )

    prompt = build_moderation_system_prompt()
    assert "<module " not in prompt
    assert ".py'>" not in prompt


def test_the_judge_prompt_is_substantial_enough_to_judge_against():
    """A prompt this short is the instructions alone, with no policy behind them.

    The bound-module version was 1385 characters. Any real pair of policy
    documents is far longer, so a length floor catches a future regression that
    empties or shortens them without anyone reading the prompt.
    """
    from src.anubis.utils.moderation.content_moderation import (
        build_moderation_system_prompt,
    )

    assert len(build_moderation_system_prompt()) > 4000


def test_clauses_can_be_quoted_from_what_the_judge_was_given():
    """A ban record and its appeal rest on violated_clauses being real lines."""
    from src.anubis.utils.moderation.content_moderation import (
        build_moderation_system_prompt,
    )
    from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE

    prompt = build_moderation_system_prompt()
    quotable = [
        line.strip()
        for line in TERMS_OF_SERVICE.splitlines()
        if len(line.strip()) > 40
    ]
    assert quotable, "the terms of service carry no quotable clause"
    assert quotable[0] in prompt


@pytest.mark.asyncio
async def test_platform_rules_reach_the_judge_and_stay_out_of_violated_clauses(
    monkeypatch,
):
    """A third-party rule is a summary of another company's document, not a quotable line.

    ``violated_clauses`` is what a ban record and its appeal rest on, so it holds
    verbatim lines of our own two documents and nothing else. A platform rule
    belongs in ``violated_platform_rules``.
    """
    import src.anubis.utils.moderation.content_moderation as moderation

    seen: dict = {}

    async def fake_model(content, platforms=None):
        seen["platforms"] = platforms

        class Verdict:
            violation = True
            reasoning = "broke a platform rule"
            violated_clauses: list[str] = []
            violated_platform_rules = ["Twitch: no automated moderation without disclosure"]

        return Verdict()

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_model)
    verdict = await moderation.judge_text("something", platforms=["twitch"])
    assert seen["platforms"] == ["twitch"]
    assert verdict["violated_platform_rules"] == [
        "Twitch: no automated moderation without disclosure"
    ]
    assert verdict["violated_clauses"] == []


@pytest.mark.asyncio
async def test_the_judge_is_always_called_with_both_arguments(monkeypatch):
    """The call shape must not depend on whether a platform was named.

    It briefly branched on arity so that older one-argument test doubles kept
    working, which meant the platform call shape was never exercised.
    """
    import src.anubis.utils.moderation.content_moderation as moderation

    calls: list = []

    async def fake_model(content, platforms=None):
        calls.append((content, platforms))

        class Verdict:
            violation = False
            reasoning = ""
            violated_clauses: list[str] = []
            violated_platform_rules: list[str] = []

        return Verdict()

    monkeypatch.setattr(moderation, "invoke_moderation_model", fake_model)
    await moderation.judge_text("no platforms named")
    await moderation.judge_text("platform named", platforms=["slack"])
    assert [platforms for _, platforms in calls] == [None, ["slack"]]


def test_a_named_platforms_rules_are_rendered_into_the_prompt():
    from src.anubis.utils.moderation.content_moderation import (
        build_moderation_system_prompt,
    )

    without = build_moderation_system_prompt()
    with_twitch = build_moderation_system_prompt(["twitch"])
    assert len(with_twitch) > len(without)
    assert "THIRD_PARTY_PLATFORM_POLICIES" in with_twitch
