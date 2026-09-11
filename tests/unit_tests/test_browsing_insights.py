"""Learning who the owner is from the owner's own web browsing.

The machine is faked at the one seam that crosses the machine boundary — the
Model Context Protocol call — and the model at the one seam that costs money,
so the digest, the thresholds, the watermarks, and every write are exercised
without a daemon, a browser, or a model.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.browsing import digest as digest_module
from src.anubis.utils.browsing import history_client
from src.anubis.utils.browsing import sweeper as sweeper_module
from src.anubis.utils.browsing.insights import (
    BROWSING_DIMENSION,
    BrowsingFact,
    BrowsingInsights,
    BrowsingTrait,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
USER_ID = "auth0|owner"
ASSISTANT_ID = "assistant-1"


def visit(minutes_ago: int, url: str, title: str, *, host: str = "", search: str = ""):
    """One visit in the shape every platform's connector returns."""
    from urllib.parse import urlparse

    return {
        "visited_at": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
        "url": url,
        "title": title,
        "host": host or (urlparse(url).hostname or ""),
        "search_terms": search,
        "visit_count": 1,
        "duration_seconds": 42.0,
        "browser_name": "Google Chrome",
        "profile_name": "Default",
        "family": "chromium",
        "device_label": "Evan's desktop",
    }


VISITS = [
    visit(600, "https://news.ycombinator.com/", "Hacker News"),
    visit(
        540,
        "https://www.google.com/search?q=qlora+fine+tuning+llama",
        "qlora fine tuning llama - Google Search",
        search="qlora fine tuning llama",
    ),
    visit(500, "https://github.com/huggingface/peft", "PEFT — parameter efficient fine tuning"),
    visit(400, "https://arxiv.org/abs/2305.14314", "QLoRA: Efficient Finetuning"),
    visit(60, "https://news.ycombinator.com/item?id=123", "Ask HN: how do you fine tune"),
    visit(30, "https://accounts.google.com/signin", "Sign in"),
    visit(20, "https://www.googletagmanager.com/gtm.js", ""),
]


def _context(**overrides):
    values = dict(
        browsing_insights_enabled="TRUE",
        browsing_insights_poll_seconds=300,
        browsing_insights_minimum_new_visits=25,
        browsing_insights_minimum_seconds_between_analyses=900,
        browsing_insights_backfill_days=30,
        browsing_insights_max_visits_per_pass=2000,
        browsing_insights_max_digest_characters=24000,
        browsing_insights_report_enabled="FALSE",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeStore:
    """A store that remembers what was written, and can search by substring."""

    def __init__(self) -> None:
        self.records: dict[tuple, dict[str, dict]] = {}

    async def aput(self, namespace, key, value):
        self.records.setdefault(tuple(namespace), {})[key] = value

    async def aget(self, namespace, key):
        value = self.records.get(tuple(namespace), {}).get(key)
        return SimpleNamespace(value=value, key=key) if value is not None else None

    async def asearch(self, namespace, query=None, limit=100):
        found = []
        for stored_namespace, rows in self.records.items():
            if stored_namespace[: len(tuple(namespace))] != tuple(namespace):
                continue
            for key, value in rows.items():
                found.append(SimpleNamespace(namespace=stored_namespace, key=key, value=value))
        return found[:limit]

    async def abatch(self, operations):
        for operation in operations:
            await self.aput(operation.namespace, operation.key, operation.value)

    async def adelete(self, namespace, key):
        self.records.get(tuple(namespace), {}).pop(key, None)

    def namespace_rows(self, *namespace_parts):
        wanted = tuple(namespace_parts)
        rows: list[dict] = []
        for stored_namespace, stored in self.records.items():
            if stored_namespace[: len(wanted)] == wanted:
                rows.extend(stored.values())
        return rows


def _connection(device_id="device-1", label="Evan's desktop", online=True):
    return SimpleNamespace(
        device_id=device_id,
        device_label=label,
        platform="linux",
        online=online,
        url="http://127.0.0.1:8000/mcp/relay/device-1",
        server_name="neuralnexus",
        allowed_roots=(),
        device_secret="secret",
    )


SAMPLE_INSIGHTS = BrowsingInsights(
    facts=[
        BrowsingFact(
            fact="Evan is teaching himself to fine tune language models with QLoRA",
            fact_context="Read across a week of browsing on his desktop",
            evidence="Searched 'qlora fine tuning llama'; read the QLoRA paper and the PEFT repository",
            confidence=0.9,
        ),
        BrowsingFact(
            fact="Evan reads Hacker News daily",
            fact_context="Recurring across the period",
            evidence="news.ycombinator.com visited most days",
            confidence=0.8,
        ),
        BrowsingFact(
            fact="Evan might be considering a holiday",
            fact_context="One visit",
            evidence="A single visit to a travel site",
            confidence=0.2,
        ),
    ],
    traits=[
        BrowsingTrait(
            trait="depth_of_focus",
            score=0.85,
            confidence=0.7,
            first_person_statement="When a subject grabs me I read it all the way down.",
            supporting_evidence="Search, then paper, then repository, then forum thread.",
        ),
        BrowsingTrait(
            trait="learning_drive",
            score=0.8,
            confidence=0.65,
            first_person_statement="I teach myself the thing I need rather than waiting.",
            supporting_evidence="A week of fine-tuning material.",
        ),
        BrowsingTrait(trait="entertainment_seeking", score=0.0, confidence=0.4),
    ],
    summary_markdown="You spent the week learning how to fine tune models.",
)


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def test_pages_the_owners_software_opened_are_not_read_as_interests():
    kept = digest_module.meaningful_visits(VISITS)
    hosts = {entry["host"] for entry in kept}
    assert "accounts.google.com" not in hosts
    assert "www.googletagmanager.com" not in hosts
    assert "news.ycombinator.com" in hosts


def test_the_digest_carries_the_searches_the_titles_and_the_counts():
    digest = digest_module.render_digest(VISITS)
    assert "qlora fine tuning llama" in digest
    assert "PEFT — parameter efficient fine tuning" in digest
    assert "news.ycombinator.com — 2 visits" in digest
    assert "Websites visited most" in digest


def test_the_digest_names_the_websites_that_are_new_in_this_period():
    digest = digest_module.render_digest(
        VISITS, known_hosts=["news.ycombinator.com", "github.com"]
    )
    new_section = digest.split("--- Websites new in this period ---")[1]
    assert "arxiv.org" in new_section
    assert "news.ycombinator.com" not in new_section.split("---")[0]


def test_a_bare_home_page_address_is_not_repeated_as_evidence():
    digest = digest_module.render_digest(VISITS)
    specific = digest.split("--- Specific addresses opened ---")[1]
    assert "https://github.com/huggingface/peft" in specific
    assert "https://news.ycombinator.com/\n" not in specific


def test_an_empty_period_produces_no_digest_and_therefore_no_model_call():
    assert digest_module.render_digest([]) == ""
    assert digest_module.render_digest(
        [visit(5, "https://localhost:3000/", "dev server")]
    ) == ""


def test_the_digest_is_trimmed_from_the_end_so_the_searches_survive():
    digest = digest_module.render_digest(VISITS, max_characters=400)
    assert len(digest) <= 420
    assert "BROWSING RECORD" in digest


def test_the_summary_counts_the_days_the_hours_and_the_browsers():
    summary = digest_module.summarize_visits(VISITS)
    assert summary["visit_count"] == 5
    # news.ycombinator.com (twice), www.google.com, github.com, arxiv.org.
    assert summary["distinct_hosts"] == 4
    assert summary["browsers"] == {"Google Chrome": 5}
    assert summary["searches"] == ["qlora fine tuning llama"]


# ---------------------------------------------------------------------------
# Asking the machine what is new
# ---------------------------------------------------------------------------


def _fake_tool_calls(monkeypatch, *, summary=None, history=None, calls=None):
    async def call_mcp_filesystem_tool(connection, tool_name, tool_arguments):
        if calls is not None:
            calls.append((tool_name, tool_arguments))
        if tool_name == history_client.TOOL_BROWSING_ACTIVITY:
            if callable(summary):
                return summary(tool_arguments)
            return summary if summary is not None else {
                "visit_count": len(VISITS),
                "latest_visit": VISITS[-1]["visited_at"],
                "watermark": VISITS[-1]["visited_at"],
                "platform": "Linux",
            }
        if tool_name == history_client.TOOL_READ_HISTORY:
            if callable(history):
                return history(tool_arguments)
            return history if history is not None else {
                "visits": [dict(entry) for entry in VISITS],
                "visit_count": len(VISITS),
                "watermark": VISITS[-1]["visited_at"],
                "platform": "Linux",
                "profiles_read": ["Google Chrome:Default"],
                "profiles_unreadable": [],
            }
        raise AssertionError(f"unexpected tool {tool_name}")

    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.mcp_client.call_mcp_filesystem_tool",
        call_mcp_filesystem_tool,
    )


def test_the_cheap_question_is_asked_with_the_watermark(monkeypatch):
    calls: list = []
    _fake_tool_calls(monkeypatch, calls=calls)
    counted = asyncio.run(history_client.new_visit_count(_connection(), "2026-09-01T00:00:00+00:00"))
    assert counted["status"] == "ok"
    assert counted["visit_count"] == len(VISITS)
    assert calls == [
        (history_client.TOOL_BROWSING_ACTIVITY, {"since": "2026-09-01T00:00:00+00:00"})
    ]


def test_a_machine_that_shares_no_history_says_so_rather_than_reading_as_empty(monkeypatch):
    _fake_tool_calls(
        monkeypatch,
        summary={"disabled": True, "reason": "Browsing history is not shared from this machine."},
    )
    counted = asyncio.run(history_client.new_visit_count(_connection(), ""))
    assert counted["status"] == "disabled"
    assert "not shared" in counted["detail"]


def test_a_connector_too_old_for_the_history_tools_is_reported_not_raised(monkeypatch):
    async def call_mcp_filesystem_tool(connection, tool_name, tool_arguments):
        raise RuntimeError("The server does not expose a tool named 'read_browser_history'.")

    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.mcp_client.call_mcp_filesystem_tool",
        call_mcp_filesystem_tool,
    )
    counted = asyncio.run(history_client.new_visit_count(_connection(), ""))
    assert counted["status"] == "unsupported"
    assert "Update the connector" in counted["detail"]


def test_every_visit_read_says_which_machine_it_came_from(monkeypatch):
    _fake_tool_calls(monkeypatch)
    read = asyncio.run(history_client.read_new_visits(_connection(), "", limit=100))
    assert read["status"] == "ok"
    assert {entry["device_label"] for entry in read["visits"]} == {"Evan's desktop"}


def test_watermarks_are_kept_per_machine():
    store = FakeStore()
    asyncio.run(
        history_client.write_watermark(
            store, USER_ID, ASSISTANT_ID, "device-1", {"watermark": "2026-09-10T00:00:00+00:00"}
        )
    )
    asyncio.run(
        history_client.write_watermark(
            store, USER_ID, ASSISTANT_ID, "device-2", {"watermark": "2026-08-01T00:00:00+00:00"}
        )
    )
    first = asyncio.run(history_client.read_watermark(store, USER_ID, ASSISTANT_ID, "device-1"))
    second = asyncio.run(history_client.read_watermark(store, USER_ID, ASSISTANT_ID, "device-2"))
    assert first["watermark"] != second["watermark"]
    assert asyncio.run(
        history_client.read_watermark(store, USER_ID, ASSISTANT_ID, "device-3")
    ) == {}


# ---------------------------------------------------------------------------
# What one pass writes
# ---------------------------------------------------------------------------


def _run_pass(store, monkeypatch, *, context=None, force=False, insights=SAMPLE_INSIGHTS):
    async def analyze_visits(visits, **kwargs):
        return insights

    monkeypatch.setattr(sweeper_module, "analyze_visits", analyze_visits)
    return asyncio.run(
        sweeper_module.analyse_machine(
            store,
            _connection(),
            context=context or _context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            target_name="Evan",
            force=force,
        )
    )


def test_a_first_pass_over_a_new_machine_runs_whatever_the_visit_count(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch, summary={"visit_count": 3, "watermark": VISITS[-1]["visited_at"]})
    outcome = _run_pass(store, monkeypatch)
    assert outcome["analysed"] is True
    assert outcome["first_pass"] is True


def test_a_later_pass_waits_until_enough_new_browsing_has_happened(monkeypatch):
    store = FakeStore()
    asyncio.run(
        history_client.write_watermark(
            store,
            USER_ID,
            ASSISTANT_ID,
            "device-1",
            {"watermark": "2026-09-01T00:00:00+00:00", "analysed_at": "2026-01-01T00:00:00+00:00"},
        )
    )
    _fake_tool_calls(monkeypatch, summary={"visit_count": 3, "watermark": "x"})
    outcome = _run_pass(store, monkeypatch)
    assert outcome["analysed"] is False
    assert "only 3 new visits" in outcome["reason"]


def test_a_machine_analysed_moments_ago_is_left_alone(monkeypatch):
    store = FakeStore()
    asyncio.run(
        history_client.write_watermark(
            store,
            USER_ID,
            ASSISTANT_ID,
            "device-1",
            {
                "watermark": "2026-09-01T00:00:00+00:00",
                "analysed_at": datetime.now(UTC).isoformat(),
            },
        )
    )
    _fake_tool_calls(monkeypatch, summary={"visit_count": 500, "watermark": "x"})
    outcome = _run_pass(store, monkeypatch)
    assert outcome["analysed"] is False
    assert outcome["reason"] == "analysed too recently"


def test_a_quiet_machine_costs_nothing(monkeypatch):
    store = FakeStore()
    calls: list = []
    _fake_tool_calls(monkeypatch, summary={"visit_count": 0, "watermark": "x"}, calls=calls)
    outcome = _run_pass(store, monkeypatch)
    assert outcome["analysed"] is False
    assert outcome["reason"] == "no new browsing"
    # The expensive tool was never called, so no rows travelled and no model ran.
    assert [name for name, _ in calls] == [history_client.TOOL_BROWSING_ACTIVITY]


def test_the_owner_asking_waives_the_thresholds(monkeypatch):
    store = FakeStore()
    asyncio.run(
        history_client.write_watermark(
            store,
            USER_ID,
            ASSISTANT_ID,
            "device-1",
            {
                "watermark": "2026-09-01T00:00:00+00:00",
                "analysed_at": datetime.now(UTC).isoformat(),
            },
        )
    )
    _fake_tool_calls(monkeypatch, summary={"visit_count": 2, "watermark": "x"})
    outcome = _run_pass(store, monkeypatch, force=True)
    assert outcome["analysed"] is True


def test_confident_facts_are_written_as_identity_the_avatar_carries(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    outcome = _run_pass(store, monkeypatch)
    facts = outcome["facts_written"]
    assert "Evan is teaching himself to fine tune language models with QLoRA" in facts
    assert "Evan reads Hacker News daily" in facts
    # The low-confidence guess never becomes something the avatar states.
    assert not any("holiday" in fact for fact in facts)
    identity_rows = store.namespace_rows(ASSISTANT_ID, USER_ID, "identity")
    assert len(identity_rows) == 2
    assert "document" in identity_rows[0]


def test_a_fact_already_known_is_not_written_twice(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    _run_pass(store, monkeypatch)
    first_count = len(store.namespace_rows(ASSISTANT_ID, USER_ID, "identity"))
    # A second pass produces the same facts; the identity namespace is loaded
    # whole into every reply, so a duplicate would cost prompt space forever.
    store.records.pop((USER_ID, ASSISTANT_ID, "browsing_watermark"), None)
    _run_pass(store, monkeypatch)
    assert len(store.namespace_rows(ASSISTANT_ID, USER_ID, "identity")) == first_count


def test_traits_fold_into_the_psychological_profile(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    outcome = _run_pass(store, monkeypatch)
    assert outcome["trait_count"] == 3
    profile_rows = store.namespace_rows(USER_ID, ASSISTANT_ID, "psychological_profile")
    assert profile_rows
    dimensions = profile_rows[0]["dimensions"]
    assert BROWSING_DIMENSION in dimensions
    traits = dimensions[BROWSING_DIMENSION]["traits"]
    assert traits["depth_of_focus"]["score"] == pytest.approx(0.85, abs=0.01)


def test_a_second_pass_moves_a_score_rather_than_stacking_a_second_copy(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    _run_pass(store, monkeypatch)
    quieter = BrowsingInsights(
        facts=[],
        traits=[
            BrowsingTrait(
                trait="depth_of_focus",
                score=0.25,
                confidence=0.7,
                first_person_statement="I skimmed a lot this week.",
            )
        ],
        summary_markdown="A scattered week.",
    )
    store.records.pop((USER_ID, ASSISTANT_ID, "browsing_watermark"), None)
    _run_pass(store, monkeypatch, insights=quieter)
    profile = store.namespace_rows(USER_ID, ASSISTANT_ID, "psychological_profile")[0]
    score = profile["dimensions"][BROWSING_DIMENSION]["traits"]["depth_of_focus"]["score"]
    assert 0.25 < score < 0.85


def test_the_watermark_advances_to_the_newest_visit_actually_read(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    outcome = _run_pass(store, monkeypatch)
    assert outcome["watermark"] == VISITS[-1]["visited_at"]
    record = asyncio.run(
        history_client.read_watermark(store, USER_ID, ASSISTANT_ID, "device-1")
    )
    assert record["watermark"] == VISITS[-1]["visited_at"]
    assert record["passes"] == 1
    assert "news.ycombinator.com" in record["hosts"]


def test_the_next_pass_asks_only_for_what_is_newer_than_the_watermark(monkeypatch):
    store = FakeStore()
    calls: list = []
    _fake_tool_calls(monkeypatch, calls=calls)
    _run_pass(store, monkeypatch)
    calls.clear()
    _fake_tool_calls(monkeypatch, summary={"visit_count": 0, "watermark": "x"}, calls=calls)
    _run_pass(store, monkeypatch)
    assert calls[0][1]["since"] == VISITS[-1]["visited_at"]


def test_the_machines_own_websites_are_remembered_so_the_next_pass_knows_what_is_new(
    monkeypatch,
):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)
    _run_pass(store, monkeypatch)
    record = asyncio.run(
        history_client.read_watermark(store, USER_ID, ASSISTANT_ID, "device-1")
    )
    assert "arxiv.org" in record["hosts"]


def test_a_pass_that_learns_nothing_writes_no_watermark_and_no_findings(monkeypatch):
    store = FakeStore()
    _fake_tool_calls(monkeypatch)

    async def analyze_visits(visits, **kwargs):
        return None

    monkeypatch.setattr(sweeper_module, "analyze_visits", analyze_visits)
    outcome = asyncio.run(
        sweeper_module.analyse_machine(
            store,
            _connection(),
            context=_context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            target_name="Evan",
        )
    )
    assert outcome["analysed"] is False
    assert not store.namespace_rows(ASSISTANT_ID, USER_ID, "identity")
    assert not store.namespace_rows(USER_ID, ASSISTANT_ID, "browsing_watermark")


# ---------------------------------------------------------------------------
# The loop over accounts
# ---------------------------------------------------------------------------


def test_only_accounts_with_a_machine_connected_are_swept(monkeypatch):
    store = FakeStore()
    sessions = [SimpleNamespace(user_id=USER_ID, device_id="device-1")]
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.all_sessions", lambda: sessions
    )

    async def read_user_connections(store_argument, user_id):
        return [
            {"device_id": "device-1", "assistant_id": ASSISTANT_ID, "status": "connected"},
            {"device_id": "device-9", "assistant_id": "other", "status": "declined"},
        ]

    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.discovery.read_user_connections",
        read_user_connections,
    )
    pairs = asyncio.run(sweeper_module.accounts_with_machines_online(store))
    assert pairs == [(USER_ID, ASSISTANT_ID)]


def test_one_machine_failing_does_not_stop_the_others(monkeypatch):
    store = FakeStore()

    async def online_connections(store_argument, user_id, assistant_id):
        return [_connection("device-1", "desktop"), _connection("device-2", "laptop")]

    monkeypatch.setattr(sweeper_module, "online_connections", online_connections)

    async def analyse_machine(store_argument, connection, **kwargs):
        if connection.device_label == "desktop":
            raise RuntimeError("the desktop fell over")
        return {"analysed": True, "device_label": "laptop", "fact_count": 1, "trait_count": 2}

    monkeypatch.setattr(sweeper_module, "analyse_machine", analyse_machine)
    outcomes = asyncio.run(
        sweeper_module.analyse_account(
            store, context=_context(), user_id=USER_ID, assistant_id=ASSISTANT_ID
        )
    )
    assert [outcome.get("analysed") for outcome in outcomes] == [False, True]
    assert "fell over" in outcomes[0]["reason"]
