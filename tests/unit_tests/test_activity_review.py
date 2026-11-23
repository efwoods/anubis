"""The activity review: "what did I do today / this week", read from data already stored.

The checkpoint read, the Postgres thread lookup, and the machine calls are
faked at their seams, so the period arithmetic, the screen timeline, and the
rule that one failing source never fails the review are exercised without a
database, a daemon, or a model.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.activity import sources as sources_module
from src.anubis.utils.activity import thread_reader
from src.anubis.utils.activity.periods import resolve_activity_period
from src.anubis.utils.activity.screen_timeline import (
    count_observed_seconds,
    screen_observations_in_period,
    screen_timeline,
)

NEW_YORK = ZoneInfo("America/New_York")
USER_ID = "owner-user"
ASSISTANT_ID = "personal-avatar"


def context(**overrides):
    """A settings object carrying only the activity review fields."""
    values = {
        "activity_review_maximum_threads": 200,
        "activity_review_maximum_blocks": 120,
        "activity_review_maximum_gap_seconds": 660,
        "activity_review_maximum_description_characters": 240,
        "activity_review_maximum_browser_visits": 5000,
        "device_fanout_timeout_seconds": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def observation(
    captured_at: datetime,
    *,
    kind: str = "writing_code",
    summary: str = "The owner edits Python in VS Code.",
    sources: tuple[str, ...] = ("screen",),
    observation_id: str | None = None,
    body: str = "screen: VS Code, file graph.py, a terminal running pytest.",
) -> HumanMessage:
    """A stored, triaged ambient observation as the thread checkpoint holds the observation."""
    identifier = observation_id or f"observation-{captured_at.isoformat()}"
    header = (
        f"[AMBIENT_OBSERVATION id={identifier} captured_at={captured_at.isoformat()}"
        f" sources={','.join(sources)} decision=ignore]"
    )
    return HumanMessage(
        content=f"{header}\n{body}",
        additional_kwargs={
            "hidden": True,
            "kind": "ambient_observation",
            "ambient": {
                "observation_id": identifier,
                "captured_at": captured_at.astimezone(UTC).isoformat(),
                "sources": list(sources),
                "decision": "ignore",
                "observation_kind": kind,
                "summary": summary,
            },
        },
    )


# --- periods ---------------------------------------------------------------


def test_today_starts_at_the_owners_local_midnight_not_the_utc_midnight():
    now = datetime(2026, 11, 18, 2, 30, tzinfo=UTC)  # 21:30 on 2026-11-17 in New York
    period = resolve_activity_period("today", None, None, "America/New_York", now=now)
    assert period.period_start == datetime(2026, 11, 17, 0, 0, tzinfo=NEW_YORK)
    assert period.period_end == now


def test_yesterday_across_the_end_of_daylight_saving_is_twenty_five_hours():
    now = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)  # Monday after the 2026-11-01 change
    period = resolve_activity_period(
        "yesterday", None, None, "America/New_York", now=now
    )
    assert period.period_start.date().isoformat() == "2026-11-01"
    assert period.period_end.date().isoformat() == "2026-11-02"
    elapsed = period.period_end.astimezone(UTC) - period.period_start.astimezone(UTC)
    assert elapsed == timedelta(hours=25)


def test_this_week_and_last_week_start_on_monday():
    now = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)  # Wednesday
    this_week = resolve_activity_period(
        "this_week", None, None, "America/New_York", now=now
    )
    last_week = resolve_activity_period(
        "last week", None, None, "America/New_York", now=now
    )
    assert this_week.period_start == datetime(2026, 11, 2, 0, 0, tzinfo=NEW_YORK)
    assert last_week.period_start == datetime(2026, 10, 26, 0, 0, tzinfo=NEW_YORK)
    assert last_week.period_end == this_week.period_start


def test_a_bare_until_date_includes_that_whole_day_and_overrides_the_period():
    period = resolve_activity_period(
        "today", "2026-11-10", "2026-11-12", "America/New_York"
    )
    assert period.name == "custom"
    assert period.period_start == datetime(2026, 11, 10, 0, 0, tzinfo=NEW_YORK)
    assert period.period_end == datetime(2026, 11, 13, 0, 0, tzinfo=NEW_YORK)


def test_an_unknown_period_or_zone_is_handled():
    with pytest.raises(ValueError):
        resolve_activity_period("fortnight", None, None, "America/New_York")
    period = resolve_activity_period("today", None, None, "Not/A_Zone")
    assert period.timezone_name == "UTC"


# --- screen timeline -------------------------------------------------------


def test_only_screen_observations_inside_the_period_are_kept_once():
    period_start = datetime(2026, 11, 17, 5, 0, tzinfo=UTC)
    period_end = datetime(2026, 11, 18, 5, 0, tzinfo=UTC)
    inside = datetime(2026, 11, 17, 15, 0, tzinfo=UTC)
    messages_by_thread = {
        "thread-a": [
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            observation(inside, observation_id="same"),
            observation(inside + timedelta(minutes=1), sources=("webcam",)),
            observation(period_start - timedelta(minutes=1)),
        ],
        # The same observation read from a second thread is counted once.
        "thread-b": [observation(inside, observation_id="same")],
    }
    kept = screen_observations_in_period(
        messages_by_thread, period_start, period_end, maximum_description_characters=240
    )
    assert len(kept) == 1
    assert kept[0].screen_description.startswith("VS Code")


def test_a_long_gap_is_not_observed_and_is_never_counted():
    start = datetime(2026, 11, 17, 14, 0, tzinfo=UTC)
    observations = screen_observations_in_period(
        {
            "thread-a": [
                observation(start),
                observation(start + timedelta(seconds=30)),
                observation(start + timedelta(seconds=60), kind="video_call"),
                # Two hours later: the share was off in between.
                observation(start + timedelta(hours=2), kind="video_call"),
            ]
        },
        start - timedelta(hours=1),
        start + timedelta(hours=3),
        maximum_description_characters=240,
    )
    not_observed_seconds = count_observed_seconds(
        observations,
        start + timedelta(hours=2, seconds=100),
        maximum_gap_seconds=660,
    )
    assert [observation.counted_seconds for observation in observations] == [
        30,
        30,
        660,
        100,
    ]
    # 60 s → 7200 s is 7140 s, of which 660 s counts.
    assert not_observed_seconds == 7140 - 660


def test_a_block_lists_seconds_per_label_instead_of_splitting_on_every_label():
    start = datetime(2026, 11, 17, 14, 0, tzinfo=UTC)  # 09:00 in New York
    # Measured shape: one piece of work labelled differently on every snapshot.
    labels = ["organization_page", "youtube_results", "reading_documentation"]
    timeline = screen_timeline(
        {
            "thread-a": [
                observation(start + timedelta(seconds=30 * i), kind=labels[i % 3])
                for i in range(6)
            ]
        },
        start - timedelta(hours=1),
        start + timedelta(seconds=180),
        owner_zone=NEW_YORK,
        maximum_gap_seconds=660,
        maximum_blocks=120,
        maximum_description_characters=240,
    )
    assert timeline["observation_count"] == 6
    assert timeline["observed_seconds"] == 180
    assert timeline["block_minutes"] == 15
    assert len(timeline["blocks"]) == 1
    block = timeline["blocks"][0]
    assert block["started_local"] == "2026-11-17 09:00"
    assert block["seconds_by_activity"] == {
        "organization_page": 60,
        "youtube_results": 60,
        "reading_documentation": 60,
    }
    # One summary, listed once even though six snapshots carried the summary.
    assert block["summaries"] == ["The owner edits Python in VS Code."]
    assert "spacing" in timeline["duration_method"]


def test_a_week_grows_the_block_length_and_keeps_every_second():
    start = datetime(2026, 11, 16, 13, 0, tzinfo=UTC)  # Monday 08:00 in New York
    # Eight hours a day for five days, one snapshot every ten minutes.
    messages = [
        observation(start + timedelta(days=day, minutes=10 * i))
        for day in range(5)
        for i in range(48)
    ]
    timeline = screen_timeline(
        {"thread-a": messages},
        start,
        start + timedelta(days=4, hours=8),
        owner_zone=NEW_YORK,
        maximum_gap_seconds=660,
        maximum_blocks=60,
        maximum_description_characters=240,
    )
    # 15-minute blocks would give 160 non-empty blocks, 30-minute 80, 60-minute 40.
    assert timeline["block_minutes"] == 60
    assert len(timeline["blocks"]) == 40
    assert (
        sum(block["observed_seconds"] for block in timeline["blocks"])
        == (timeline["observed_seconds"])
    )


# --- thread lookup ---------------------------------------------------------


class _FakeCursor:
    def __init__(self, recorder):
        self.recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def execute(self, sql, parameters):
        self.recorder.append((sql, parameters))

    async def fetchall(self):
        return [("thread-a",), ("thread-b",)]


class _FakeConnection:
    def __init__(self, recorder):
        self.recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def cursor(self):
        return _FakeCursor(self.recorder)


class _FakePool:
    def __init__(self):
        self.recorder: list = []

    def connection(self):
        return _FakeConnection(self.recorder)


def test_threads_are_found_by_owner_avatar_and_period():
    pool = _FakePool()
    period_start = datetime(2026, 11, 17, 5, 0, tzinfo=UTC)
    period_end = datetime(2026, 11, 18, 5, 0, tzinfo=UTC)
    thread_ids = asyncio.run(
        thread_reader.threads_touched_in_period(
            pool, USER_ID, ASSISTANT_ID, period_start, period_end, 50
        )
    )
    assert thread_ids == ["thread-a", "thread-b"]
    sql, parameters = pool.recorder[0]
    # ``state_updated_at`` is not refreshed on every run (measured 2026-11-18:
    # a thread with screen observations on 2026-11-17 still carried
    # 2026-11-04), so the filter must take the later of the two stamps.
    assert "GREATEST(updated_at, state_updated_at)" in sql
    assert "COALESCE(state_updated_at" not in sql
    assert USER_ID in parameters["metadata_filter"]
    assert ASSISTANT_ID in parameters["metadata_filter"]
    assert parameters["period_start"] == period_start
    assert parameters["maximum_threads"] == 50


# --- sources ---------------------------------------------------------------


def test_one_failing_source_leaves_the_others_answering(monkeypatch):
    async def failing_screen(*_arguments, **_keywords):
        raise RuntimeError("checkpoint read failed")

    monkeypatch.setattr(sources_module, "screen_section", failing_screen)
    period = resolve_activity_period("today", None, None, "America/New_York")
    review = asyncio.run(
        sources_module.gather_activity(
            context(),
            period,
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            connections=[],
        )
    )
    assert review["sources"]["screen"]["status"] == "error"
    assert "checkpoint read failed" in review["sources"]["screen"]["detail"]
    assert review["sources"]["development"]["status"] == "unavailable"
    assert review["sources"]["browser"]["status"] == "unavailable"
    assert set(review["unavailable_sources"]) == {"screen", "development", "browser"}


def test_the_screen_source_is_unavailable_without_the_api_lifespan(monkeypatch):
    monkeypatch.setattr(thread_reader, "_activity_graph", None)
    period = resolve_activity_period("today", None, None, "UTC")
    section = asyncio.run(
        sources_module.screen_section(
            context(), period, user_id=USER_ID, assistant_id=ASSISTANT_ID
        )
    )
    assert section["status"] == "unavailable"


def test_the_browser_read_never_moves_the_insights_watermark(monkeypatch):
    from src.anubis.utils.browsing import history_client

    now = datetime.now(UTC)
    calls: dict[str, list] = {"read": [], "watermark": []}

    async def fake_read_new_visits(connection, since, *, limit, default_days=30, **_):
        calls["read"].append(since)
        return {
            "status": "ok",
            "visit_count": 2,
            "visits": [
                {
                    "url": "https://github.com/owner/repo/pull/7",
                    "host": "github.com",
                    "title": "Pull request 7",
                    "visited_at": (now - timedelta(minutes=5)).isoformat(),
                },
                {
                    "url": "https://news.example.com/old",
                    "host": "news.example.com",
                    "title": "Old visit",
                    "visited_at": (now - timedelta(days=3)).isoformat(),
                },
            ],
        }

    async def fake_write_watermark(*arguments, **keywords):
        calls["watermark"].append(arguments)

    monkeypatch.setattr(history_client, "read_new_visits", fake_read_new_visits)
    monkeypatch.setattr(history_client, "write_watermark", fake_write_watermark)
    period = resolve_activity_period(
        "custom", (now - timedelta(hours=1)).isoformat(), None, "UTC", now=now
    )
    section = asyncio.run(
        sources_module.browser_section(
            context(), period, [SimpleNamespace(device_label="Ubuntu")]
        )
    )
    assert section["status"] == "ok"
    assert section["visit_count"] == 1
    assert section["top_hosts"][0][0] == "github.com"
    assert calls["read"] == [period.period_start.astimezone(UTC).isoformat()]
    assert calls["watermark"] == []


def test_no_visible_repository_reports_commits_as_unknown_not_zero(monkeypatch):
    from src.anubis.utils.tools.data_analysis import development_tools

    async def fake_fan_out(connections, device_label, tool_name, arguments, *_, **__):
        if tool_name == development_tools.DAEMON_LIST_CLAUDE_CODE_SESSIONS:
            return {
                "rows": [{"device_label": "Ubuntu", "duration_minutes": 30}],
                "unreached": [],
            }
        return {"rows": [], "unreached": []}

    monkeypatch.setattr(development_tools, "fan_out_development_tool", fake_fan_out)
    period = resolve_activity_period("today", None, None, "UTC")
    section = asyncio.run(
        sources_module.development_section(
            context(), period, [SimpleNamespace(device_label="Ubuntu")]
        )
    )
    assert section["status"] == "ok"
    assert section["visible_repository_count"] == 0
    assert "unknown, not zero" in section["commit_detail"]
    assert section["claude_code_session_minutes"] == 30
