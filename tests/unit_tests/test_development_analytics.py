"""Unit tests for development analytics and the development tool wrappers.

Everything runs on synthetic rows: no git, no daemon, no model. The labeling
test injects a fake model whose structured runnable returns a canned
``FeatureLabeling``; the tool tests monkeypatch the single Model Context
Protocol call so the fan-out and merge logic is exercised without a network.
"""

import asyncio
from datetime import datetime, timezone

from src.anubis.utils.analytics.development import (
    STRUCTURED_OUTPUT_STREAM_TAG,
    FeatureLabel,
    FeatureLabeling,
    build_labeling_listing,
    feature_effort,
    forecast_feature_time,
    label_features,
    match_sessions_to_commits,
    merge_intervals,
    parse_moment,
    session_hours,
    sprint_summary,
    upcoming_from_plans,
    work_in_progress,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import STRUCTURED_OUTPUT_STREAM_TAG as MODEL_STREAM_TAG
from src.anubis.utils.tools.data_analysis import McpConnection
from src.anubis.utils.tools.data_analysis.development_tools import (
    DEVELOPMENT_TOOL_NAMES,
    build_development_tools,
)

REPOSITORY = "/home/evan/gh/example"
OTHER_REPOSITORY = "/home/evan/gh/other"

COMMITS = [
    {
        "sha": "a" * 40,
        "author": "Evan",
        "authored_at": "2026-09-01T11:00:00+00:00",
        "subject": "Add login page",
        "insertions": 120,
        "deletions": 10,
        "files_changed": 3,
        "repository": REPOSITORY,
    },
    {
        "sha": "b" * 40,
        "author": "Evan",
        "authored_at": "2026-09-01T13:30:00+00:00",
        "subject": "Style login page",
        "insertions": 40,
        "deletions": 5,
        "files_changed": 2,
        "repository": REPOSITORY,
    },
    {
        "sha": "c" * 40,
        "author": "Evan",
        "authored_at": "2026-09-03T15:00:00+00:00",
        "subject": "Fix flaky test",
        "insertions": 4,
        "deletions": 4,
        "files_changed": 1,
        "repository": REPOSITORY,
    },
    {
        "sha": "d" * 40,
        "author": "Pat",
        "authored_at": "2026-09-01T11:30:00+00:00",
        "subject": "Unrelated commit in another repository",
        "insertions": 1,
        "deletions": 1,
        "files_changed": 1,
        "repository": OTHER_REPOSITORY,
    },
]

SESSIONS = [
    {
        "session_id": "s-login",
        "cwd": f"{REPOSITORY}/src",
        "git_branch": "main",
        "started_at": "2026-09-01T10:00:00+00:00",
        "ended_at": "2026-09-01T12:00:00+00:00",
        "duration_minutes": 120.0,
        "first_prompt": "Add a login page",
    },
    {
        "session_id": "s-login-overlap",
        "cwd": REPOSITORY,
        "git_branch": "main",
        "started_at": "2026-09-01T11:00:00+00:00",
        "ended_at": "2026-09-01T13:00:00+00:00",
        "duration_minutes": 120.0,
        "first_prompt": "Polish the login page styling",
    },
    {
        "session_id": "s-tests",
        "cwd": REPOSITORY,
        "git_branch": "feature/tests",
        "started_at": "2026-09-03T14:00:00+00:00",
        "ended_at": "2026-09-03T14:30:00+00:00",
        "duration_minutes": 30.0,
        "first_prompt": "Fix the failing unit test",
    },
    {
        "session_id": "s-elsewhere",
        "cwd": "/home/evan/notes",
        "git_branch": None,
        "started_at": "2026-09-02T09:00:00+00:00",
        "ended_at": "2026-09-02T09:15:00+00:00",
        "duration_minutes": 15.0,
        "first_prompt": "Write meeting notes",
    },
]


def test_parse_moment_accepts_zulu_and_naive_timestamps():
    assert parse_moment("2026-09-01T10:00:00Z") == datetime(
        2026, 9, 1, 10, tzinfo=timezone.utc
    )
    assert parse_moment("2026-09-01") == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert parse_moment("not a date") is None
    assert parse_moment(None) is None


def test_match_sessions_to_commits_uses_cwd_and_time_window():
    result = match_sessions_to_commits(
        SESSIONS, COMMITS, [REPOSITORY, {"path": OTHER_REPOSITORY}]
    )
    by_session = {match["session_id"]: match for match in result["matches"]}
    # The session in /home/evan/notes lies in no repository and is omitted.
    assert set(by_session) == {"s-login", "s-login-overlap", "s-tests"}
    assert by_session["s-login"]["repository"] == REPOSITORY
    # 11:00 lands inside the session; 13:30 lands inside the two-hour lag.
    assert by_session["s-login"]["commit_shas"] == ["a" * 40, "b" * 40]
    assert by_session["s-login-overlap"]["commit_shas"] == ["a" * 40, "b" * 40]
    # 15:00 is within two hours of the 14:30 session end.
    assert by_session["s-tests"]["commit_shas"] == ["c" * 40]
    # The other repository's commit never matches a session in this repository.
    assert result["unmatched_commit_shas"] == ["d" * 40]
    assert result["sessions_by_commit"]["a" * 40] == ["s-login", "s-login-overlap"]


def test_merge_intervals_and_session_hours_count_overlap_once():
    first = (
        datetime(2026, 9, 1, 10, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
    )
    second = (
        datetime(2026, 9, 1, 11, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 13, tzinfo=timezone.utc),
    )
    third = (
        datetime(2026, 9, 1, 14, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 15, tzinfo=timezone.utc),
    )
    assert merge_intervals([third, second, first]) == [(first[0], second[1]), third]
    assert session_hours(SESSIONS[:2]) == 3.0
    assert session_hours(SESSIONS) == 3.75


class _FakeStructuredRunnable:
    """Stands in for ``model.with_structured_output(...)``."""

    def __init__(self, labeling, calls):
        self._labeling = labeling
        self._calls = calls
        self.tags = []

    def with_config(self, **config):
        self.tags = list(config.get("tags") or [])
        return self

    async def ainvoke(self, messages):
        self._calls.append(messages)
        return self._labeling


class _FakeModel:
    """Chat-model stand-in exposing only ``with_structured_output``."""

    def __init__(self, labeling):
        self.calls = []
        self.schema = None
        self.runnable = _FakeStructuredRunnable(labeling, self.calls)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.runnable


LABELING = FeatureLabeling(
    features=[
        FeatureLabel(
            feature="Login page",
            commit_shas=["a" * 12, "b" * 40],
            session_ids=["s-login", "s-login-overlap"],
            status="done",
        ),
        FeatureLabel(
            feature="Test stability",
            commit_shas=["c" * 40],
            session_ids=["s-tests"],
            status="in_progress",
        ),
        FeatureLabel(
            feature="Billing", commit_shas=[], session_ids=[], status="planned"
        ),
    ]
)


def test_label_features_makes_one_structured_call_with_the_stream_tag():
    fake_model = _FakeModel(LABELING)
    plan_items = [{"text": "Billing metering"}, "Rate limits"]

    result = asyncio.run(
        label_features(GlobalContext(), COMMITS, SESSIONS, plan_items, model=fake_model)
    )

    assert result is LABELING
    assert fake_model.schema is FeatureLabeling
    assert len(fake_model.calls) == 1
    assert fake_model.runnable.tags == [STRUCTURED_OUTPUT_STREAM_TAG]
    # The local constant must stay equal to the graph's tag value.
    assert STRUCTURED_OUTPUT_STREAM_TAG == MODEL_STREAM_TAG
    system_message, user_message = fake_model.calls[0]
    assert system_message["role"] == "system"
    assert "commit aaaaaaaaaaaa" in user_message["content"]
    assert "session s-login" in user_message["content"]
    assert "plan Billing metering" in user_message["content"]
    assert "plan Rate limits" in user_message["content"]


def test_labeling_listing_is_capped():
    many_commits = [
        {"sha": f"{index:040d}", "authored_at": "2026-09-01", "subject": f"c{index}"}
        for index in range(250)
    ]
    listing = build_labeling_listing(many_commits, SESSIONS, ["plan"], limit=200)
    assert "Commits (200 of 250):" in listing
    assert "session s-login" not in listing
    assert listing.count("\ncommit ") == 200


def test_feature_effort_merges_hours_and_sums_lines():
    rows = {row["feature"]: row for row in feature_effort(LABELING, SESSIONS, COMMITS)}
    login = rows["Login page"]
    assert login["status"] == "done"
    assert login["session_count"] == 2
    assert login["hours"] == 3.0
    assert login["commit_count"] == 2
    assert login["insertions"] == 160
    assert login["deletions"] == 15
    assert login["first_activity_at"] == "2026-09-01T10:00:00+00:00"
    assert login["last_activity_at"] == "2026-09-01T13:30:00+00:00"
    assert login["calendar_days"] == round(3.5 / 24, 2)
    billing = rows["Billing"]
    assert billing["hours"] == 0.0
    assert billing["commit_count"] == 0
    assert billing["first_activity_at"] is None


def test_sprint_summary_filters_the_window():
    summary = sprint_summary(COMMITS, SESSIONS, "2026-09-01", "2026-09-02T23:59:59Z")
    assert summary["commit_count"] == 3
    assert summary["insertions"] == 161
    assert summary["deletions"] == 16
    assert summary["files_changed"] == 6
    assert summary["session_count"] == 3
    assert summary["session_hours"] == 3.25
    assert summary["active_days"] == 2
    assert summary["by_repository"][REPOSITORY]["commits"] == 2
    assert summary["by_repository"][OTHER_REPOSITORY]["commits"] == 1
    assert summary["by_author"] == {"Evan": 2, "Pat": 1}
    assert summary["highlights"][0] == "Style login page"

    unbounded = sprint_summary(COMMITS, SESSIONS, None, None)
    assert unbounded["commit_count"] == 4
    assert unbounded["session_count"] == 4


def test_work_in_progress_reports_dirty_repositories_and_recent_sessions():
    status_rows = [
        {
            "repository": REPOSITORY,
            "device_label": "Ubuntu",
            "branch": "main",
            "upstream": "origin/main",
            "ahead": 2,
            "behind": 0,
            "staged": ["a.py"],
            "modified": ["a.py", "b.py"],
            "untracked": ["c.py"],
        },
        {
            "repository": OTHER_REPOSITORY,
            "device_label": "Ubuntu",
            "branch": "main",
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "staged": [],
            "modified": [],
            "untracked": [],
        },
    ]
    # Cutoff is 2026-09-02T00:00: the two login sessions ended on 1 September
    # and drop out; the test-fixing and note-writing sessions stay.
    result = work_in_progress(
        status_rows, SESSIONS, active_days=3, now="2026-09-05T00:00:00+00:00"
    )
    assert [entry["repository"] for entry in result["dirty_repositories"]] == [
        REPOSITORY
    ]
    dirty = result["dirty_repositories"][0]
    assert dirty["staged_count"] == 1
    assert dirty["modified_count"] == 2
    assert dirty["untracked_count"] == 1
    assert dirty["paths"] == ["a.py", "b.py", "c.py"]
    assert [entry["session_id"] for entry in result["recent_sessions"]] == [
        "s-tests",
        "s-elsewhere",
    ]
    assert result["as_of"] == "2026-09-05T00:00:00+00:00"


def test_upcoming_from_plans_parses_checkbox_todo_and_planned_markers():
    documents = [
        {
            "path": "features/roadmap.md",
            "text": "\n".join(
                [
                    "# Roadmap",
                    "- [x] Login page",
                    "- [ ] Billing metering",
                    "TODO: rate limits",
                    "- **Phase 11 — Stripe metering** 🔲 planned — usage records",
                    "plain prose line",
                ]
            ),
        }
    ]
    items = upcoming_from_plans(documents)
    assert [(item["marker"], item["line_number"]) for item in items] == [
        ("checkbox", 3),
        ("todo", 4),
        ("planned", 5),
    ]
    assert items[0]["text"] == "Billing metering"
    assert items[1]["text"] == "rate limits"
    assert items[2]["text"] == "planned — usage records"
    assert all(item["path"] == "features/roadmap.md" for item in items)


def test_forecast_feature_time_uses_the_forecast_band_or_the_mean():
    history = [{"hours": 4.0}, {"hours": 6.0}, {"hours": 8.0}, {"hours": 10.0}]
    result = forecast_feature_time(history, ["Billing", {"text": "Rate limits"}])
    assert result["history_points"] == 4
    assert result["method"] == "linear_trend"
    assert [row["feature"] for row in result["planned"]] == ["Billing", "Rate limits"]
    assert result["planned"][0]["point_hours"] == 12.0
    assert result["planned"][1]["point_hours"] == 14.0
    assert result["planned"][0]["lower_hours"] <= result["planned"][0]["point_hours"]
    assert result["planned"][0]["upper_hours"] >= result["planned"][0]["point_hours"]
    assert result["total_hours"] == 26.0

    sparse = forecast_feature_time([3.0, 5.0], ["Only"])
    assert sparse["method"] == "mean"
    assert sparse["planned"][0]["point_hours"] == 4.0
    assert sparse["planned"][0]["lower_hours"] is None

    assert forecast_feature_time(history, [])["planned"] == []


_UBUNTU = McpConnection(
    url="http://localhost:8000/mcp",
    transport="streamable_http",
    server_name="Ubuntu-OS-Filesystem",
    allowed_roots=("/data",),
    device_id="d-ubuntu",
    device_label="Ubuntu",
    platform="ubuntu",
)
_MACOS = McpConnection(
    url="http://localhost:8000/mcp/relay/d-macos",
    transport="streamable_http",
    server_name="macOS-Filesystem",
    allowed_roots=("/Users/evan/data",),
    device_id="d-macos",
    device_label="macOS",
    platform="macos",
)


def test_development_tool_set_names():
    tools = build_development_tools(GlobalContext(), [_UBUNTU])
    assert tuple(tool.name for tool in tools) == DEVELOPMENT_TOOL_NAMES
    for built_tool in tools:
        assert "device_label" in built_tool.args


def test_git_log_fans_out_merges_rows_and_reports_unreached(monkeypatch):
    async def run():
        import src.anubis.utils.tools.data_analysis.analysis_tools as analysis_tools_module

        async def _fake_call(connection, tool_name, tool_args):
            assert tool_name == "git_log"
            assert tool_args["repository"] == REPOSITORY
            assert tool_args["since"] == "2026-09-01"
            if connection.device_id == "d-ubuntu":
                return [COMMITS[0], COMMITS[2]]
            raise RuntimeError("Not a git repository: /home/evan/gh/example")

        monkeypatch.setattr(
            analysis_tools_module, "call_mcp_filesystem_tool", _fake_call
        )
        tools = {
            built_tool.name: built_tool
            for built_tool in build_development_tools(
                GlobalContext(), [_UBUNTU, _MACOS]
            )
        }
        result = await tools["git_log"].coroutine(
            repository=REPOSITORY, since="2026-09-01"
        )
        # Rows are merged newest first and each names the machine.
        assert [row["sha"] for row in result["rows"]] == ["c" * 40, "a" * 40]
        assert {row["device_label"] for row in result["rows"]} == {"Ubuntu"}
        assert result["unreached"] == [
            {
                "device_label": "macOS",
                "detail": "Not a git repository: /home/evan/gh/example",
            }
        ]

        # Naming one machine sends the call only there.
        only_ubuntu = await tools["git_log"].coroutine(
            repository=REPOSITORY, since="2026-09-01", device_label="ubuntu"
        )
        assert only_ubuntu["unreached"] == []
        assert len(only_ubuntu["rows"]) == 2

        unknown = await tools["git_log"].coroutine(
            repository=REPOSITORY, since="2026-09-01", device_label="Windows"
        )
        assert "No connected machine is named 'Windows'" in unknown["error"]

    asyncio.run(run())


def test_git_status_and_sessions_wrap_mapping_results_as_rows(monkeypatch):
    async def run():
        import src.anubis.utils.tools.data_analysis.analysis_tools as analysis_tools_module

        async def _fake_call(connection, tool_name, tool_args):
            if tool_name == "git_status":
                return {
                    "repository": REPOSITORY,
                    "branch": "main",
                    "modified": ["a.py"],
                }
            if tool_name == "list_claude_code_sessions":
                if connection.device_id == "d-macos":
                    return [{"disabled": True, "detail": "switched off"}]
                return [SESSIONS[2], SESSIONS[0]]
            raise AssertionError(tool_name)

        monkeypatch.setattr(
            analysis_tools_module, "call_mcp_filesystem_tool", _fake_call
        )
        tools = {
            built_tool.name: built_tool
            for built_tool in build_development_tools(
                GlobalContext(), [_UBUNTU, _MACOS]
            )
        }
        status = await tools["git_status"].coroutine(repository=REPOSITORY)
        assert [row["device_label"] for row in status["rows"]] == ["Ubuntu", "macOS"]
        assert status["rows"][0]["modified"] == ["a.py"]

        sessions = await tools["list_claude_code_sessions"].coroutine(
            since="2026-09-01"
        )
        session_rows = sessions["rows"]
        # Dated rows come first, newest first; the disabled marker row goes last.
        assert [row.get("session_id") for row in session_rows] == [
            "s-tests",
            "s-login",
            None,
        ]
        assert session_rows[-1] == {
            "device_label": "macOS",
            "disabled": True,
            "detail": "switched off",
        }

    asyncio.run(run())
