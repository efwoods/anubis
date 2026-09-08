"""The analytics chat tools: charts, reports, schedules, gating, and unavailability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.analytics.analytics_tools import (
    ANALYTICS_TOOL_NAMES,
    build_analytics_tools,
    parse_iso_datetime,
    period_from,
)
from src.anubis.utils.analytics.charts import TurnChartCollector
from src.anubis.utils.analytics.reports import (
    InMemoryReportRepository,
    set_report_repository,
)
from src.anubis.utils.analytics.schedules import (
    InMemoryScheduleRepository,
    set_schedule_repository,
)


class _FakeStore:
    def __init__(self):
        self.values = {}

    async def aput(self, namespace, key, value):
        self.values[(namespace, key)] = value

    async def asearch(self, namespace, *, limit=10, **kwargs):
        return []


def _context(admin_user_id="admin"):
    return SimpleNamespace(
        admin_user_id=admin_user_id,
        finance_sync_min_interval_minutes=360,
        plaid_client_id="c",
        plaid_secret="s",
        plaid_environment="sandbox",
    )


def _tools(**overrides):
    arguments = {
        "store": _FakeStore(),
        "pool": None,
        "user_id": "owner",
        "assistant_id": "avatar",
        "connected_accounts": [],
        "thread_id": "thread-1",
    }
    arguments.update(overrides)
    context = overrides.pop("context", None) or _context()
    tools = build_analytics_tools(context, **{k: v for k, v in arguments.items() if k != "context"})
    return {tool.name: tool for tool in tools}


@pytest.fixture(autouse=True)
def _reset_repositories():
    TurnChartCollector.end_turn()
    yield
    set_report_repository(None)
    set_schedule_repository(None)
    TurnChartCollector.end_turn()


def test_every_expected_tool_is_built():
    tools = _tools()
    assert set(tools) == set(ANALYTICS_TOOL_NAMES)
    for tool in tools.values():
        assert tool.description.strip()


def test_period_defaults_to_thirty_days():
    start, end = period_from(None, None)
    assert (end - start).days == 30
    assert parse_iso_datetime("2026-09-01").tzinfo is not None
    assert parse_iso_datetime("2026-09-01T10:00:00Z").hour == 10


@pytest.mark.asyncio
async def test_make_chart_persists_to_the_store_and_collects_the_spec():
    store = _FakeStore()
    tools = _tools(store=store)
    TurnChartCollector.begin_turn()
    result = await tools["make_chart"].ainvoke(
        {
            "spec": {
                "type": "line",
                "title": "Messages per day",
                "x": {"label": "day", "values": ["2026-09-01", "2026-09-02"]},
                "series": [{"name": "messages", "values": [3, 5]}],
            }
        }
    )
    assert result["status"] == "ok"
    assert result["png_artifact_name"] == f"chart_{result['chart_id']}.png"
    stored_keys = list(store.values)
    assert stored_keys == [(("owner", "avatar", "created"), f"/{result['png_artifact_name']}")] or (
        stored_keys[0][1] == f"/{result['png_artifact_name']}"
    )
    stored_value = list(store.values.values())[0]
    assert stored_value["encoding"] == "base64"
    assert stored_value["size_bytes"] > 100
    collected = TurnChartCollector.collect()
    assert len(collected) == 1
    assert collected[0]["chart_id"] == result["chart_id"]
    assert collected[0]["png_artifact_name"] == result["png_artifact_name"]


@pytest.mark.asyncio
async def test_make_chart_rejects_an_invalid_spec():
    tools = _tools()
    result = await tools["make_chart"].ainvoke(
        {"spec": {"type": "pie", "title": "Bad", "x": {"values": ["a"]}, "series": [{"name": "s", "values": [1, 2]}]}}
    )
    assert result["status"] == "error"
    assert "values" in result["message"]


@pytest.mark.asyncio
async def test_make_chart_uses_the_analysis_bundle_workspace(tmp_path):
    persisted = []

    async def fake_persist_workspace_file(bundle, store, candidate_path):
        persisted.append((bundle, candidate_path))
        return {"name": f"{candidate_path.stem}_2026_09_07_09_00_00.png"}

    import src.anubis.utils.tools.data_analysis.analysis_tools as analysis_tools_module

    original = analysis_tools_module.persist_workspace_file
    analysis_tools_module.persist_workspace_file = fake_persist_workspace_file
    try:
        bundle = SimpleNamespace(workspace_path=tmp_path / "workspace", store=None, persisted_artifacts=[])
        tools = _tools(analysis_bundle=bundle)
        TurnChartCollector.begin_turn()
        result = await tools["make_chart"].ainvoke(
            {"spec": {"type": "bar", "title": "T", "x": {"values": ["a"]}, "series": [{"name": "s", "values": [1]}]}}
        )
    finally:
        analysis_tools_module.persist_workspace_file = original
    assert result["png_artifact_name"].endswith("_2026_09_07_09_00_00.png")
    assert persisted[0][1].exists()
    assert TurnChartCollector.collect()[0]["png_artifact_name"] == result["png_artifact_name"]


@pytest.mark.asyncio
async def test_save_report_stores_collected_charts_and_lists_them():
    repository = InMemoryReportRepository()
    set_report_repository(repository)
    tools = _tools()
    TurnChartCollector.begin_turn()
    TurnChartCollector.add({"chart_id": "one", "type": "line"}, "chart_one.png")
    TurnChartCollector.add({"chart_id": "two", "type": "bar"}, "chart_two.png")

    saved = await tools["save_report"].ainvoke(
        {
            "kind": "finance",
            "title": "August spend",
            "summary_markdown": "Burn rate 3k",
            "period_start": "2026-08-01",
            "period_end": "2026-08-31",
            "sources": ["query_finances"],
        }
    )
    assert saved["status"] == "ok"
    assert saved["chart_count"] == 2
    row = repository.rows[saved["report_id"]]
    assert [chart["chart_id"] for chart in row["charts"]] == ["one", "two"]
    assert row["thread_id"] == "thread-1"
    assert row["period_start"].month == 8

    subset = await tools["save_report"].ainvoke(
        {"kind": "custom", "title": "Only two", "summary_markdown": "x", "chart_ids": ["two"]}
    )
    assert subset["chart_count"] == 1

    listed = await tools["list_reports"].ainvoke({"query": "burn"})
    assert [report["title"] for report in listed["reports"]] == ["August spend"]
    assert listed["reports"][0]["charts"][0]["png_artifact_name"] == "chart_one.png"


@pytest.mark.asyncio
async def test_platform_metrics_are_forbidden_for_non_admins():
    tools = _tools(connected_accounts=[])
    result = await tools["query_platform_metrics"].ainvoke({"metric": "messages_per_day"})
    assert result["status"] == "forbidden"

    # The configured administrator passes the gate without any connection;
    # with no pool published the tool then reports the store as unavailable.
    admin_tools = _tools(context=_context(admin_user_id="owner"), connected_accounts=[])
    result = await admin_tools["query_platform_metrics"].ainvoke({"metric": "messages_per_day"})
    assert result["status"] == "unavailable"

    unknown = await tools["query_platform_metrics"].ainvoke({"metric": "nope"})
    assert unknown["status"] == "error"
    assert "messages_per_day" in unknown["metrics"]


@pytest.mark.asyncio
async def test_pool_none_answers_unavailable_for_admins_and_finance_and_vendors():
    admin_tools = _tools(
        context=_context(admin_user_id="owner"),
        connected_accounts=[{"provider": "plaid", "kind": "bank", "display_label": "Chase"}],
    )
    platform = await admin_tools["query_platform_metrics"].ainvoke({"metric": "messages_per_day"})
    assert platform["status"] == "unavailable"
    finance = await admin_tools["query_finances"].ainvoke({"metric": "spend"})
    assert finance["status"] == "unavailable"
    vendors = await admin_tools["query_vendor_usage"].ainvoke({})
    assert vendors["status"] == "unavailable"
    reports = await admin_tools["list_reports"].ainvoke({})
    assert reports["status"] == "unavailable"
    schedules = await admin_tools["list_report_schedules"].ainvoke({})
    assert schedules["status"] == "unavailable"


@pytest.mark.asyncio
async def test_query_finances_without_a_bank_names_the_missing_connection():
    tools = _tools(connected_accounts=[{"provider": "gmail", "kind": "mailbox"}])
    result = await tools["query_finances"].ainvoke({"metric": "spend"})
    assert result["status"] == "unavailable"
    assert result["missing_connection"] == "plaid"
    accounts = await _tools(
        connected_accounts=[{"provider": "plaid", "kind": "bank", "transport": {"accounts": [{"account_id": "a", "name": "Checking"}]}}]
    )["query_finances"].ainvoke({"metric": "accounts"})
    assert accounts["accounts"][0]["name"] == "Checking"


@pytest.mark.asyncio
async def test_schedule_report_cancel_and_list():
    repository = InMemoryScheduleRepository()
    set_schedule_repository(repository)
    tools = _tools(timezone_name="America/Los_Angeles")
    created = await tools["schedule_report"].ainvoke(
        {"kind": "sprint_digest", "title": "Sprint", "question": "What shipped?", "interval": "weekly"}
    )
    assert created["status"] == "ok"
    assert created["interval"] == "weekly"
    assert created["next_run_at"].endswith("+00:00")
    listed = await tools["list_report_schedules"].ainvoke({})
    assert len(listed["schedules"]) == 1
    cancelled = await tools["cancel_report_schedule"].ainvoke({"schedule_id": created["schedule_id"]})
    assert cancelled["status"] == "ok"
    again = await tools["cancel_report_schedule"].ainvoke({"schedule_id": created["schedule_id"]})
    assert again["status"] == "error"
    empty = await tools["schedule_report"].ainvoke({"kind": "custom", "title": "x", "question": "  ", "interval": "daily"})
    assert empty["status"] == "error"


@pytest.mark.asyncio
async def test_forecast_metric_relays_the_method_and_errors():
    tools = _tools()
    projected = await tools["forecast_metric"].ainvoke({"values": [1, 2, 3, 4], "horizon": 2})
    assert projected["status"] == "ok"
    assert projected["method"] == "linear_trend"
    assert len(projected["point"]) == 2
    short = await tools["forecast_metric"].ainvoke({"values": [1], "horizon": 2})
    assert short["status"] == "error"
