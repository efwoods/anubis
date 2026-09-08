"""Chart specifications, rendering, and the per-turn collector."""

from __future__ import annotations

import asyncio

import pytest

from src.anubis.utils.analytics.charts import (
    ChartSpec,
    TurnChartCollector,
    chart_artifact_name,
    render_chart_png,
)


def _spec(**overrides):
    base = {
        "type": "line",
        "title": "Messages per day",
        "x": {"label": "day", "values": ["2026-09-01", "2026-09-02", "2026-09-03"]},
        "series": [{"name": "messages", "values": [3, None, 5], "unit": "count"}],
    }
    base.update(overrides)
    return base


def test_spec_validates_shape_and_generates_an_id():
    spec = ChartSpec.model_validate(_spec())
    assert len(spec.chart_id) == 12
    assert chart_artifact_name(spec) == f"chart_{spec.chart_id}.png"


def test_spec_rejects_mismatched_lengths_and_bad_pies():
    with pytest.raises(ValueError, match="values but the x axis has"):
        ChartSpec.model_validate(_spec(series=[{"name": "s", "values": [1]}]))
    with pytest.raises(ValueError, match="exactly one series"):
        ChartSpec.model_validate(
            _spec(type="pie", series=[{"name": "a", "values": [1, 2, 3]}, {"name": "b", "values": [1, 2, 3]}])
        )
    with pytest.raises(ValueError, match="negative"):
        ChartSpec.model_validate(_spec(type="pie", series=[{"name": "a", "values": [1, -2, 3]}]))
    with pytest.raises(ValueError, match="finite"):
        ChartSpec.model_validate(_spec(series=[{"name": "a", "values": [1.0, float("nan"), 3.0]}]))


@pytest.mark.parametrize("chart_type", ["line", "bar", "area", "stacked_bar", "pie"])
def test_render_every_type_to_png(chart_type):
    series = [{"name": "a", "values": [1, 2, 3]}]
    if chart_type != "pie":
        series.append({"name": "b", "values": [2, 1, 0]})
    spec = ChartSpec.model_validate(_spec(type=chart_type, series=series))
    png_bytes = render_chart_png(spec)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 1000


def test_turn_collector_is_scoped_to_a_turn():
    TurnChartCollector.end_turn()
    TurnChartCollector.add({"chart_id": "ignored"}, "ignored.png")
    assert TurnChartCollector.collect() == []
    assert TurnChartCollector.active() is False

    TurnChartCollector.begin_turn()
    assert TurnChartCollector.active() is True
    TurnChartCollector.add({"chart_id": "one", "type": "line"}, "chart_one.png")
    collected = TurnChartCollector.collect()
    assert collected == [{"chart_id": "one", "type": "line", "png_artifact_name": "chart_one.png"}]
    collected[0]["chart_id"] = "mutated"
    assert TurnChartCollector.collect()[0]["chart_id"] == "one"
    TurnChartCollector.end_turn()
    assert TurnChartCollector.collect() == []


def test_turn_collector_is_visible_from_child_tasks():
    async def scenario():
        TurnChartCollector.begin_turn()

        async def child():
            TurnChartCollector.add({"chart_id": "child"}, "child.png")

        await asyncio.create_task(child())
        charts = TurnChartCollector.collect()
        TurnChartCollector.end_turn()
        return charts

    charts = asyncio.run(scenario())
    assert [chart["chart_id"] for chart in charts] == ["child"]
