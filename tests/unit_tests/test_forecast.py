"""The linear-trend forecaster used for burn rate, revenue, and user projections."""

from __future__ import annotations

from src.anubis.utils.analytics.forecast import (
    METHOD_LINEAR_TREND,
    METHOD_LINEAR_TREND_SEASONAL,
    forecast_series,
)


def test_too_few_points_returns_an_error():
    result = forecast_series([1.0, 2.0], 3)
    assert "error" in result
    assert "at least 3" in result["error"]


def test_linear_series_projects_exactly():
    result = forecast_series([10, 20, 30, 40], 3)
    assert result["method"] == METHOD_LINEAR_TREND
    assert result["point"] == [50.0, 60.0, 70.0]
    assert result["lower"] == result["point"]
    assert result["upper"] == result["point"]
    assert result["slope_per_step"] == 10.0
    assert result["observations"] == 4
    assert result["horizon"] == 3


def test_noisy_series_has_a_widening_band_and_ignores_junk():
    result = forecast_series([1, 3, 2, 4, 3, 5, "x", None, float("nan")], 3)
    assert result["observations"] == 6
    widths = [upper - lower for lower, upper in zip(result["lower"], result["upper"])]
    assert widths[0] > 0
    assert widths[0] < widths[1] < widths[2]


def test_seasonal_adjustment_needs_two_full_seasons():
    weekly = [10, 12, 11, 13, 20, 22, 21] * 2
    seasonal = forecast_series(weekly, 7, season_length=7)
    assert seasonal["method"] == METHOD_LINEAR_TREND_SEASONAL
    assert len(seasonal["point"]) == 7
    too_short = forecast_series(weekly[:10], 3, season_length=7)
    assert too_short["method"] == METHOD_LINEAR_TREND


def test_horizon_is_at_least_one():
    result = forecast_series([1, 2, 3], 0)
    assert result["horizon"] == 1
    assert len(result["point"]) == 1
