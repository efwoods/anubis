"""A small, explainable forecaster for owner metrics.

The owner asks "what does next quarter look like" about series that are short
(weeks or months of daily points) and noisy. A linear trend fitted by least
squares, with an optional seasonal-naive correction and a residual-based
uncertainty band, is honest about that: the method is stated in the result so
the avatar can say how the number was produced rather than presenting a
projection as a fact.
"""

from __future__ import annotations

import math
from typing import Any

METHOD_LINEAR_TREND = "linear_trend"
METHOD_LINEAR_TREND_SEASONAL = "linear_trend_with_seasonal_naive"

MINIMUM_POINTS = 3
CONFIDENCE_MULTIPLIER = 1.96


def forecast_series(
    values: list[float], horizon: int, season_length: int | None = None
) -> dict[str, Any]:
    """Project ``values`` ``horizon`` steps ahead.

    Returns ``{"point", "lower", "upper", "method", "slope_per_step"}``, each
    band a list of ``horizon`` floats. The band is
    ``1.96 * residual standard deviation * sqrt(step)``, widening with the
    distance from the last observation. When ``season_length`` is given and
    the series holds at least two full seasons, the average residual at each
    seasonal position is added back (seasonal naive adjustment). Fewer than
    three points returns ``{"error": ...}``.
    """
    import numpy

    cleaned: list[float] = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            cleaned.append(number)
    if len(cleaned) < MINIMUM_POINTS:
        return {
            "error": (
                f"at least {MINIMUM_POINTS} numeric points are needed to forecast; "
                f"received {len(cleaned)}"
            )
        }
    horizon = max(1, int(horizon))
    observations = numpy.asarray(cleaned, dtype=float)
    steps = numpy.arange(len(observations), dtype=float)
    slope, intercept = numpy.polyfit(steps, observations, 1)
    fitted = slope * steps + intercept
    residuals = observations - fitted

    method = METHOD_LINEAR_TREND
    seasonal_offsets: list[float] = [0.0] * horizon
    if (
        season_length
        and int(season_length) >= 2
        and len(observations) >= 2 * int(season_length)
    ):
        season = int(season_length)
        by_position: dict[int, list[float]] = {}
        for index, residual in enumerate(residuals):
            by_position.setdefault(index % season, []).append(float(residual))
        seasonal_index = {
            position: float(numpy.mean(entries)) for position, entries in by_position.items()
        }
        seasonal_offsets = [
            seasonal_index.get((len(observations) + step) % season, 0.0)
            for step in range(horizon)
        ]
        method = METHOD_LINEAR_TREND_SEASONAL
        residuals = numpy.asarray(
            [
                float(residual) - seasonal_index.get(index % season, 0.0)
                for index, residual in enumerate(residuals)
            ]
        )

    residual_std = float(numpy.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    point: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for step in range(1, horizon + 1):
        future_index = len(observations) - 1 + step
        projected = float(slope * future_index + intercept) + seasonal_offsets[step - 1]
        half_width = CONFIDENCE_MULTIPLIER * residual_std * math.sqrt(step)
        point.append(round(projected, 6))
        lower.append(round(projected - half_width, 6))
        upper.append(round(projected + half_width, 6))
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "method": method,
        "slope_per_step": round(float(slope), 6),
        "observations": len(observations),
        "horizon": horizon,
    }
