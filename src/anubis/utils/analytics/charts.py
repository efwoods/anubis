"""Chart specifications, rendering, and the per-turn chart collector.

The avatar never draws a chart by hand. Every chart passes through one
validated ``ChartSpec`` (below), which is rendered once to a PNG by
``render_chart_png`` and persisted as a created artifact; the same
specification is stored on the saved report so the browser can redraw the
chart interactively and the conversation partner can read the exact numbers.

``TurnChartCollector`` remembers every chart made during one graph turn. The
graph calls ``begin_turn`` before the agent runs and ``collect`` afterwards, so
the charts reach the client with the reply and ``save_report`` can attach the
specifications the model made in that same turn.
"""

from __future__ import annotations

import contextvars
import io
import math
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

ChartType = Literal["line", "bar", "area", "pie", "stacked_bar"]

CHART_WIDTH_PIXELS = 1200
CHART_HEIGHT_PIXELS = 675
CHART_DOTS_PER_INCH = 100

CHART_BACKGROUND_COLOUR = "#0b0b0d"
CHART_TEXT_COLOUR = "#e6e6ea"
CHART_GRID_COLOUR = "#2a2a30"
CHART_PALETTE = (
    "#4f8cff",
    "#ff7a59",
    "#3ddc97",
    "#f6c945",
    "#c084fc",
    "#22d3ee",
    "#fb7185",
    "#a3e635",
)


def new_chart_id() -> str:
    """Return a fresh twelve-character hexadecimal chart id."""
    return uuid4().hex[:12]


class ChartAxis(BaseModel):
    """The category (x) axis: one label per position."""

    label: str = ""
    values: list[str | float | int] = Field(default_factory=list)

    @field_validator("values")
    @classmethod
    def _values_finite(cls, values: list[Any]) -> list[Any]:
        """Reject non-finite numbers on the axis."""
        for value in values:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("axis values must be finite numbers or strings")
        return values


class ChartSeries(BaseModel):
    """One plotted series; ``None`` marks a missing point."""

    name: str
    values: list[float | None] = Field(default_factory=list)
    unit: str | None = None

    @field_validator("values")
    @classmethod
    def _values_finite(cls, values: list[float | None]) -> list[float | None]:
        """Reject NaN and infinities; ``None`` is allowed as a gap."""
        for value in values:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("series values must be finite numbers or null")
        return values


class ChartSpec(BaseModel):
    """A complete chart the renderer and the browser can both draw."""

    chart_id: str = Field(default_factory=new_chart_id)
    type: ChartType
    title: str
    x: ChartAxis
    series: list[ChartSeries] = Field(min_length=1)
    unit: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _shape_is_consistent(self) -> ChartSpec:
        """Every series matches the axis length; a pie holds exactly one series."""
        axis_length = len(self.x.values)
        for series in self.series:
            if len(series.values) != axis_length:
                raise ValueError(
                    f"series '{series.name}' has {len(series.values)} values but the "
                    f"x axis has {axis_length}"
                )
        if self.type == "pie":
            if len(self.series) != 1:
                raise ValueError("a pie chart holds exactly one series")
            for value in self.series[0].values:
                if value is not None and value < 0:
                    raise ValueError("pie slices cannot be negative")
        return self


def chart_artifact_name(spec: ChartSpec) -> str:
    """Return the persisted PNG name for one chart: ``chart_<chart_id>.png``."""
    return f"chart_{spec.chart_id}.png"


def _style_axes(axes: Any) -> None:
    """Apply the dark theme to one axes object."""
    axes.set_facecolor(CHART_BACKGROUND_COLOUR)
    for spine in axes.spines.values():
        spine.set_color(CHART_GRID_COLOUR)
    axes.tick_params(colors=CHART_TEXT_COLOUR, labelsize=10)
    axes.xaxis.label.set_color(CHART_TEXT_COLOUR)
    axes.yaxis.label.set_color(CHART_TEXT_COLOUR)
    axes.title.set_color(CHART_TEXT_COLOUR)
    axes.grid(True, color=CHART_GRID_COLOUR, linewidth=0.6, alpha=0.8)
    axes.set_axisbelow(True)


def render_chart_png(spec: ChartSpec) -> bytes:
    """Render ``spec`` to a 1200 by 675 pixel PNG on a dark background.

    matplotlib is imported here, not at module scope, so the analytics package
    stays cheap to import on the request path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    import numpy

    figure, axes = pyplot.subplots(
        figsize=(
            CHART_WIDTH_PIXELS / CHART_DOTS_PER_INCH,
            CHART_HEIGHT_PIXELS / CHART_DOTS_PER_INCH,
        ),
        dpi=CHART_DOTS_PER_INCH,
    )
    figure.patch.set_facecolor(CHART_BACKGROUND_COLOUR)
    try:
        labels = [str(value) for value in spec.x.values]
        positions = numpy.arange(len(labels))
        colours = [
            CHART_PALETTE[index % len(CHART_PALETTE)]
            for index in range(len(spec.series))
        ]

        if spec.type == "pie":
            series = spec.series[0]
            slice_values = [float(value or 0.0) for value in series.values]
            slice_colours = [
                CHART_PALETTE[index % len(CHART_PALETTE)]
                for index in range(len(slice_values))
            ]
            axes.set_facecolor(CHART_BACKGROUND_COLOUR)
            if sum(slice_values) > 0:
                wedges, texts, autotexts = axes.pie(
                    slice_values,
                    labels=labels,
                    colors=slice_colours,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"edgecolor": CHART_BACKGROUND_COLOUR},
                )
                for text in [*texts, *autotexts]:
                    text.set_color(CHART_TEXT_COLOUR)
            axes.axis("equal")
        else:
            _style_axes(axes)
            if spec.type == "line":
                for series, colour in zip(spec.series, colours):
                    values = [
                        numpy.nan if value is None else float(value)
                        for value in series.values
                    ]
                    axes.plot(
                        positions, values, color=colour, linewidth=2.2,
                        marker="o", markersize=3.5, label=series.name,
                    )
            elif spec.type == "area":
                for series, colour in zip(spec.series, colours):
                    values = [
                        0.0 if value is None else float(value)
                        for value in series.values
                    ]
                    axes.fill_between(
                        positions, values, color=colour, alpha=0.35, label=series.name
                    )
                    axes.plot(positions, values, color=colour, linewidth=1.8)
            elif spec.type == "bar":
                group_count = len(spec.series)
                bar_width = 0.8 / max(group_count, 1)
                for index, (series, colour) in enumerate(zip(spec.series, colours)):
                    values = [
                        0.0 if value is None else float(value)
                        for value in series.values
                    ]
                    offsets = positions + (index - (group_count - 1) / 2) * bar_width
                    axes.bar(
                        offsets, values, width=bar_width, color=colour, label=series.name
                    )
            elif spec.type == "stacked_bar":
                bottoms = numpy.zeros(len(labels))
                for series, colour in zip(spec.series, colours):
                    values = numpy.array(
                        [0.0 if value is None else float(value) for value in series.values]
                    )
                    axes.bar(
                        positions, values, width=0.7, bottom=bottoms,
                        color=colour, label=series.name,
                    )
                    bottoms = bottoms + values
            axes.set_xticks(positions)
            rotation = 45 if len(labels) > 8 else 0
            axes.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
            if spec.x.label:
                axes.set_xlabel(spec.x.label)
            unit = spec.unit or next(
                (series.unit for series in spec.series if series.unit), None
            )
            if unit:
                axes.set_ylabel(unit)
            if len(spec.series) > 1:
                legend = axes.legend(
                    facecolor=CHART_BACKGROUND_COLOUR,
                    edgecolor=CHART_GRID_COLOUR,
                    labelcolor=CHART_TEXT_COLOUR,
                )
                legend.get_frame().set_alpha(0.9)

        axes.set_title(spec.title, color=CHART_TEXT_COLOUR, fontsize=15, pad=14)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=CHART_DOTS_PER_INCH,
            facecolor=CHART_BACKGROUND_COLOUR,
        )
        return buffer.getvalue()
    finally:
        pyplot.close(figure)


_turn_charts: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("analytics_turn_charts", default=None)
)


class TurnChartCollector:
    """Per-turn list of charts, carried on a context variable.

    ``begin_turn`` installs a fresh list for the current task; child tasks the
    agent spawns inherit the same list object, so charts made inside tool
    calls are visible to the graph node that started the turn.
    """

    @staticmethod
    def begin_turn() -> None:
        """Start collecting charts for the current turn."""
        _turn_charts.set([])

    @staticmethod
    def add(spec_dict: dict[str, Any], png_artifact_name: str) -> None:
        """Record one chart made during this turn (no-op outside a turn)."""
        charts = _turn_charts.get()
        if charts is None:
            return
        charts.append({**dict(spec_dict), "png_artifact_name": png_artifact_name})

    @staticmethod
    def collect() -> list[dict[str, Any]]:
        """Return the charts made so far in this turn (a copy)."""
        charts = _turn_charts.get()
        return [dict(chart) for chart in (charts or [])]

    @staticmethod
    def end_turn() -> None:
        """Stop collecting; later ``add`` calls are ignored."""
        _turn_charts.set(None)

    @staticmethod
    def active() -> bool:
        """Whether a turn is currently collecting."""
        return _turn_charts.get() is not None
