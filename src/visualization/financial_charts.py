"""Plotly charts and display formatters for the AnalysisHandoff metrics.

Presentation only: values are rendered exactly as delivered by the API.
``change_pct`` is already in percentage points and is never recomputed here.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

NOT_AVAILABLE = "N/A"
CURRENT_COLOR = "#2a78d6"   # categorical slot 1 (blue)
PREVIOUS_COLOR = "#7b8794"  # neutral comparison series
UNIT_LABELS = {"usd": "USD values", "usdPerShare": "Per-share values"}


def format_value(value: float | None, unit: str | None) -> str:
    """Human-readable value; scales large USD amounts to K/M/B for display only."""

    if value is None:
        return NOT_AVAILABLE
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if unit == "usd":
        for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if magnitude >= threshold:
                return f"{sign}${magnitude / threshold:,.2f}{suffix}"
        return f"{sign}${magnitude:,.2f}"
    if unit == "usdPerShare":
        return f"{sign}${magnitude:,.2f}"
    return f"{value:,.2f} {unit or ''}".strip()


def format_change_pct(change_pct: float | None) -> str:
    """``-1.589`` -> ``-1.59%``. The value is already in percentage points."""

    return NOT_AVAILABLE if change_pct is None else f"{change_pct:+.2f}%"


def _bar(metrics: list[dict], field: str, period_field: str, name: str, color: str,
         show_legend: bool) -> go.Bar:
    return go.Bar(
        name=name,
        orientation="h",
        y=[m["name"] for m in metrics],
        x=[m.get(field) for m in metrics],
        text=[format_value(m.get(field), m.get("unit")) for m in metrics],
        textposition="outside",
        customdata=[
            [m.get(period_field) or NOT_AVAILABLE, m.get("comparison_type") or NOT_AVAILABLE]
            for m in metrics
        ],
        hovertemplate=(
            "<b>%{y}</b><br>" + name + ": %{text}<br>"
            "Period: %{customdata[0]}<br>Comparison: %{customdata[1]}<extra></extra>"
        ),
        marker={"color": color, "cornerradius": 4},
        legendgroup=name,
        showlegend=show_legend,
    )


def render_metrics_comparison_chart(financial_metrics: list[dict]) -> go.Figure:
    """Current vs previous horizontal bars, one panel (one axis) per unit.

    Metrics without any value are skipped; a single missing side renders as
    an absent bar. An empty input yields a figure with a placeholder note.
    """

    plottable = [
        m for m in financial_metrics
        if m.get("current_value") is not None or m.get("previous_value") is not None
    ]
    if not plottable:
        figure = go.Figure()
        figure.add_annotation(
            text="No comparable metrics available",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        figure.update_layout(height=200)
        return figure

    by_unit: dict[str | None, list[dict]] = {}
    for metric in plottable:
        by_unit.setdefault(metric.get("unit"), []).append(metric)

    figure = make_subplots(
        rows=len(by_unit),
        cols=1,
        subplot_titles=[UNIT_LABELS.get(u, u or "Unitless values") for u in by_unit],
        row_heights=[len(group) for group in by_unit.values()],
        vertical_spacing=0.12,
    )
    for row, group in enumerate(by_unit.values(), start=1):
        figure.add_trace(
            _bar(group, "current_value", "current_period", "Current", CURRENT_COLOR, row == 1),
            row=row, col=1,
        )
        figure.add_trace(
            _bar(group, "previous_value", "previous_period", "Previous", PREVIOUS_COLOR, row == 1),
            row=row, col=1,
        )
        figure.update_yaxes(autorange="reversed", row=row, col=1)

    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=True)
    figure.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        height=130 + 55 * len(plottable),
        margin={"l": 10, "r": 60, "t": 60, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uniformtext_minsize=10,
    )
    return figure
