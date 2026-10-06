import plotly.graph_objects as go

from src.visualization.financial_charts import (
    format_change_pct,
    format_value,
    render_metrics_comparison_chart,
)


def _metric(name, current, previous, change=None, unit="usd", comparison="QoQ"):
    return {
        "name": name,
        "current_value": current,
        "previous_value": previous,
        "change_pct": change,
        "unit": unit,
        "comparison_type": comparison,
        "current_period": "2026-04-01/2026-06-30",
        "previous_period": "2026-01-01/2026-03-31",
        "source_ids": [],
    }


def test_chart_plots_values_unchanged_one_panel_per_unit():
    metrics = [
        _metric("Revenue", 1_250_000_000.0, 1_180_000_000.0, 5.93),
        _metric("Net Income", -210_000_000.0, 225_000_000.0, -193.33),
        _metric("Diluted EPS", 1.42, 1.51, -5.96, unit="usdPerShare"),
    ]

    figure = render_metrics_comparison_chart(metrics)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 4  # Actual + Anterior for each of the 2 units
    usd_current, usd_previous, eps_current, _ = figure.data
    assert list(usd_current.x) == [1_250_000_000.0, -210_000_000.0]
    assert list(usd_previous.x) == [1_180_000_000.0, 225_000_000.0]
    assert list(eps_current.x) == [1.42]
    assert usd_current.xaxis != eps_current.xaxis  # never mix units on one axis


def test_chart_handles_empty_list():
    figure = render_metrics_comparison_chart([])

    assert len(figure.data) == 0
    assert figure.layout.annotations[0].text == "Sin métricas comparables para graficar"


def test_chart_handles_none_values():
    metrics = [
        _metric("Total Debt", 1_500_000_000.0, None, None, comparison=None),
        _metric("Ghost", None, None, None, comparison=None),
    ]

    figure = render_metrics_comparison_chart(metrics)

    current, previous = figure.data
    assert list(current.y) == ["Total Debt"]  # metric without any value is skipped
    assert list(previous.x) == [None]
    assert list(previous.text) == ["N/D"]
    figure.to_json()  # serializes without errors

    assert len(render_metrics_comparison_chart(metrics[1:]).data) == 0


def test_format_change_pct_uses_percentage_points_as_is():
    assert format_change_pct(-1.589) == "-1.59%"
    assert format_change_pct(5.932) == "+5.93%"
    assert format_change_pct(0.0) == "+0.00%"
    assert format_change_pct(None) == "N/D"


def test_format_value():
    assert format_value(None, "usd") == "N/D"
    assert format_value(1_250_000_000.0, "usd") == "$1.25B"
    assert format_value(-210_000_000.0, "usd") == "-$210.00M"
    assert format_value(1.42, "usdPerShare") == "$1.42"
    assert format_value(3.0, "shares") == "3.00 shares"
