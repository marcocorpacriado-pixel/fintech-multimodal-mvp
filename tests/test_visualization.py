import plotly.graph_objects as go
import pytest

from src.visualization.financial_charts import (
    format_change_pct,
    format_value,
    render_metrics_change_chart,
    render_metrics_comparison_chart,
)
from src.visualization.presentation import (
    error_presentation,
    filing_option_label,
    format_metric_period,
    human_source_label,
    normalize_ticker_for_ui,
    select_executive_metrics,
    sentiment_label,
    sort_filings,
    verification_label,
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
    assert figure.layout.annotations[0].text == "No comparable metrics available"


def test_chart_handles_none_values():
    metrics = [
        _metric("Total Debt", 1_500_000_000.0, None, None, comparison=None),
        _metric("Ghost", None, None, None, comparison=None),
    ]

    figure = render_metrics_comparison_chart(metrics)

    current, previous = figure.data
    assert list(current.y) == ["Total Debt"]  # metric without any value is skipped
    assert list(previous.x) == [None]
    assert list(previous.text) == ["N/A"]
    figure.to_json()  # serializes without errors

    assert len(render_metrics_comparison_chart(metrics[1:]).data) == 0


def test_change_chart_uses_delivered_percentages_and_neutral_colour():
    metrics = [
        _metric("Revenue", 100.0, 90.0, 11.111),
        _metric("Total Debt", 50.0, None, None, comparison=None),
    ]

    figure = render_metrics_change_chart(metrics)

    assert list(figure.data[0].x) == [11.111]
    assert list(figure.data[0].y) == ["Revenue"]
    assert figure.data[0].marker.color == "#607d8b"
    assert "Total Debt" not in figure.data[0].y


def test_change_chart_handles_no_comparable_metrics():
    figure = render_metrics_change_chart(
        [_metric("Total Debt", 50.0, None, None, comparison=None)]
    )

    assert len(figure.data) == 0
    assert figure.layout.annotations[0].text == "No comparable changes available"


def test_format_change_pct_uses_percentage_points_as_is():
    assert format_change_pct(-1.589) == "-1.59%"
    assert format_change_pct(5.932) == "+5.93%"
    assert format_change_pct(0.0) == "+0.00%"
    assert format_change_pct(None) == "N/A"


def test_format_value():
    assert format_value(None, "usd") == "N/A"
    assert format_value(1_250_000_000.0, "usd") == "$1.25B"
    assert format_value(-210_000_000.0, "usd") == "-$210.00M"
    assert format_value(1.42, "usdPerShare") == "$1.42"
    assert format_value(3.0, "shares") == "3.00 shares"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL", "AAPL"),
        ("aapl", "AAPL"),
        (" BRK.B ", "BRK.B"),
        ("BRK-B", "BRK-B"),
        ("", None),
        ("@@@", None),
        ("AAPL$", None),
        ("AAPL INC", None),
    ],
)
def test_ui_ticker_validation_matches_backend_contract(raw, expected):
    assert normalize_ticker_for_ui(raw) == expected


def test_filing_options_are_human_readable_and_newest_first():
    older = {
        "filing_date": "2026-05-01",
        "report_date": "2026-03-28",
        "form": "10-Q",
        "accession": "older",
    }
    amendment = {
        "filing_date": "2026-08-04",
        "report_date": "2026-06-27",
        "form": "10-Q/A",
        "accession": "amendment",
    }

    ordered = sort_filings([older, amendment])

    assert [item["accession"] for item in ordered] == ["amendment", "older"]
    assert filing_option_label(ordered[0]) == (
        "Report Jun 27, 2026 · Filed Aug 04, 2026 · 10-Q/A"
    )


@pytest.mark.parametrize(
    ("code", "title_fragment"),
    [
        ("INPUT_ERROR", "configuration"),
        ("FILING_NOT_FOUND", "not found"),
        ("SEC_INGESTION_ERROR", "SEC data"),
        ("LLM_PROVIDER_ERROR", "AI provider"),
        ("GROUNDING_ERROR", "evidence checks"),
    ],
)
def test_error_copy_is_actionable_and_category_specific(code, title_fragment):
    assert title_fragment in error_presentation(code).title


def test_verification_labels_distinguish_valid_warning_and_failure():
    assert verification_label({"valid": True, "issues": []}) == "VERIFIED"
    assert verification_label(
        {"valid": True, "issues": [{"severity": "warning"}]}
    ) == "VERIFIED WITH WARNINGS"
    assert verification_label(
        {"valid": False, "issues": [{"severity": "error"}]}
    ) == "FAILED VERIFICATION"


def test_metric_periods_are_humanized_without_inventing_quarters():
    assert format_metric_period("2026-04-01/2026-06-27") == "Apr 1–Jun 27, 2026"
    assert format_metric_period("2026-06-27") == "Jun 27, 2026"
    assert format_metric_period(None) == "Not available"


def test_executive_snapshot_uses_fixed_canonical_order_without_recalculation():
    metrics = [
        _metric("Capital Expenditures", 7.0, 8.0, -12.5),
        _metric("Operating Cash Flow", 117.0, 82.0, 43.1),
        _metric("Diluted EPS", 2.02, 2.01, 0.5, unit="usdPerShare"),
        _metric("Revenue", 109.4, 111.2, -1.6),
        _metric("Net Income", 29.8, 29.6, 0.7),
    ]

    snapshot = select_executive_metrics(metrics)

    assert [metric["name"] for metric in snapshot] == [
        "Revenue",
        "Net Income",
        "Diluted EPS",
        "Operating Cash Flow",
    ]
    assert snapshot[0]["change_pct"] == -1.6


def test_human_source_labels_prioritize_readable_provenance():
    assert human_source_label("10-Q", "PART_I_ITEM_2") == (
        "10-Q · Item 2 · Management Discussion & Analysis"
    )
    assert human_source_label("10-Q", "PART_II_ITEM_1A") == (
        "10-Q · Item 1A · Risk Factors"
    )
    assert human_source_label("10-Q", None) == "10-Q · Unsectioned filing content"


@pytest.mark.parametrize(
    ("outlook", "expected"),
    [
        (
            {"sentiment": "positive", "confidence": 0.942, "model": "ProsusAI/finbert"},
            "POSITIVE · 94.2% confidence · FinBERT",
        ),
        ({"sentiment": "mixed", "confidence": None, "model": None}, "MIXED"),
        ({"sentiment": "unknown"}, "UNKNOWN"),
    ],
)
def test_sentiment_label_shows_model_confidence_only_when_scored(outlook, expected):
    assert sentiment_label(outlook) == expected
