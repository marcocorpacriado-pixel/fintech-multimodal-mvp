import plotly.graph_objects as go
import pytest

from src.visualization.financial_charts import (
    format_change_pct,
    format_value,
    render_metrics_comparison_chart,
)
from src.visualization.presentation import (
    error_presentation,
    filing_option_label,
    normalize_ticker_for_ui,
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
