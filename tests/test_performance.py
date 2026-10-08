"""Performance tab helpers: session summary, UX targets, viability projection."""

import pytest

from src.visualization.performance import (
    DEFAULT_COST_USD,
    event_rows,
    latency_chart,
    latency_rows,
    project_monthly_cost,
    summarize,
)

EVENTS = [
    {"kind": "analysis", "latency_ms": 40_000, "cost_usd": 0.02, "label": "META"},
    {"kind": "chat", "latency_ms": 4_000, "ttft_ms": 900, "cost_usd": 0.004},
    {"kind": "chat", "latency_ms": 12_000, "ttft_ms": 1_500, "cost_usd": 0.006},
    {"kind": "tts", "latency_ms": 3_000, "cost_usd": 0.0001, "estimated": True},
]


def test_summary_totals_and_medians():
    summary = summarize(EVENTS)

    assert summary["count"] == 4
    assert summary["total_cost_usd"] == pytest.approx(0.0301)
    assert summary["median_chat_ttft_ms"] == 1_200
    assert summary["median_tts_ms"] == 3_000


def test_latency_rows_compare_median_against_targets():
    rows = {row["Interaction"]: row for row in latency_rows(EVENTS)}

    assert rows["Chat: first token"]["Status"] == "✅ within target"
    assert rows["Chat: full answer"]["Measured (median)"] == "8.0 s"
    assert rows["Chat: full answer"]["Status"] == "✅ within target"
    assert rows["Speech-to-text"]["Status"] == "— not measured yet"


def test_projection_uses_measured_averages_and_defaults():
    projection = project_monthly_cost(
        EVENTS, sessions_per_month=1_000, chat_turns=5, audio_plays=2,
        voice_questions=1, session_minutes=10, instance_usd_per_hour=0.288,
    )

    inference = 0.02 + 5 * 0.005 + 2 * 0.0001 + 1 * DEFAULT_COST_USD["stt"]
    assert projection["inference_per_session"] == pytest.approx(inference)
    assert projection["infra_per_session"] == pytest.approx(0.048)
    assert projection["monthly_total"] == pytest.approx((inference + 0.048) * 1_000)


def test_log_is_newest_first_and_marks_estimates():
    rows = event_rows(EVENTS)

    assert rows[0]["Component"] == "Text-to-speech"
    assert rows[0]["Cost"].endswith("(est.)")
    assert rows[-1]["Detail"] == "META"


def test_chart_has_one_trace_per_component():
    figure = latency_chart(EVENTS)

    assert {trace.name for trace in figure.data} == {
        "Filing analysis", "Chat answer", "Text-to-speech",
    }
