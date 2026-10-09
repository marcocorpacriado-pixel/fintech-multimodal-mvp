"""Behavioral tests for the Dash presentation against FastAPI in-process.

Callback logic is exercised as plain functions (the Dash wrappers only read
``ctx``); a couple of tests also go through Dash's own HTTP endpoints to prove
the callback wiring is valid.
"""

import ast
import base64
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dash import dcc, html
from fastapi.testclient import TestClient

from app import dash_app
from app.api_client import fetch_filings, fetch_pricing, fetch_voices
from app.components import (
    build_result,
    chat_error_view,
    chat_history,
    perf_summary_view,
    performance_section,
    summary_audio_view,
)
from src.api.main import _run_demo_analysis, app as api_app
from src.extraction import (
    LLMTransportError,
    ModelOutputProblem,
    ModelOutputRejectedError,
    SECFilingMetadata,
    SECServiceError,
)

APP_DIR = Path(__file__).resolve().parents[1] / "app"
api_client = TestClient(api_app)
KNOWN_FILING = SECFilingMetadata(
    ticker="AAPL",
    company="Apple Inc.",
    filing_date=date(2026, 7, 31),
    report_date=date(2026, 6, 27),
    form="10-Q",
    accession="0000320193-26-000020",
)
VOICES = ["af_heart", "bf_emma"]


def _route_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.request(method, url.removeprefix("http://localhost:8000"), **kwargs)


def _stream_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.stream(method, url.removeprefix("http://localhost:8000"), **kwargs)


@pytest.fixture(autouse=True)
def _in_process_api(monkeypatch):
    """Route every UI HTTP call to the in-process API; start with cold caches."""

    for cached in (fetch_voices, fetch_filings, fetch_pricing):
        cached.cache_clear()
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=VOICES),
    ):
        yield


class _FakeChatClient:
    """Stands in for OpenRouterChatClient: records the call, yields tokens."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def open(self, *, system_prompt, messages):
        self.calls.append({"system_prompt": system_prompt, "messages": messages})
        return iter(self.tokens)


def _real_handoff(*, warnings: bool = False):
    handoff = _run_demo_analysis()
    issues = handoff.verification.issues if warnings else []
    return handoff.model_copy(
        update={
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "period": "2026-06-27",
            "verification": handoff.verification.model_copy(
                update={"valid": True, "issues": issues}
            ),
            "pipeline_metadata": handoff.pipeline_metadata.model_copy(
                update={
                    "analysis_mode": "real",
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-flash",
                    "filing_date": date(2026, 7, 31),
                }
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Helpers over Dash component trees
# --------------------------------------------------------------------------- #
def _walk(node):
    """Every component below ``node`` (including it), through any child prop."""

    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk(child)
    elif hasattr(node, "_prop_names"):
        yield node
        for prop in ("children", "custom_spinner"):
            yield from _walk(getattr(node, prop, None))


def _text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(_text(child) for child in node)
    label = getattr(node, "label", None)
    own = f"{label} " if isinstance(label, str) else ""
    return own + _text(getattr(node, "children", None))


def _by_id(node, component_id):
    return next((c for c in _walk(node) if getattr(c, "id", None) == component_id), None)


def _with_class(node, name):
    return [c for c in _walk(node) if name in (getattr(c, "className", "") or "").split()]


def _demo_outputs(**overrides):
    args = {
        "trigger": None,
        "mode": "Demo",
        "ticker": None,
        "accession": None,
        "filings": [],
        "last_request": None,
    }
    args.update(overrides)
    return dash_app.run_analysis_logic(**args)


def _demo_handoff():
    return _demo_outputs()[0]


def _result(handoff=None):
    return build_result(handoff or _demo_handoff(), VOICES)


# --------------------------------------------------------------------------- #
# Analysis dashboard
# --------------------------------------------------------------------------- #
def test_demo_analysis_renders_professional_dashboard():
    handoff = _demo_handoff()
    result = build_result(handoff, VOICES)
    text = _text(result)

    assert "Deterministic synthetic fixture | No SEC or LLM request" in text
    assert "DEMO | SYNTHETIC" in text
    assert "Demo Corp" in text
    assert "VERIFIED WITH WARNINGS" in text
    assert "POSITIVE · 94.2% confidence · FinBERT" in text
    assert "$1.25B" in text and "N/A" in text
    assert "Item 2 · Management Discussion & Analysis" in text
    assert "Technical details" in text

    outlook_html = next(
        c.children for c in _walk(result) if isinstance(c, dcc.Markdown) and "POLARITY" in c.children
    )
    assert "POLARITY: POSITIVE" in outlook_html
    assert "Confidence: 94.2% · Model: ProsusAI/finbert" in outlook_html
    assert "Key evidence detected · FinBERT 94.6%" in outlook_html
    assert 'title="Impact: 94.0% (Integrated Gradients)"' in outlook_html
    assert ">growth</span>" in outlook_html and "Low impact" in outlook_html

    technical = next(
        c for c in _walk(result) if getattr(c, "className", None) == "data-table" and "Analysis mode" in _text(c)
    )
    cells = [_text(td) for td in _walk(technical) if isinstance(td, html.Td)]
    values = dict(zip(cells[::2], cells[1::2]))
    assert values["Analysis mode"] == "demo"
    assert values["Provider"] == "fixture"
    assert values["Generation attempts"] == "1"
    assert values["Repair used"] == "no"


def test_heatmap_pills_expose_exact_impact_on_hover():
    result = _result()
    outlook_html = next(
        c.children for c in _walk(result) if isinstance(c, dcc.Markdown) and "xai-pill" in c.children
    )
    demand_pill = re.search(r"<span class=\"xai-pill\"[^>]*>demand</span>", outlook_html)
    assert demand_pill, "demo 'demand' attribution (0.94) must render as a pill"
    pill = demand_pill.group(0)
    assert 'data-tooltip="Impact: 94.0%"' in pill
    assert 'title="Impact: 94.0% (Integrated Gradients)"' in pill
    assert "cursor: help" in pill and "display: inline-block" in pill
    assert "position: relative" in pill and "pointer-events: none" not in pill
    assert 'tabindex="0"' in pill

    # The instant tooltip itself is the app's global CSS rule for the pill class.
    css = (APP_DIR / "assets" / "style.css").read_text(encoding="utf-8")
    assert ".xai-pill:hover::after" in css
    assert "content: attr(data-tooltip)" in css


def test_html_outlook_cannot_be_split_by_blank_lines_in_filing_text():
    handoff = _demo_handoff()
    handoff["management_outlook"]["summary"] = "line one\n\n*not emphasis*\n\n<b>raw</b>"
    markup = next(
        c.children for c in _walk(build_result(handoff, VOICES)) if isinstance(c, dcc.Markdown) and "POLARITY" in c.children
    )

    assert "\n" not in markup
    assert "<b>raw</b>" not in markup and "&lt;b&gt;raw&lt;/b&gt;" in markup


def test_first_visit_runs_demo_once_and_never_needs_a_click():
    with patch("src.api.main._run_demo_analysis", wraps=_run_demo_analysis) as demo:
        outputs = _demo_outputs(trigger=None, mode="Real")  # initial call ignores the form

    handoff, error, request = outputs[:3]
    assert handoff["company"] == "Demo Corp (synthetic data)"
    assert error is None and request == {"ticker": "DEMO", "mode": "demo"}
    assert demo.call_count == 1


def test_real_mode_uses_discovered_filing_and_separates_dates():
    handoff = _real_handoff()
    with (
        patch("src.api.main.discover_sec_filings", return_value=[KNOWN_FILING]) as discovery,
        patch("src.api.main._run_real_analysis", return_value=handoff) as analysis,
    ):
        options, value, filings, message, _ = dash_app.filings_view_logic("Real", "AAPL", "10-Q", None)
        assert any("Report Jun 27, 2026" in option["label"] for option in options)
        assert value == KNOWN_FILING.accession and message is None
        outputs = dash_app.run_analysis_logic(
            "analyze", "Real", "AAPL", value, filings, None
        )

    assert discovery.called
    request = analysis.call_args.args[0]
    assert request.ticker == "AAPL"
    assert request.filing_date == date(2026, 7, 31)
    assert request.filing_type == "10-Q"
    text = _text(build_result(outputs[0], VOICES))
    assert "Report period: Jun 27, 2026" in text
    assert "Filed: Jul 31, 2026" in text
    assert "LIVE ANALYSIS" in text and "openrouter" in text
    assert "deepseek/deepseek-v4-flash" in text
    assert str(handoff.pipeline_metadata.retrieval_count) in text


def test_ticker_is_chosen_from_list_and_drives_filing_discovery():
    options = _by_id(dash_app.serve_layout(), "ticker").options
    assert {"label": "MSFT · Microsoft Corporation", "value": "MSFT"} in options
    assert _by_id(dash_app.serve_layout(), "ticker").value == "AAPL"
    assert not isinstance(_by_id(dash_app.serve_layout(), "ticker"), dcc.Input)  # closed list, no free text

    with patch("src.api.main.discover_sec_filings", return_value=[]) as discovery:
        dash_app.filings_view_logic("Real", "MSFT", "10-Q", None)

    assert discovery.call_args.kwargs["ticker"] == "MSFT"


def test_demo_mode_does_not_query_filings():
    with patch("src.api.main.discover_sec_filings") as discovery:
        with pytest.raises(dash_app.PreventUpdate):
            dash_app.filings_view_logic("Demo", "AAPL", "10-Q", None)

    discovery.assert_not_called()


def test_no_filings_has_clear_empty_state():
    with patch("src.api.main.discover_sec_filings", return_value=[]):
        options, value, filings, message, _ = dash_app.filings_view_logic("Real", "AAPL", "10-Q", None)

    assert (options, value, filings) == ([], None, [])
    assert "No filings were found" in _text(message)


def test_sec_unavailable_has_safe_retryable_filing_state():
    with patch("src.api.main.discover_sec_filings", side_effect=SECServiceError("private provider details")):
        _, _, _, message, retry_class = dash_app.filings_view_logic("Real", "AAPL", "10-Q", None)

    text = _text(message)
    assert "SEC data is temporarily unavailable" in text
    assert "private provider details" not in text
    assert "hidden" not in retry_class  # the retry button is shown


def test_retrying_filing_lookup_bypasses_the_cache():
    with patch("src.api.main.discover_sec_filings", side_effect=SECServiceError("down")):
        dash_app.filings_view_logic("Real", "AAPL", "10-Q", None)
    with patch("src.api.main.discover_sec_filings", return_value=[KNOWN_FILING]) as discovery:
        options, *_ = dash_app.filings_view_logic("Real", "AAPL", "10-Q", "retry-filings")
        dash_app.filings_view_logic("Real", "AAPL", "10-Q", None)  # now cached

    assert options and discovery.call_count == 1


def test_grounding_error_is_blocked_and_explained_without_partial_result():
    problem = ModelOutputProblem("key_risks[0].evidence_id", "INVALID_EVIDENCE_ID", "raw evidence and secret details")
    with patch("src.api.main._run_real_analysis", side_effect=ModelOutputRejectedError([problem])):
        outputs = _demo_outputs(
            trigger="analyze", mode="Real", ticker="AAPL", accession="a", filings=[_filing_dict("a")]
        )

    handoff, error = outputs[:2]
    assert handoff is None and outputs[3] == []  # no partial result, no perf event
    page, retry_class = dash_app.render_result(handoff, error)
    text = _text(page)
    assert "did not pass evidence checks" in text
    assert "exact, contiguous filing excerpt" in text
    assert "raw evidence" not in text
    assert "Canonical financial metrics" not in text
    assert "hidden" not in retry_class


def _filing_dict(accession):
    return {
        "accession": accession,
        "filing_date": "2026-07-31",
        "report_date": "2026-06-27",
        "form": "10-Q",
    }


def test_retryable_llm_error_preserves_configuration_and_offers_retry():
    with patch("src.api.main._run_real_analysis", side_effect=LLMTransportError("body with sk-or-SECRET")):
        outputs = _demo_outputs(
            trigger="analyze", mode="Real", ticker="AAPL", accession="a", filings=[_filing_dict("a")]
        )
    handoff, error, request = outputs[:3]
    page, retry_class = dash_app.render_result(handoff, error)
    text = _text(page)

    assert "AI provider did not return a usable response" in text
    assert "SECRET" not in text
    assert "hidden" not in retry_class

    # Retry replays the stored request without re-reading the form.
    with patch("src.api.main._run_real_analysis", return_value=_real_handoff()) as analysis:
        retried = _demo_outputs(trigger="retry-analysis", mode="Demo", last_request=request)
    assert analysis.call_args.args[0].ticker == "AAPL"
    assert retried[0]["ticker"] == "AAPL" and retried[1] is None


def test_incomplete_real_selection_warns_without_touching_state():
    outputs = _demo_outputs(trigger="analyze", mode="Real", ticker="AAPL", accession=None, filings=[])

    assert "Select an available filing" in outputs[-1]
    assert all(value is dash_app.no_update for value in outputs[:-1])


def test_unreachable_api_is_a_safe_error():
    with patch("httpx.request", side_effect=__import__("httpx").ConnectError("boom")):
        outputs = _demo_outputs()

    assert outputs[0] is None and outputs[1]["code"] == "API_UNAVAILABLE"
    assert "analysis service is unavailable" in _text(dash_app.render_result(None, outputs[1])[0])


def test_failed_verification_never_renders_analysis_as_valid():
    handoff = _run_demo_analysis()
    blocking_issue = handoff.verification.issues[0].model_copy(update={"severity": "error"})
    invalid = handoff.model_copy(
        update={
            "verification": handoff.verification.model_copy(
                update={"valid": False, "issues": [blocking_issue]}
            )
        }
    )
    with patch("src.api.main._run_demo_analysis", return_value=invalid):
        result = build_result(_demo_handoff(), VOICES)

    text = _text(result)
    assert "failed deterministic verification" in text
    assert "FAILED VERIFICATION" in text
    assert "Canonical financial metrics" not in text
    assert not [c for c in _walk(result) if isinstance(c, dcc.Tabs)]


def test_empty_state_before_any_analysis():
    page, retry_class = dash_app.render_result(None, None)

    assert "Analyze filing" in _text(page)
    assert "hidden" in retry_class


def test_results_use_layered_information_architecture_and_snapshot():
    result = _result()
    tabs = next(c for c in _walk(result) if getattr(c, "id", None) == "result-tabs")

    assert [tab.label for tab in tabs.children] == [
        "Overview",
        "Financials",
        "Narrative",
        "Sources",
        "Performance",
    ]
    assert "Executive snapshot" in _text(result)
    overview = tabs.children[0]
    assert [_text(c) for c in _with_class(overview, "metric__label")] == [
        "Revenue",
        "Net Income",
        "Diluted EPS",
        "Operating Cash Flow",
    ]
    assert len(_with_class(result, "metric__label")) == 11  # 4-metric snapshot plus the 7-metric grid


def test_change_chart_is_default_and_values_view_is_available():
    result = _result()
    chart_tabs = _by_id(result, "chart-tabs")

    assert chart_tabs.value == "change"
    assert [tab.label for tab in chart_tabs.children] == ["Change %", "Values"]
    assert all(isinstance(_walk_graph(tab), dcc.Graph) for tab in chart_tabs.children)
    assert len([c for c in _walk(result) if isinstance(c, dcc.Graph)]) == 2


def _walk_graph(tab):
    return next(c for c in _walk(tab.children) if isinstance(c, dcc.Graph))


def test_charts_use_the_dark_terminal_theme():
    figure = _walk_graph(_by_id(_result(), "chart-tabs").children[0]).figure

    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.font.color == "#F8FAFC"


def test_audio_preference_is_local_to_summary_not_sidebar():
    layout = dash_app.serve_layout()
    sidebar = next(c for c in _walk(layout) if isinstance(c, html.Aside))
    result = _result()

    assert "Voice" not in _text(sidebar)
    voice = _by_id(result, "summary-voice")
    assert voice.options == VOICES and voice.value == "af_heart"
    assert _text(_by_id(result, "summary-copy")) == "Copy summary"
    assert _text(_by_id(result, "summary-listen")) == "Listen to summary"


def test_tts_uses_executive_summary_without_rerunning_analysis():
    handoff = _demo_handoff()
    audio = SimpleNamespace(audio_bytes=b"RIFF-test-wav")
    with (
        patch("src.api.main.synthesize", return_value=audio) as synthesize,
        patch("src.api.main._run_demo_analysis", wraps=_run_demo_analysis) as demo,
    ):
        stored, perf = dash_app.summary_listen_logic(handoff, "af_heart", [])

    assert synthesize.call_args.kwargs["text"] == _run_demo_analysis().executive_summary
    assert demo.call_count == 0  # listening never re-runs the analysis
    assert stored["src"] == "data:audio/wav;base64," + base64.b64encode(b"RIFF-test-wav").decode()
    assert [e["kind"] for e in perf] == ["tts"]
    view = summary_audio_view(stored)
    assert isinstance(view[0], html.Audio)
    assert "Audio generated from synthetic demo analysis" in _text(view)


def test_summary_audio_failure_is_a_safe_error():
    with patch("src.api.main.synthesize", side_effect=RuntimeError("kokoro exploded")):
        stored, perf = dash_app.summary_listen_logic(_demo_handoff(), "af_heart", [])

    assert perf == [] and "error" in stored
    assert "kokoro exploded" not in _text(summary_audio_view(stored))


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def _ask(question, handoff=None, messages=None, perf=None):
    """Submit a typed question, then answer it, like the two chained callbacks."""

    handoff = handoff or _demo_handoff()
    submitted = dash_app.chat_submit_logic("chat-send", question, None, messages or [], perf or [])
    new_messages, _, perf_after, _, _, token = submitted
    return dash_app.chat_answer_logic(token, new_messages, handoff, perf_after), new_messages


def test_chat_answers_with_cited_sources():
    fake = _FakeChatClient(["Revenue rose ", "5.9% QoQ [M1]."])
    with patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake):
        (messages, error, perf), _ = _ask("How did revenue change?")

    assert error is None
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "How did revenue change?"}]
    assert "[M1] Revenue: current=1,250,000,000.0 usd" in fake.calls[0]["system_prompt"]
    assert messages[-1] == {"role": "assistant", "content": "Revenue rose 5.9% QoQ [M1]."}
    text = _text(chat_history(messages, {}, "local", "af_heart", _demo_handoff()))
    assert "Sources cited (1)" in text
    assert "Metric: Revenue" in text
    assert "Listen" in text


def test_chat_shows_the_question_immediately_and_marks_the_answer_pending():
    messages, error, _, cleared, status, token = dash_app.chat_submit_logic(
        "chat-send", "  Main risks?  ", None, [], []
    )

    assert messages == [{"role": "user", "content": "Main risks?"}]
    assert error is None and cleared == "" and status == "" and token
    pending = chat_history(messages, {}, "local", "af_heart", _demo_handoff())
    assert "Thinking…" in _text(pending)


def test_chat_ignores_empty_and_concurrent_submissions():
    with pytest.raises(dash_app.PreventUpdate):
        dash_app.chat_submit_logic("chat-send", "   ", None, [], [])
    with pytest.raises(dash_app.PreventUpdate):  # an answer is already in flight
        dash_app.chat_submit_logic("chat-send", "again", None, [{"role": "user", "content": "x"}], [])


def test_chat_suggestions_submit_their_own_text():
    messages, *_ = dash_app.chat_submit_logic({"type": "chat-suggestion", "index": 1}, "", None, [], [])

    assert messages[-1]["content"] == "What are the main risks?"


def test_chat_provider_error_is_safe_and_history_stays_answerable():
    with patch(
        "src.api.main.OpenRouterChatClient.from_env",
        side_effect=LLMTransportError("provider body sk-or-SECRET"),
    ):
        (messages, error, _), _ = _ask("Main risks?")

    assert messages == []  # the unanswered question is dropped: history must end answered
    text = _text(chat_error_view(error))
    assert "The assistant could not answer (LLM_PROVIDER_ERROR)." in text
    assert "SECRET" not in text


def test_new_analysis_clears_chat_audio_error_and_performance_log():
    outputs = _demo_outputs(trigger="analyze", mode="Demo")
    _, error, request, perf, chat, chat_audio, summary_audio, chat_error, warning = outputs

    assert (chat, chat_audio, summary_audio, chat_error, warning) == ([], {}, None, None, "")
    assert error is None and request == {"ticker": "DEMO", "mode": "demo"}
    assert len(perf) == 1
    assert "demo fixture, no inference" in perf[0]["label"]
    assert perf[0]["cost_usd"] is None  # never counted as a measured analysis cost


def test_chat_is_a_floating_launcher_not_a_tab():
    handoff = _demo_handoff()
    closed = dash_app.chat_panel_logic(None, handoff, "chat-panel")
    assert closed == ("chat-panel", "chat-fab", "app-shell")

    opened = dash_app.chat_panel_logic("chat-fab", handoff, closed[0])
    assert opened == ("chat-panel chat-panel--open", "chat-fab hidden", "app-shell app-shell--chat-open")
    # Not a modal: the report keeps rendering and the page just makes room.
    assert "Executive snapshot" in _text(_result(handoff))
    assert all("Ask" not in tab.label for tab in _by_id(_result(handoff), "result-tabs").children)

    assert dash_app.chat_panel_logic("chat-close", handoff, opened[0]) == closed
    # A new report starts a fresh chat; no report means no launcher at all.
    assert dash_app.chat_panel_logic("st-handoff", handoff, opened[0]) == closed
    assert dash_app.chat_panel_logic(None, None, opened[0])[1] == "chat-fab hidden"


def test_launcher_is_hidden_for_a_failed_verification():
    handoff = _demo_handoff()
    handoff["verification"] = {**handoff["verification"], "valid": False}

    assert dash_app.chat_panel_logic("chat-fab", handoff, "chat-panel")[1] == "chat-fab hidden"


def test_chat_history_renders_markdown_for_answers_and_plain_text_for_questions():
    messages = [
        {"role": "user", "content": "**not bold** <b>x</b>"},
        {"role": "assistant", "content": "**Risks:** cash $830M to $910M"},
    ]
    bubbles = chat_history(messages, {}, "local", "af_heart", _demo_handoff())

    assert bubbles[0].children == "**not bold** <b>x</b>"  # React escapes it
    assert isinstance(bubbles[1].children[0], dcc.Markdown)
    assert bubbles[1].children[0].children == "**Risks:** cash $830M to $910M"


def test_chat_listen_uses_selected_groq_engine_and_voice():
    messages = [
        {"role": "user", "content": "Risks?"},
        {"role": "assistant", "content": "Cash fell ~8.8% [R1]."},
    ]
    audio = SimpleNamespace(audio_bytes=b"RIFF-groq")
    with (
        patch("src.api.main.synthesize_groq", return_value=audio) as groq,
        patch("src.api.main.synthesize") as kokoro,
    ):
        cache, error, perf = dash_app.chat_listen_logic(1, "groq", "hannah", messages, {}, [])

    groq.assert_called_once_with("Cash fell about 8.8 percent.", voice="hannah")
    kokoro.assert_not_called()
    assert error is None and [e["kind"] for e in perf] == ["tts"]
    assert cache["1:groq:hannah"]["note"] == "Read with Groq (fast) · hannah"
    shown = _text(chat_history(messages, cache, "groq", "hannah", _demo_handoff()))
    assert "Read with Groq (fast) · hannah" in shown and "Listen" not in shown
    # Cached per engine and voice: another voice offers Listen again.
    assert "Listen" in _text(chat_history(messages, cache, "groq", "troy", _demo_handoff()))


def test_spanish_answer_falls_back_to_kokoro_even_with_groq_selected():
    messages = [
        {"role": "user", "content": "¿Riesgos?"},
        {"role": "assistant", "content": "El efectivo cayó un ~8,8% [R1]."},
    ]
    audio = SimpleNamespace(audio_bytes=b"RIFF-kokoro")
    with (
        patch("src.api.main.list_voices", return_value=["af_heart", "ef_dora"]),
        patch("src.api.main.synthesize_groq") as groq,
        patch("src.api.main.synthesize", return_value=audio) as kokoro,
    ):
        cache, _, _ = dash_app.chat_listen_logic(1, "groq", "troy", messages, {}, [])

    groq.assert_not_called()
    kokoro.assert_called_once()
    assert kokoro.call_args.kwargs["language"] == "es"
    assert cache["1:groq:troy"]["note"].startswith("Read with Local (Kokoro)")
    assert "(Groq has no Spanish voice)" in cache["1:groq:troy"]["note"]


def test_chat_listen_rejects_user_messages_and_reports_audio_errors():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    with pytest.raises(dash_app.PreventUpdate):
        dash_app.chat_listen_logic(0, "local", "af_heart", messages, {}, [])
    with patch("src.api.main.synthesize", side_effect=RuntimeError("boom")):
        cache, error, perf = dash_app.chat_listen_logic(1, "local", "af_heart", messages, {}, [])

    assert cache is dash_app.no_update and perf is dash_app.no_update
    assert "Audio could not be generated" in _text(chat_error_view(error))


def test_chat_voices_follow_the_selected_engine():
    with patch("src.api.main.list_voices", return_value=["af_heart", "bf_emma", "ef_dora", "troy"]):
        local, default = dash_app.chat_voices_logic("local")
        fetch_voices.cache_clear()
        groq, _ = dash_app.chat_voices_logic("groq")

    assert local == ["af_heart", "bf_emma"] and default == "af_heart"
    assert groq[0] == "troy"


def test_voice_question_is_transcribed_and_logged():
    wav = base64.b64encode(b"RIFFwav").decode()
    transcript = SimpleNamespace(text="  What about risks? ", language="en", duration=3.0)
    with patch("src.api.main.transcribe", return_value=transcript):
        messages, error, perf, _, status, token = dash_app.chat_submit_logic(
            "st-voice", "", {"wav": wav, "t": 1}, [], []
        )

    assert messages == [{"role": "user", "content": "What about risks?"}]
    assert error is None and status == "" and token
    assert [e["kind"] for e in perf] == ["stt"]
    assert perf[0]["units"] == "3.0 s audio"


def test_voice_question_failures_are_safe():
    with patch("src.api.main.transcribe", side_effect=RuntimeError("groq key sk-SECRET")):
        outputs = dash_app.chat_submit_logic(
            "st-voice", "", {"wav": base64.b64encode(b"x").decode()}, [], []
        )
    assert outputs[0] is dash_app.no_update and outputs[1]["code"] == "UNKNOWN_ERROR"
    assert "SECRET" not in _text(chat_error_view(outputs[1]))

    bad = dash_app.chat_submit_logic("st-voice", "", {"wav": "%%%not-base64"}, [], [])
    assert bad[0] is dash_app.no_update and bad[1]["code"] == "INPUT_ERROR"


# --------------------------------------------------------------------------- #
# Performance tab
# --------------------------------------------------------------------------- #
def test_performance_tab_logs_chat_turn_and_hides_the_metrics_trailer():
    class Stream:
        usage = SimpleNamespace(prompt_tokens=900, completion_tokens=80, cost=0.0042)
        first_token_s = 0.5
        total_s = 1.0

        def __iter__(self):
            yield "Revenue rose [M1]."

    fake = _FakeChatClient([])
    fake.open = lambda **_: Stream()
    fake.model = "test/model"
    perf = _demo_outputs()[3]
    assert [e["kind"] for e in perf] == ["analysis"]
    with patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake):
        (messages, error, perf), _ = _ask("Revenue?", perf=perf)

    assert [e["kind"] for e in perf] == ["analysis", "chat"]
    chat = perf[-1]
    assert chat["cost_usd"] == 0.0042
    assert chat["units"] == "900 in / 80 out tokens"
    assert chat["provider"] == "OpenRouter · test/model"
    assert messages[-1]["content"] == "Revenue rose [M1]."
    view = perf_summary_view(perf)
    assert "\x1e" not in _text(view)
    tiles = {
        _text(c.children[0]): _text(c.children[1]) for c in _with_class(view, "metric")
    }
    assert tiles["Inferences"] == "2"
    assert tiles["Session cost"] == "$0.0042"


def test_performance_projection_uses_the_pricing_endpoint_and_inputs():
    events = _demo_outputs()[3]
    view = dash_app.projection_logic(events, 1000, 5, 2, 1, 10)
    text = _text(view)

    assert "Monthly total" in text and "Cloud Run instance" in text
    assert "billed per request" in text
    cheaper = _text(dash_app.projection_logic(events, 100, 0, 0, 0, 1))
    assert cheaper != text


def test_performance_section_declares_the_widgets_the_callbacks_listen_to():
    section = performance_section()

    for component_id in (
        "perf-summary", "perf-projection", "proj-sessions", "proj-turns",
        "proj-audios", "proj-voices", "proj-minutes",
    ):
        assert _by_id(section, component_id) is not None


# --------------------------------------------------------------------------- #
# Wiring, packaging and architecture rules
# --------------------------------------------------------------------------- #
def _collect_ids(node):
    return {c.id for c in _walk(node) if isinstance(getattr(c, "id", None), str)}


def _callback_ids():
    ids = set()
    for entry in dash_app.app._callback_list:
        for ref in [*entry["inputs"], *entry.get("state", [])]:
            ids.add(ref["id"])
        for output in re.findall(r"([\w-]+)(?:@[\w]+)?\.\w+", entry["output"].replace("..", " ")):
            ids.add(output)
    return {i for i in ids if not i.startswith("{")}


def test_every_callback_id_exists_in_the_layout_or_the_report():
    layout_ids = _collect_ids(dash_app.serve_layout())
    report_ids = _collect_ids(_result())
    missing = _callback_ids() - layout_ids - report_ids

    assert not missing, f"callbacks reference components that are never rendered: {missing}"
    # Report widgets are created per analysis, never in the static layout.
    assert {"result-tabs", "summary-listen", "proj-sessions"} <= report_ids - layout_ids


def test_layout_is_rebuilt_per_page_load():
    assert callable(dash_app.app.layout)
    assert dash_app.serve_layout() is not dash_app.serve_layout()


def test_static_layout_has_no_pending_state_without_a_report():
    layout = dash_app.serve_layout()

    assert "hidden" in _by_id(layout, "chat-fab").className.split()
    assert "chat-panel--open" not in _by_id(layout, "chat-panel").className
    assert _by_id(layout, "st-perf").data == [] and _by_id(layout, "st-chat").data == []


def _dash_client():
    return dash_app.server.test_client()


def test_health_endpoint_for_smoke_checks():
    response = _dash_client().get("/_health")

    assert response.status_code == 200 and response.data == b"ok"


def test_dash_serves_layout_and_assets():
    client = _dash_client()

    layout = client.get("/_dash-layout")
    assert layout.status_code == 200 and "Analysis configuration" in layout.get_data(as_text=True)
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/assets/recorder.js").status_code == 200


def _callback_key(first_output):
    return next(key for key in dash_app.app.callback_map if key.startswith(f"..{first_output}"))


def test_dash_endpoint_runs_the_initial_demo_analysis():
    """The autoload callback end to end through Dash's own HTTP protocol."""

    key = _callback_key("st-handoff.data")
    spec = dash_app.app.callback_map[key]
    outputs = [
        {"id": part.split(".")[0], "property": part.split(".")[1]}
        for part in key.strip(".").split("...")
    ]
    body = {
        "output": key,
        "outputs": outputs,
        "inputs": [{**ref, "value": None} for ref in spec["inputs"]],
        "state": [
            {**ref, "value": {"mode": "Demo"}.get(ref["id"], None)} for ref in spec["state"]
        ],
        "changedPropIds": [],
    }
    response = _dash_client().post("/_dash-update-component", json=body)

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = json.loads(response.get_data(as_text=True))["response"]
    assert payload["st-handoff"]["data"]["company"] == "Demo Corp (synthetic data)"
    assert payload["st-error"]["data"] is None
    assert payload["st-chat"]["data"] == [] and payload["st-chat-audio"]["data"] == {}
    assert payload["st-perf"]["data"][0]["kind"] == "analysis"


def test_ui_never_imports_backend_modules_or_streamlit():
    forbidden = ("src.extraction", "src.integration", "src.audio", "src.api", "edgar", "streamlit")
    for path in APP_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module if isinstance(node, ast.ImportFrom) else alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not [m for m in imported if m and m.startswith(forbidden)], path.name


def test_no_decorative_emoji_or_informal_pictograms_in_the_ui_sources():
    for path in [*APP_DIR.glob("*.py"), *(APP_DIR / "assets").glob("*")]:
        source = path.read_text(encoding="utf-8")
        for informal_icon in ("🏊", "♿", "🐶", "🐱"):
            assert informal_icon not in source, path.name


def test_idle_pause_machinery_is_gone_because_there_is_no_websocket():
    for path in [*APP_DIR.glob("*.py"), *(APP_DIR / "assets").glob("*")]:
        assert "IDLE_TIMEOUT" not in path.read_text(encoding="utf-8"), path.name
    assert not (APP_DIR / "static").exists()
