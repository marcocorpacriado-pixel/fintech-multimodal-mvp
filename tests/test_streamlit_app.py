"""Behavioral tests for the Streamlit presentation against FastAPI in-process."""

import ast
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api.main import _run_demo_analysis, app as api_app
from src.extraction import (
    LLMTransportError,
    ModelOutputProblem,
    ModelOutputRejectedError,
    SECFilingMetadata,
    SECServiceError,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
api_client = TestClient(api_app)
KNOWN_FILING = SECFilingMetadata(
    ticker="AAPL",
    company="Apple Inc.",
    filing_date=date(2026, 7, 31),
    report_date=date(2026, 6, 27),
    form="10-Q",
    accession="0000320193-26-000020",
)


def _route_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.request(method, url.removeprefix("http://localhost:8000"), **kwargs)


def _stream_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.stream(method, url.removeprefix("http://localhost:8000"), **kwargs)


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


def _app_test() -> AppTest:
    st.cache_data.clear()
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


def _all_text(at: AppTest) -> str:
    kinds = (
        "markdown",
        "caption",
        "warning",
        "error",
        "success",
        "info",
        "title",
        "header",
        "subheader",
    )
    return " ".join(
        str(element.value)
        for kind in kinds
        for element in getattr(at, kind, [])
    )


def _button(at: AppTest, label: str):
    return next(button for button in at.button if button.label == label)


def _run_real_sidebar(at: AppTest) -> AppTest:
    return at.sidebar.radio[0].set_value("Real").run()


def test_demo_analysis_renders_professional_dashboard():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart", "bf_emma"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    assert not at.exception
    text = _all_text(at)
    assert "Deterministic synthetic fixture | No SEC or LLM request" in text
    assert "DEMO | SYNTHETIC" in text
    assert "Demo Corp" in text
    assert "Financial Intelligence Copilot" in text
    assert "VERIFIED WITH WARNINGS" in text
    assert "POSITIVE · 94.2% confidence · FinBERT" in text
    outlook_html = "".join(element.proto.body for element in at.get("html"))
    assert "POLARITY: POSITIVE" in outlook_html
    assert "Confidence: 94.2% · Model: ProsusAI/finbert" in outlook_html
    assert "Key evidence detected · FinBERT 94.6%" in outlook_html
    assert 'title="Impact: 94.0% (Integrated Gradients)"' in outlook_html
    assert ">growth</span>" in outlook_html and "Low impact" in outlook_html
    assert len(at.metric) == 11  # 4-metric snapshot plus the 7-metric financial grid
    assert at.metric[0].value == "$1.25B"
    assert "N/A" in text
    assert any("Item 2 · Management Discussion & Analysis" in e.label for e in at.expander)
    assert any(expander.label == "Technical details" for expander in at.expander)
    technical = dict(at.dataframe[-1].value.itertuples(index=False, name=None))
    assert technical["Analysis mode"] == "demo"
    assert technical["Provider"] == "fixture"
    assert technical["Generation attempts"] == "1"
    assert technical["Repair used"] == "no"


def test_heatmap_pills_expose_exact_impact_on_hover():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    outlook_html = "".join(element.proto.body for element in at.get("html"))
    demand_pill = re.search(r"<span class=\"xai-pill\"[^>]*>demand</span>", outlook_html)
    assert demand_pill, "demo 'demand' attribution (0.94) must render as a pill"
    pill = demand_pill.group(0)
    assert 'data-tooltip="Impact: 94.0%"' in pill
    assert 'title="Impact: 94.0% (Integrated Gradients)"' in pill
    assert "cursor: help" in pill and "display: inline-block" in pill
    assert "position: relative" in pill and "pointer-events: none" not in pill
    assert 'tabindex="0"' in pill

    # The instant tooltip itself is the app's global CSS rule for the pill class.
    source = APP_PATH.read_text(encoding="utf-8")
    assert ".xai-pill:hover::after" in source
    assert "content: attr(data-tooltip)" in source


def test_first_visit_autoloads_demo_once():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main._run_demo_analysis", wraps=_run_demo_analysis) as demo,
    ):
        at = _app_test().run()
        at.run()

    assert not at.exception
    assert "Demo Corp" in _all_text(at)
    assert demo.call_count == 1


def test_real_mode_uses_discovered_filing_and_separates_dates():
    handoff = _real_handoff()
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.discover_sec_filings", return_value=[KNOWN_FILING]) as discovery,
        patch("src.api.main._run_real_analysis", return_value=handoff) as analysis,
    ):
        at = _run_real_sidebar(_app_test().run())
        assert any("Report Jun 27, 2026" in option for option in at.sidebar.selectbox[2].options)
        _button(at, "Analyze filing").click().run()

    assert not at.exception
    assert discovery.called
    request = analysis.call_args.args[0]
    assert request.ticker == "AAPL"
    assert request.filing_date == date(2026, 7, 31)
    assert request.filing_type == "10-Q"
    text = _all_text(at)
    assert "Report period: Jun 27, 2026" in text
    assert "Filed: Jul 31, 2026" in text
    assert "VERIFIED" in text
    technical = dict(at.dataframe[-1].value.itertuples(index=False, name=None))
    assert technical["Provider"] == "openrouter"
    assert technical["Model"] == "deepseek/deepseek-v4-flash"
    assert technical["Retrieved evidence chunks"] == str(
        handoff.pipeline_metadata.retrieval_count
    )


def test_ticker_is_chosen_from_list_and_drives_filing_discovery():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.discover_sec_filings", return_value=[]) as discovery,
    ):
        at = _run_real_sidebar(_app_test().run())
        ticker = at.sidebar.selectbox(key="ticker_select")
        assert ticker.value == "AAPL"
        assert "MSFT · Microsoft Corporation" in ticker.options
        ticker.set_value("MSFT").run()

    assert not at.sidebar.text_input
    assert discovery.call_args.kwargs["ticker"] == "MSFT"


def test_no_filings_has_clear_empty_state():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.discover_sec_filings", return_value=[]),
    ):
        at = _run_real_sidebar(_app_test().run())

    assert "No filings were found" in _all_text(at)
    assert not at.exception


def test_sec_unavailable_has_safe_retryable_filing_state():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch(
            "src.api.main.discover_sec_filings",
            side_effect=SECServiceError("private provider details"),
        ),
    ):
        at = _run_real_sidebar(_app_test().run())

    text = _all_text(at)
    assert "SEC data is temporarily unavailable" in text
    assert "private provider details" not in text
    assert any(button.label == "Retry filing lookup" for button in at.sidebar.button)


def test_grounding_error_is_blocked_and_explained_without_partial_result():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.discover_sec_filings", return_value=[KNOWN_FILING]),
        patch(
            "src.api.main._run_real_analysis",
            side_effect=ModelOutputRejectedError(
                [
                    ModelOutputProblem(
                        "key_risks[0].evidence_id",
                        "INVALID_EVIDENCE_ID",
                        "raw evidence and secret details",
                    )
                ]
            ),
        ),
    ):
        at = _run_real_sidebar(_app_test().run())
        _button(at, "Analyze filing").click().run()
        at.run()

    text = _all_text(at)
    assert "did not pass evidence checks" in text
    assert "exact, contiguous filing excerpt" in text
    assert "raw evidence" not in text
    assert len(at.metric) == 0
    assert any(button.label == "Retry analysis" for button in at.button)
    assert not at.exception


def test_retryable_llm_error_preserves_configuration_and_offers_retry():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.discover_sec_filings", return_value=[KNOWN_FILING]),
        patch(
            "src.api.main._run_real_analysis",
            side_effect=LLMTransportError("body with sk-or-SECRET"),
        ),
    ):
        at = _run_real_sidebar(_app_test().run())
        _button(at, "Analyze filing").click().run()
        at.run()

    text = _all_text(at)
    assert "AI provider did not return a usable response" in text
    assert "selected configuration is preserved" in text
    assert "SECRET" not in text
    assert at.sidebar.selectbox(key="ticker_select").value == "AAPL"
    assert any(button.label == "Retry analysis" for button in at.button)


def test_failed_verification_never_renders_analysis_as_valid():
    handoff = _run_demo_analysis()
    blocking_issue = handoff.verification.issues[0].model_copy(
        update={"severity": "error"}
    )
    invalid = handoff.model_copy(
        update={
            "verification": handoff.verification.model_copy(
                update={"valid": False, "issues": [blocking_issue]}
            )
        }
    )
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main._run_demo_analysis", return_value=invalid),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    assert "failed deterministic verification" in _all_text(at)
    assert len(at.metric) == 0
    assert not at.exception


def test_tts_still_uses_executive_summary_without_rerunning_analysis():
    audio = SimpleNamespace(audio_bytes=b"RIFF-test-wav")
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.synthesize", return_value=audio) as synthesize,
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Listen to summary").click().run()

    assert not at.exception
    assert synthesize.call_args.kwargs["text"] == _run_demo_analysis().executive_summary
    assert "Audio generated from synthetic demo analysis" in _all_text(at)


def test_results_use_layered_information_architecture_and_snapshot():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    assert [tab.label for tab in at.tabs] == [
        ":material/dashboard: Overview",
        ":material/monitoring: Financials",
        ":material/article: Narrative",
        ":material/verified_user: Sources",
    ]
    assert "Executive snapshot" in _all_text(at)
    assert [metric.label for metric in at.metric[:4]] == [
        "Revenue",
        "Net Income",
        "Diluted EPS",
        "Operating Cash Flow",
    ]


def test_change_chart_is_default_and_values_view_is_available():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    chart_radio = next(radio for radio in at.radio if radio.label == "Financial chart")
    assert chart_radio.value == "Change %"
    assert chart_radio.options == ["Change %", "Values"]
    assert len(at.get("plotly_chart")) == 1

    chart_radio.set_value("Values").run()
    assert not at.exception


def test_audio_preference_is_local_to_summary_not_sidebar():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart", "bf_emma"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()

    assert not any(box.label == "Voice" for box in at.sidebar.selectbox)
    assert any(box.label == "Voice" for box in at.selectbox)
    assert any(button.label == "Copy summary" for button in at.button)
    assert any(button.label == "Listen to summary" for button in at.button)


def test_loading_uses_neutral_spinner_and_no_status_widget_or_decorative_emoji():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "st.status(" not in source
    assert 'st.spinner("Preparing analysis")' in source
    for informal_icon in ("🏊", "♿", "🐶", "🐱"):
        assert informal_icon not in source


def test_native_running_widget_with_informal_pictograms_is_hidden():
    # Streamlit's own header "Running..." widget cycles cyclist/swimmer/wheelchair
    # icons on every rerun; the app hides it and relies on neutral spinners.
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()

    assert not at.exception
    source = APP_PATH.read_text(encoding="utf-8")
    assert "stStatusWidget" in source
    assert "stStatusWidget\"] { display: none; }" in source


def test_app_never_imports_backend_modules():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("src.extraction", "src.integration", "src.audio", "src.api", "edgar")
    assert not [module for module in imported if module and module.startswith(forbidden)]


def test_chat_streams_grounded_answer_with_cited_sources():
    fake = _FakeChatClient(["Revenue rose ", "5.9% QoQ [M1]."])
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Ask about this filing").click().run()
        at.chat_input[0].set_value("How did revenue change?").run()

    assert not at.exception
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": "How did revenue change?"}
    ]
    assert "[M1] Revenue: current=1,250,000,000.0 usd" in fake.calls[0]["system_prompt"]
    messages = at.session_state["chat_messages"]
    assert messages[-1] == {"role": "assistant", "content": "Revenue rose 5.9% QoQ [M1]."}
    assert any(expander.label == "Sources cited (1)" for expander in at.expander)
    assert "Chat uses the configured LLM over the synthetic demo analysis." in _all_text(at)


def test_chat_provider_error_is_safe_and_history_stays_answerable():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch(
            "src.api.main.OpenRouterChatClient.from_env",
            side_effect=LLMTransportError("provider body sk-or-SECRET"),
        ),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Ask about this filing").click().run()
        at.chat_input[0].set_value("Main risks?").run()

    assert not at.exception
    assert at.session_state["chat_messages"] == []
    text = _all_text(at)
    assert "The assistant could not answer (LLM_PROVIDER_ERROR)." in text
    assert "SECRET" not in text


def test_new_analysis_clears_chat_history():
    fake = _FakeChatClient(["Answer [S1]."])
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
        patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Ask about this filing").click().run()
        at.chat_input[0].set_value("Summarize").run()
        assert len(at.session_state["chat_messages"]) == 2
        _button(at, "Analyze filing").click().run()

    assert not at.exception
    assert "chat_messages" not in at.session_state
    assert not at.chat_input  # the panel closes with the old report


def test_chat_is_a_floating_launcher_not_a_tab():
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        assert not at.chat_input  # closed until the launcher is used
        _button(at, "Ask about this filing").click().run()

        assert not at.exception
        assert len(at.chat_input) == 1
        assert all("Ask" not in tab.label for tab in at.tabs)
        # Panel is not a modal: the report keeps rendering behind it.
        assert "Executive snapshot" in _all_text(at)
        at.button(key="chat_close").click().run()

    assert not at.exception
    assert not at.chat_input
    assert any(button.label == "Ask about this filing" for button in at.button)


def test_chat_markdown_keeps_formatting_but_blocks_latex():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    source = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_chat_markdown"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[source], type_ignores=[]), "x", "exec"), namespace)

    assert namespace["_chat_markdown"]("**Risks:** cash $830M to $910M") == (
        "**Risks:** cash \\$830M to \\$910M"
    )


def test_chat_listen_uses_selected_groq_engine_and_voice():
    fake = _FakeChatClient(["Cash fell ~8.8% [R1]."])
    audio = SimpleNamespace(audio_bytes=b"RIFF-groq")
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart", "ef_dora"]),
        patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake),
        patch("src.api.main.synthesize_groq", return_value=audio) as groq,
        patch("src.api.main.synthesize") as kokoro,
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Ask about this filing").click().run()
        at.chat_input[0].set_value("Risks?").run()
        engine = next(radio for radio in at.radio if radio.label == "Voice engine")
        engine.set_value("groq").run()
        voice = next(box for box in at.selectbox if box.label == "English voice")
        assert voice.options == ["troy", "hannah", "austin", "autumn", "diana", "daniel"]
        voice.set_value("hannah").run()
        _button(at, "Listen").click().run()

    assert not at.exception
    groq.assert_called_once_with("Cash fell about 8.8 percent.", voice="hannah")
    kokoro.assert_not_called()
    assert "Read with Groq (fast) · hannah" in _all_text(at)


def test_spanish_answer_falls_back_to_kokoro_even_with_groq_selected():
    fake = _FakeChatClient(["El efectivo cayó un ~8,8% [R1]."])
    audio = SimpleNamespace(audio_bytes=b"RIFF-kokoro")
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("httpx.stream", side_effect=_stream_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart", "ef_dora"]),
        patch("src.api.main.OpenRouterChatClient.from_env", return_value=fake),
        patch("src.api.main.synthesize_groq") as groq,
        patch("src.api.main.synthesize", return_value=audio) as kokoro,
    ):
        at = _app_test().run()
        _button(at, "Analyze filing").click().run()
        _button(at, "Ask about this filing").click().run()
        at.chat_input[0].set_value("¿Riesgos?").run()
        next(r for r in at.radio if r.label == "Voice engine").set_value("groq").run()
        _button(at, "Listen").click().run()

    assert not at.exception
    groq.assert_not_called()
    kokoro.assert_called_once_with(
        text="El efectivo cayó un aproximadamente 8,8 por ciento.",
        voice="ef_dora",
        language="es",
    )
    assert "Read with Local (Kokoro) · ef_dora (Groq has no Spanish voice)" in _all_text(at)


def _idle_guard_html(timeout, warning=30):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    wanted = {"idle_guard_html", "PAUSED_PAGE", "IDLE_WARNING_SECONDS"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and node.targets[0].id in wanted
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else False)
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "x", "exec"), namespace)
    return namespace["idle_guard_html"](timeout, warning)


def test_idle_guard_pauses_after_timeout_and_warns_before():
    script = _idle_guard_html(300)

    assert "timeoutMs = 300000, warnMs = 30000" in script
    assert 'window.location.replace("app/static/paused.html")' in script
    assert "window.__idleGuard" in script  # installed once per page load
    for event in ("pointerdown", "keydown", "wheel", "touchstart"):
        assert event in script


def test_paused_page_is_served_statically_with_resume_link():
    root = APP_PATH.parents[1]
    page = (APP_PATH.parent / "static" / "paused.html").read_text(encoding="utf-8")

    assert "enableStaticServing = true" in (root / ".streamlit" / "config.toml").read_text()
    assert "Session paused" in page and 'href="/"' in page


def test_idle_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("IDLE_TIMEOUT_SECONDS", "0")
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart"]),
    ):
        at = _app_test().run()

    assert not at.exception
    assert not any("__idleGuard" in str(element.proto) for element in at.get("html"))
