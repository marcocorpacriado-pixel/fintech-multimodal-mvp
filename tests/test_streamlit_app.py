"""Smoke tests: the Streamlit app driven against the real API in-process."""

import ast
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.api.main import app as api_app
from src.extraction import LLMTransportError

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
api_client = TestClient(api_app)


def _route_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.request(method, url.removeprefix("http://localhost:8000"), **kwargs)


@pytest.fixture
def app_test():
    st.cache_data.clear()
    with (
        patch("httpx.request", side_effect=_route_to_test_client),
        patch("src.api.main.list_voices", return_value=["af_heart", "bf_emma"]),
    ):
        yield AppTest.from_file(str(APP_PATH), default_timeout=30)


def _all_text(at: AppTest) -> str:
    return " ".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "error", "success", "title")
        for element in getattr(at, kind)
    )


def test_demo_analysis_renders_dashboard(app_test):
    at = app_test.run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    text = _all_text(at)
    assert "DEMO MODE (SYNTHETIC DATA)" in text
    assert "Demo Corp" in text
    assert "SECTION\\_MISMATCH" in text or "SECTION_MISMATCH" in text
    assert len(at.metric) == 7
    assert at.metric[0].value == "$1.25B"
    assert "N/D" in text  # Total Debt has no previous value
    assert [tab.label for tab in at.tabs] == [
        "🎙️ Resumen Ejecutivo & Audio",
        "📊 Desglose Financiero & Gráficos",
        "⚖️ Drivers & Riesgos",
        "🛡️ Compliance & Verificación",
    ]
    assert at.main.selectbox[0].label == "Voz (Kokoro)"  # media card, not sidebar


def test_demo_autoloads_on_first_visit_only(app_test):
    at = app_test.run()

    assert not at.exception
    assert "Demo Corp" in _all_text(at)
    assert len(at.tabs) == 4

    def api_down(method, url, **kwargs):
        raise httpx.ConnectError("API down")

    with patch("httpx.request", side_effect=api_down) as request:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        at.sidebar.radio[0].set_value("real").run()

    assert not at.exception
    assert "handoff" not in at.session_state
    analysis_calls = [c for c in request.call_args_list if c.args[1].endswith("/analysis")]
    assert len(analysis_calls) == 1  # no retry on every rerun


def test_real_mode_selectors_build_payload_from_catalog(app_test):
    at = app_test.run()
    at.sidebar.radio[0].set_value("real").run()

    assert not at.sidebar.date_input
    ticker_box = at.sidebar.selectbox[0]
    nvda = next(i for i, o in enumerate(ticker_box.options) if o.startswith("NVDA"))
    ticker_box.select_index(nvda).run()
    at.sidebar.selectbox[1].set_value("10-K").run()
    filing_box = at.sidebar.selectbox[2]
    assert filing_box.options[0] == "2026-02-25 · FY2026 (cierre 2026-01-25)"

    with patch(
        "src.api.main._run_real_analysis",
        side_effect=LLMTransportError("body with sk-or-SECRET"),
    ) as run_real:
        at.sidebar.button[0].click().run()

    request = run_real.call_args.args[0]
    assert (request.ticker, request.filing_type, request.period) == (
        "NVDA",
        "10-K",
        "2026-02-25",
    )
    assert not at.exception
    assert "LLM_PROVIDER_ERROR" in at.error[0].value
    assert "SECRET" not in _all_text(at)


def test_real_mode_without_catalog_falls_back_safely(app_test):
    def catalog_down(method, url, **kwargs):
        if url.endswith("/api/v1/filings/catalog"):
            raise httpx.ConnectError("API down")
        return _route_to_test_client(method, url, **kwargs)

    with patch("httpx.request", side_effect=catalog_down):
        at = app_test.run()
        at.sidebar.radio[0].set_value("real").run()
        at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.sidebar.selectbox  # no catalog selectors without the API
    assert "catálogo" in at.sidebar.warning[0].value
    assert "handoff" not in at.session_state


def test_app_never_imports_backend_modules():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("src.extraction", "src.integration", "src.audio", "src.api", "edgar")
    assert not [m for m in imported if m and m.startswith(forbidden)]
