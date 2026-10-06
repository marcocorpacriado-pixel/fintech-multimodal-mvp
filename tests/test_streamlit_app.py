"""Smoke tests: the Streamlit app driven against the real API in-process."""

import ast
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api.main import app as api_app
from src.extraction import LLMTransportError

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
api_client = TestClient(api_app)


def _route_to_test_client(method, url, timeout=None, **kwargs):
    return api_client.request(method, url.removeprefix("http://localhost:8000"), **kwargs)


@pytest.fixture
def app_test():
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


def test_api_error_is_shown_without_internals(app_test):
    at = app_test.run()
    at.sidebar.radio[0].set_value("real").run()
    at.sidebar.date_input[0].set_value(date(2026, 7, 31)).run()
    with patch(
        "src.api.main._run_real_analysis",
        side_effect=LLMTransportError("body with sk-or-SECRET"),
    ):
        at.sidebar.button[0].click().run()

    assert not at.exception
    assert "LLM_PROVIDER_ERROR" in at.error[0].value
    assert "SECRET" not in _all_text(at)


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
