"""ONNX threads follow the container CPU quota, not the host core count."""

from pathlib import Path
from unittest.mock import patch

from src.audio import tts


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("KOKORO_THREADS", "3")

    assert tts._onnx_threads() == 3


def test_cgroup_quota_sets_threads(monkeypatch):
    monkeypatch.delenv("KOKORO_THREADS", raising=False)
    with patch.object(Path, "read_text", return_value="200000 100000\n"):
        assert tts._onnx_threads() == 2


def test_unlimited_quota_falls_back_to_available_cpus(monkeypatch):
    monkeypatch.delenv("KOKORO_THREADS", raising=False)
    with (
        patch.object(Path, "read_text", return_value="max 100000\n"),
        patch("os.sched_getaffinity", return_value={0, 1, 2, 3}, create=True),
    ):
        assert tts._onnx_threads() == 4
