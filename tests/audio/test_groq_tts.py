"""Groq Orpheus TTS: chunking to the 200-char limit and WAV concatenation."""

import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from src.audio import groq_tts
from src.audio.groq_tts import split_for_groq, synthesize_groq


def _wav(seconds: float, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(rate * seconds), dtype="float32"), rate, format="WAV")
    return buffer.getvalue()


class _FakeSpeech:
    def __init__(self):
        self.inputs = []

    def create(self, *, model, voice, input, response_format):
        assert model == groq_tts.GROQ_TTS_MODEL and response_format == "wav"
        self.inputs.append(input)
        return SimpleNamespace(read=lambda: _wav(0.5))


def test_split_respects_limit_and_keeps_all_words():
    text = ("Revenue rose about six percent. " * 12) + ("word " * 80)
    chunks = split_for_groq(text)

    assert all(len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_short_sentences_are_merged_into_one_request():
    assert split_for_groq("One. Two. Three.") == ["One. Two. Three."]


def test_synthesize_concatenates_chunks_in_order():
    speech = _FakeSpeech()
    client = SimpleNamespace(audio=SimpleNamespace(speech=speech))
    text = "First sentence is here. " * 10 + "Last one."

    with patch("src.audio.groq_tts._get_client", return_value=client):
        result = synthesize_groq(text, voice="hannah")

    chunks = split_for_groq(text)
    assert sorted(speech.inputs) == sorted(chunks)
    assert result.voice == "hannah" and result.language == "en-us"
    pauses = (len(chunks) - 1) * groq_tts._PAUSE_SECONDS
    assert result.duration == pytest.approx(0.5 * len(chunks) + pauses, abs=0.01)
    samples, rate = sf.read(io.BytesIO(result.audio_bytes))
    assert rate == 24000 and len(samples) / rate == pytest.approx(result.duration)


def test_rejects_unknown_voice_and_empty_text():
    with pytest.raises(ValueError):
        synthesize_groq("Hello", voice="af_heart")
    with pytest.raises(ValueError):
        synthesize_groq("   ")
