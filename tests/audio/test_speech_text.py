"""Text normalization before TTS: no symbols, tags or Markdown read aloud."""

import pytest

from src.audio.speech_text import normalize_for_speech


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        (
            "**Main risks:** Cash fell ~8.8% (to $830M) [R1].",
            "en",
            "Main risks: Cash fell about 8.8 percent, to 830 million dollars.",
        ),
        (
            "Revenue rose +5.9% QoQ to $1.25B [M1]; EPS $1.42 vs. $1.51 — down -5.96%.",
            "en",
            "Revenue rose plus 5.9 percent quarter over quarter to 1.25 billion "
            "dollars, EPS 1.42 dollars versus 1.51 dollars, down minus 5.96 percent.",
        ),
        (
            "**Riesgos:** El efectivo cayó un ~8,8% (a $830M) [R1].",
            "es",
            "Riesgos: El efectivo cayó un aproximadamente 8,8 por ciento, "
            "a 830 millones de dólares.",
        ),
        (
            "Net income fell 6.67% YoY & YTD cash flow grew in Q2.",
            "en",
            "Net income fell 6.67 percent year over year and year to date cash flow "
            "grew in second quarter.",
        ),
    ],
)
def test_normalizes_financial_markdown(text, language, expected):
    assert normalize_for_speech(text, language) == expected


def test_lists_and_newlines_become_pauses_and_markers_are_dropped():
    text = "**Key points:**\n- Liquidity [R1]\n- Debt rose\n\n[stream interrupted]"

    assert normalize_for_speech(text) == "Key points: Liquidity. Debt rose."


def test_no_brackets_survive_for_groq_vocal_directions():
    out = normalize_for_speech("Strong demand [cheerful] and growth [M3] [S1].")

    assert "[" not in out and "]" not in out
    assert out == "Strong demand and growth."


def test_plain_text_is_unchanged():
    assert normalize_for_speech("Hello there.") == "Hello there."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue rose about 5.9 percent and cash fell [R1].", "en"),
        ("Los ingresos subieron un 5,9% y el efectivo cayó [R1].", "es"),
        ("¿Cuáles son los riesgos?", "es"),
        ("Management expects continued demand for subscription services.", "en"),
    ],
)
def test_detects_answer_language(text, expected):
    from src.audio.speech_text import detect_speech_language

    assert detect_speech_language(text) == expected


def test_undetectable_text_uses_default():
    from src.audio.speech_text import detect_speech_language

    assert detect_speech_language("Hola", default="es") == "es"
    assert detect_speech_language("OK", default="en") == "en"
