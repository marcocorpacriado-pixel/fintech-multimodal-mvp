"""
Normalización de texto antes de TTS (Kokoro y Groq).

Los modelos TTS leen literalmente lo que reciben: con Markdown y símbolos
financieros Kokoro dice "asterisk asterisk", "tilde", "Coco" (por QoQ) o
pega las etiquetas de cita ("$1.25BM1"). Groq/Orpheus además interpreta
cualquier ``[...]`` como dirección de actuación. Este módulo convierte el texto
del chat/resumen en prosa hablable, sin tocar el valor de las cifras.

Función pura, sin dependencias: se aplica en la API para ambos proveedores.
"""

from __future__ import annotations

import re
from typing import Literal

SpeechLanguage = Literal["en", "es"]

_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "percent": "percent", "dollars": "dollars", "plus": "plus", "minus": "minus",
        "about": "about", "and": "and", "versus": "versus",
        "K": "thousand", "M": "million", "B": "billion", "T": "trillion",
        "QoQ": "quarter over quarter", "YoY": "year over year", "YTD": "year to date",
        "Q1": "first quarter", "Q2": "second quarter", "Q3": "third quarter",
        "Q4": "fourth quarter",
    },
    "es": {
        "percent": "por ciento", "dollars": "dólares", "plus": "más", "minus": "menos",
        "about": "aproximadamente", "and": "y", "versus": "frente a",
        "K": "mil", "M": "millones de", "B": "mil millones de", "T": "billones de",
        "QoQ": "trimestre contra trimestre", "YoY": "interanual",
        "YTD": "en lo que va de año",
        "Q1": "primer trimestre", "Q2": "segundo trimestre", "Q3": "tercer trimestre",
        "Q4": "cuarto trimestre",
    },
}

_NUMBER = r"\d[\d,.]*\d|\d"
_BRACKETS = re.compile(r"\s*\[[^\]\n]{0,40}\]")  # citas [M1] y direcciones [x]
_MONEY = re.compile(rf"\$\s?({_NUMBER})(?:\s?([KMBT])\b)?")
_PERCENT = re.compile(rf"({_NUMBER})\s?%")
_SIGN = re.compile(rf"(^|[\s(])([+\-−])(?=\d)")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)


def _money(match: re.Match[str], words: dict[str, str]) -> str:
    number, scale = match.group(1), match.group(2)
    if scale:  # "1.25 billion dollars" / "1.25 mil millones de dólares"
        return f"{number} {words[scale]} {words['dollars']}"
    return f"{number} {words['dollars']}"


_ES_MARKERS = re.compile(r"[áéíóúñ¿¡]", re.IGNORECASE)
_ES_WORDS = frozenset(
    "el la los las de del que y en un una por para con se al es son está están "
    "más pero como sus su lo le ha han fue sobre entre también muy este esta "
    "ingresos efectivo análisis riesgos trimestre".split()
)
_EN_WORDS = frozenset(
    "the and of to in is are was were with for that this on by from it its as "
    "has have be not but revenue cash analysis risks quarter".split()
)


def detect_speech_language(text: str, default: SpeechLanguage = "en") -> SpeechLanguage:
    """Idioma (en/es) de un texto corto, por palabras frecuentes y acentos.

    Heurística sin dependencias: suficiente para elegir voz/motor TTS de una
    respuesta del chat, que siempre está en uno de los dos idiomas. Si no hay
    señales (p. ej. "Hola"), devuelve ``default``.
    """

    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    spanish = sum(word in _ES_WORDS for word in words) + 2 * len(_ES_MARKERS.findall(text))
    english = sum(word in _EN_WORDS for word in words)
    if spanish == english:
        return default
    return "es" if spanish > english else "en"


def normalize_for_speech(text: str, language: SpeechLanguage = "en") -> str:
    """Convierte Markdown/símbolos en texto hablable para un motor TTS."""

    words = _WORDS["es" if language == "es" else "en"]
    out = text.replace("[stream interrupted]", "")
    out = _BRACKETS.sub("", out)

    # Markdown: encabezados, viñetas, énfasis y código.
    out = _HEADING.sub("", out)
    out = _LIST_MARKER.sub("", out)
    out = re.sub(r"(\*\*|__|\*|`)", "", out)
    out = re.sub(r"(?<!\w)_(?!\w)", " ", out)

    # Abreviaturas financieras (palabra completa, sensibles a mayúsculas).
    for abbr in ("QoQ", "YoY", "YTD", "Q1", "Q2", "Q3", "Q4"):
        out = re.sub(rf"\b{abbr}\b", words[abbr], out)
    out = re.sub(r"\bvs\.?(?=\s)", words["versus"], out)

    # Cifras: dinero, porcentajes, signos y aproximaciones.
    out = _MONEY.sub(lambda m: _money(m, words), out)
    out = _PERCENT.sub(lambda m: f"{m.group(1)} {words['percent']}", out)
    out = _SIGN.sub(
        lambda m: f"{m.group(1)}{words['plus'] if m.group(2) == '+' else words['minus']} ",
        out,
    )
    out = re.sub(r"~\s?", f"{words['about']} ", out)
    out = out.replace("&", f" {words['and']} ")

    # Puntuación que se lee mal: paréntesis, rayas, punto y coma.
    out = re.sub(r"\s*\(([^)]*)\)", r", \1,", out)
    out = re.sub(r"\s*[—–]\s*", ", ", out)
    out = out.replace(";", ",")

    # Saltos de línea → pausas; limpiar comas y espacios repetidos.
    out = re.sub(r"(?<![.!?:,])\s*\n+\s*", ". ", out)
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"\s+([,.!?:])", r"\1", out)
    out = re.sub(r",\s*([,.!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(" ,")


__all__ = ["SpeechLanguage", "detect_speech_language", "normalize_for_speech"]
