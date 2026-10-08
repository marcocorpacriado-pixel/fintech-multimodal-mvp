"""
Text-to-Speech vía Groq (Orpheus, canopylabs/orpheus-v1-english).

Alternativa rápida a Kokoro local: la síntesis corre en Groq, no en la CPU de
Cloud Run. Restricciones del modelo (docs Groq):
  - Solo inglés.
  - Máximo 200 caracteres por petición → el texto se trocea por frases y los
    trozos se sintetizan en paralelo y se concatenan en un único WAV.
  - ``[...]`` se interpreta como dirección de actuación ([cheerful]); el texto
    debe llegar ya normalizado (ver ``speech_text.normalize_for_speech``).

Requiere GROQ_API_KEY (la misma que STT) y que el admin de la org acepte los
términos del modelo en la consola de Groq.
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from src.audio.stt import _get_client
from src.audio.tts import SynthesisResult

GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
GROQ_VOICES = ["troy", "hannah", "austin", "autumn", "diana", "daniel"]
DEFAULT_GROQ_VOICE = "troy"
MAX_CHARS_PER_REQUEST = 200
MAX_PARALLEL_REQUESTS = 4
_PAUSE_SECONDS = 0.12  # silencio entre trozos para que no suene pegado

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,:])\s+")


def _wrap_words(text: str, limit: int) -> list[str]:
    chunks, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word[:limit]
    if current:
        chunks.append(current)
    return chunks


def split_for_groq(text: str, limit: int = MAX_CHARS_PER_REQUEST) -> list[str]:
    """Trocea en piezas ≤ ``limit`` respetando frases, luego cláusulas, luego palabras."""

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if not sentence:
            continue
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_END.split(sentence):
            pieces.extend([clause] if len(clause) <= limit else _wrap_words(clause, limit))

    # Reagrupar frases cortas para minimizar peticiones.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + 1 + len(piece) <= limit:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _synthesize_chunk(text: str, voice: str) -> tuple[np.ndarray, int]:
    response = _get_client().audio.speech.create(
        model=GROQ_TTS_MODEL,
        voice=voice,
        input=text,
        response_format="wav",
    )
    samples, sample_rate = sf.read(io.BytesIO(response.read()), dtype="float32")
    return samples, sample_rate


def synthesize_groq(text: str, *, voice: str | None = None) -> SynthesisResult:
    """Sintetiza ``text`` (inglés, ya normalizado) con Groq Orpheus."""

    chosen_voice = voice or DEFAULT_GROQ_VOICE
    if chosen_voice not in GROQ_VOICES:
        raise ValueError(f"Voz Groq no soportada: {chosen_voice}")
    chunks = split_for_groq(text)
    if not chunks:
        raise ValueError("El texto no puede estar vacío.")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as pool:
        parts = list(pool.map(lambda chunk: _synthesize_chunk(chunk, chosen_voice), chunks))

    sample_rate = parts[0][1]
    if any(rate != sample_rate for _, rate in parts):
        raise RuntimeError("Groq devolvió trozos con sample rates distintos")
    pause = np.zeros(int(sample_rate * _PAUSE_SECONDS), dtype="float32")
    audio = np.concatenate(
        [segment for samples, _ in parts for segment in (samples, pause)][:-1]
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return SynthesisResult(
        audio_bytes=buffer.getvalue(),
        sample_rate=sample_rate,
        duration=len(audio) / sample_rate,
        voice=chosen_voice,
        language="en-us",
    )


__all__ = [
    "DEFAULT_GROQ_VOICE",
    "GROQ_TTS_MODEL",
    "GROQ_VOICES",
    "split_for_groq",
    "synthesize_groq",
]
