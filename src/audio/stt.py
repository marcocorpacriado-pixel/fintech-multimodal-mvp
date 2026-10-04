"""
Speech-to-Text usando Groq API (whisper-large-v3-turbo).

Esta capa pertenece a la **capa de modelos IA** del proyecto (ver CLAUDE.md):
expone funciones puras que luego la capa de lógica de negocio (src/api/, Marco)
orquesta vía FastAPI. NO contiene endpoints HTTP.

Modelo: whisper-large-v3-turbo
  - Detecta idioma automáticamente (ES/EN y 97 más).
  - ~216x realtime en Groq (una call de 1h se transcribe en segundos).
  - Devuelve segmentos con timestamps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---- Config -----------------------------------------------------------------

DEFAULT_MODEL = "whisper-large-v3-turbo"
# Alternativa de mayor calidad (más lenta y cara): "whisper-large-v3"
# Soporte multilingüe idéntico. Para earnings calls en inglés "turbo" basta.

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpga"}
MAX_FILE_SIZE_MB = 25  # Límite Groq (free tier). Dividir antes si supera.


# ---- Modelos de datos -------------------------------------------------------


@dataclass
class Segment:
    """Un segmento temporal de la transcripción."""

    start: float  # segundos
    end: float
    text: str


@dataclass
class Transcription:
    """Resultado de una transcripción."""

    text: str
    language: str  # ISO-639-1 detectado (ej. 'en', 'es')
    duration: float  # segundos
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in self.segments
            ],
        }


# ---- Cliente singleton ------------------------------------------------------

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta GROQ_API_KEY en el entorno. Añádela a .env en la raíz del proyecto."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---- API pública ------------------------------------------------------------


def transcribe(
    audio: str | Path | bytes,
    *,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    temperature: float = 0.0,
    response_format: Literal["verbose_json", "json", "text"] = "verbose_json",
) -> Transcription:
    """
    Transcribe un audio a texto usando Groq Whisper.

    Args:
        audio: ruta al archivo, Path o bytes del audio.
        language: código ISO-639-1 ('es', 'en'). Si es None, auto-detecta.
                  Pasarlo explícitamente mejora precisión y latencia.
        model: por defecto 'whisper-large-v3-turbo'.
        prompt: contexto opcional (jerga, nombres propios) para mejorar precisión.
                Útil para earnings calls: p.ej. "Apple Inc, Tim Cook, revenue, EBITDA".
        temperature: 0.0 = determinista (recomendado para transcripción).
        response_format: 'verbose_json' incluye segmentos y timestamps.

    Returns:
        Transcription con texto, idioma detectado, duración y segmentos.

    Raises:
        FileNotFoundError, ValueError, RuntimeError
    """
    audio_bytes, filename = _load_audio(audio)
    _validate_size(audio_bytes)

    client = _get_client()

    kwargs: dict = {
        "file": (filename, audio_bytes),
        "model": model,
        "response_format": response_format,
        "temperature": temperature,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    if response_format == "verbose_json":
        kwargs["timestamp_granularities"] = ["segment"]

    response = client.audio.transcriptions.create(**kwargs)

    # Groq devuelve dict-like en verbose_json con atributos text/language/duration/segments
    if response_format == "verbose_json":
        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in getattr(response, "segments", []) or []
        ]
        return Transcription(
            text=response.text.strip(),
            language=getattr(response, "language", language or "unknown"),
            duration=getattr(response, "duration", 0.0),
            segments=segments,
        )

    # json / text: devolver estructura mínima
    text = response if isinstance(response, str) else response.text
    return Transcription(
        text=text.strip(),
        language=language or "unknown",
        duration=0.0,
        segments=[],
    )


# ---- Helpers internos -------------------------------------------------------


def _load_audio(audio: str | Path | bytes) -> tuple[bytes, str]:
    """Normaliza la entrada a (bytes, filename). Valida extensión si es ruta."""
    if isinstance(audio, bytes):
        return audio, "audio.wav"  # Groq usa la extensión para hintar formato

    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Formato no soportado: {path.suffix}. "
            f"Soportados: {sorted(SUPPORTED_FORMATS)}"
        )
    return path.read_bytes(), path.name


def _validate_size(data: bytes) -> None:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Archivo demasiado grande ({size_mb:.1f} MB). "
            f"Límite Groq: {MAX_FILE_SIZE_MB} MB. "
            f"TODO: implementar split automático en chunks."
        )
