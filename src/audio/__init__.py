"""
Módulo de audio (Cristian).

Expone funciones puras de STT y TTS que la capa de API/UI consumirá.
Ver README.md para arquitectura y roadmap.
"""

from src.audio.stt import Transcription, Segment, transcribe
from src.audio.tts import SynthesisResult, language_for_voice, list_voices, synthesize

__all__ = [
    "transcribe",
    "synthesize",
    "list_voices",
    "language_for_voice",
    "Transcription",
    "Segment",
    "SynthesisResult",
]
