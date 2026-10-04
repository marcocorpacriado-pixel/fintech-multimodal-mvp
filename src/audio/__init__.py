"""
Módulo de audio (Cristian).

Expone funciones puras de STT y TTS que la capa de API/UI consumirá.
Ver README.md para arquitectura y roadmap.
"""

from src.audio.stt import Transcription, Segment, transcribe
from src.audio.tts import SynthesisResult, synthesize, list_voices

__all__ = [
    "transcribe",
    "synthesize",
    "list_voices",
    "Transcription",
    "Segment",
    "SynthesisResult",
]
