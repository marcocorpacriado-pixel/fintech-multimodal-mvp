"""
Módulo de audio (Cristian).

Expone funciones puras de STT y TTS que la capa de API/UI consumirá.
Ver README.md para arquitectura y roadmap.
"""

from src.audio.stt import Transcription, Segment, transcribe
from src.audio.tts import SynthesisResult, language_for_voice, list_voices, synthesize
from src.audio.groq_tts import DEFAULT_GROQ_VOICE, GROQ_VOICES, synthesize_groq
from src.audio.speech_text import detect_speech_language, normalize_for_speech

__all__ = [
    "transcribe",
    "synthesize",
    "list_voices",
    "language_for_voice",
    "synthesize_groq",
    "GROQ_VOICES",
    "DEFAULT_GROQ_VOICE",
    "normalize_for_speech",
    "detect_speech_language",
    "Transcription",
    "Segment",
    "SynthesisResult",
]
