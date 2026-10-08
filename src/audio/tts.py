"""
Text-to-Speech usando Kokoro ONNX (modelo local, 82M params, multilingüe).

Esta capa pertenece a la **capa de modelos IA** del proyecto (ver CLAUDE.md):
expone funciones puras que luego la capa de lógica de negocio (src/api/, Marco)
orquesta vía FastAPI. NO contiene endpoints HTTP.

Modelo: Kokoro v1.0 (ONNX)
  - 82M parámetros, corre en CPU en tiempo razonable.
  - Multilingüe: EN/ES/FR/IT/PT/HI/JA/ZH.
  - Varias voces por idioma.

Nota: el modelo (.onnx + voices.bin) se descarga la primera vez a ~/.cache
o al directorio indicado por KOKORO_MODEL_DIR en .env.

TODO (roadmap): ver src/audio/README.md — está previsto añadir backend
alternativo con servicio externo (OpenAI TTS / ElevenLabs) para acelerar
inferencia en producción manteniendo la misma API pública.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# ---- Config -----------------------------------------------------------------

# Mapa de código de idioma simple → código Kokoro
LANG_MAP: dict[str, str] = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt-br",
    "ja": "ja",
    "zh": "zh",
    "hi": "hi",
}

# Voz por defecto por idioma. (Ver kokoro_onnx.Kokoro().get_voices() para lista completa.)
DEFAULT_VOICE_BY_LANG: dict[str, str] = {
    "en-us": "af_heart",   # femenina EN-US, natural
    "en-gb": "bf_emma",    # femenina EN-GB
    "es": "ef_dora",       # femenina ES
    "fr-fr": "ff_siwis",
    "it": "if_sara",
    "pt-br": "pf_dora",
    "ja": "jf_alpha",
    "zh": "zf_xiaoxiao",
    "hi": "hf_alpha",
}

# Convención Kokoro: primera letra de la voz = idioma
# (a=en-us, b=en-gb, e=es, f=fr, i=it, p=pt, j=ja, z=zh, h=hi).
_VOICE_PREFIX_BY_LANG: dict[str, str] = {
    "en-us": "a", "en-gb": "b", "es": "e", "fr-fr": "f",
    "it": "i", "pt-br": "p", "ja": "j", "zh": "z", "hi": "h",
}

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

DEFAULT_MODEL_DIR = Path(
    os.getenv("KOKORO_MODEL_DIR", Path.home() / ".cache" / "kokoro")
)


# ---- Modelos de datos -------------------------------------------------------


@dataclass
class SynthesisResult:
    """Resultado de una síntesis TTS."""

    audio_bytes: bytes          # WAV listo para escribir en disco o enviar por red
    sample_rate: int            # típicamente 24000 Hz
    duration: float             # segundos
    voice: str                  # voz usada
    language: str               # idioma usado (código Kokoro)

    def save(self, path: str | Path) -> Path:
        """Guarda el audio en disco. Devuelve la ruta final."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.audio_bytes)
        return out


# ---- Cliente singleton ------------------------------------------------------

_kokoro = None  # type: ignore[var-annotated]


def _ensure_model_files() -> tuple[Path, Path]:
    """Descarga el modelo y voces si no existen localmente."""
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DEFAULT_MODEL_DIR / "kokoro-v1.0.onnx"
    voices_path = DEFAULT_MODEL_DIR / "voices-v1.0.bin"

    if not model_path.exists():
        print(f"[kokoro] Descargando modelo a {model_path} (~325 MB, solo la primera vez)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    if not voices_path.exists():
        print(f"[kokoro] Descargando voces a {voices_path} (~27 MB)...")
        urllib.request.urlretrieve(VOICES_URL, voices_path)

    return model_path, voices_path


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise RuntimeError(
                "kokoro-onnx no está instalado. Ejecuta: pip install kokoro-onnx soundfile"
            ) from e

        import onnxruntime as ort

        model_path, voices_path = _ensure_model_files()
        # Por defecto ONNX Runtime lanza tantos hilos como cores tenga el HOST;
        # en un contenedor limitado (Cloud Run: 2 vCPU) eso provoca throttling
        # y la síntesis tarda ~9x más. Ajustamos los hilos a la cuota real.
        threads, source = _onnx_threads_with_source()
        # Visible en los logs de Cloud Run: confirma cuántas CPUs detecta.
        print(f"[kokoro] ONNX threads={threads} (source: {source})", flush=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        _kokoro = Kokoro.from_session(session, str(voices_path))
    return _kokoro


def _onnx_threads() -> int:
    """Hilos ONNX = CPUs disponibles según la cuota cgroup (o KOKORO_THREADS)."""
    return _onnx_threads_with_source()[0]


def _onnx_threads_with_source() -> tuple[int, str]:
    override = os.getenv("KOKORO_THREADS")
    if override and override.isdigit() and int(override) > 0:
        return int(override), "env"
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period))), "cgroup"
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0))), "affinity"
    return max(1, os.cpu_count() or 1), "cpu_count"


# ---- API pública ------------------------------------------------------------


def synthesize(
    text: str,
    *,
    language: str = "en",
    voice: str | None = None,
    speed: float = 1.0,
    output_format: Literal["wav", "mp3"] = "wav",
) -> SynthesisResult:
    """
    Convierte texto a audio hablado.

    Args:
        text: texto a sintetizar.
        language: 'en', 'es', 'fr', 'it', 'pt', 'ja', 'zh', 'hi' (o códigos extendidos).
        voice: nombre de voz específica. Si None, usa la default del idioma.
        speed: 0.5 - 2.0 (1.0 = natural).
        output_format: 'wav' (sin pérdida) o 'mp3' (requiere pydub + ffmpeg).

    Returns:
        SynthesisResult con bytes de audio y metadatos.
    """
    if not text.strip():
        raise ValueError("El texto no puede estar vacío.")

    lang_code = LANG_MAP.get(language.lower())
    if lang_code is None and language.lower() in LANG_MAP.values():
        lang_code = language.lower()  # ya es un código Kokoro ('fr-fr', 'pt-br')
    if lang_code is None:
        raise ValueError(
            f"Idioma no soportado: {language}. Soportados: {sorted(set(LANG_MAP.values()))}"
        )

    chosen_voice = voice or DEFAULT_VOICE_BY_LANG[lang_code]

    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(
        text, voice=chosen_voice, speed=speed, lang=lang_code
    )

    # Serializar a WAV en memoria
    import io
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    audio_bytes = buf.getvalue()

    if output_format == "mp3":
        audio_bytes = _wav_to_mp3(audio_bytes)

    return SynthesisResult(
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        duration=len(samples) / sample_rate,
        voice=chosen_voice,
        language=lang_code,
    )


def list_voices(language: str | None = None) -> list[str]:
    """Lista voces disponibles, opcionalmente filtradas por idioma."""
    kokoro = _get_kokoro()
    voices = list(kokoro.get_voices())
    if language is None:
        return voices
    lang_code = LANG_MAP.get(language.lower(), language)
    prefix = _VOICE_PREFIX_BY_LANG.get(lang_code)
    if not prefix:
        return voices
    return [v for v in voices if v.startswith(prefix)]


def language_for_voice(voice: str | None, default: str = "en-us") -> str:
    """Idioma Kokoro que corresponde a una voz según su prefijo (ef_dora → es).

    Sin esto, una voz española sintetizada con lang='en' se pronuncia en inglés.
    """
    if not voice:
        return default
    for lang_code, prefix in _VOICE_PREFIX_BY_LANG.items():
        if voice.startswith(prefix):
            return lang_code
    return default


# ---- Helpers internos -------------------------------------------------------


def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convierte WAV → MP3 vía pydub (requiere ffmpeg en PATH)."""
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError("pydub no instalado. pip install pydub") from e

    import io
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    out = io.BytesIO()
    audio.export(out, format="mp3", bitrate="128k")
    return out.getvalue()
