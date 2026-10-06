"""
Test manual del módulo src/audio/.

Round-trip:
  1. Sintetiza una frase en EN y otra en ES (TTS Kokoro).
  2. Guarda ambos WAV en data/tmp/.
  3. Vuelve a transcribirlos (STT Groq).
  4. Imprime texto original vs. transcrito.

Uso:
    python scripts/test_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir ejecutar desde cualquier cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio import synthesize, transcribe  # noqa: E402

TMP_DIR = ROOT / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    (
        "en",
        "In the fourth quarter, Apple reported record revenue of ninety "
        "billion dollars, driven by strong iPhone sales in emerging markets.",
    ),
    (
        "es",
        "En el cuarto trimestre, Apple reportó ingresos récord de noventa "
        "mil millones de dólares, impulsados por fuertes ventas de iPhone.",
    ),
]


def main() -> None:
    print("=" * 70)
    print("TEST ROUND-TRIP: TTS Kokoro → STT Groq Whisper")
    print("=" * 70)

    for lang, text in SAMPLES:
        print(f"\n--- Idioma: {lang.upper()} ---")
        print(f"Texto original: {text}")

        # 1) Sintetizar
        print("[1/2] Sintetizando con Kokoro...")
        tts_result = synthesize(text, language=lang)
        out_path = TMP_DIR / f"test_{lang}.wav"
        tts_result.save(out_path)
        print(f"      ✓ Audio guardado: {out_path}")
        print(
            f"      Duración: {tts_result.duration:.2f}s | "
            f"Voz: {tts_result.voice} | SR: {tts_result.sample_rate} Hz"
        )

        # 2) Transcribir de vuelta
        print("[2/2] Transcribiendo con Groq Whisper...")
        stt_result = transcribe(out_path, language=lang)
        print(f"      ✓ Texto transcrito: {stt_result.text}")
        print(
            f"      Idioma detectado: {stt_result.language} | "
            f"Duración: {stt_result.duration:.2f}s | "
            f"Segmentos: {len(stt_result.segments)}"
        )

    print("\n" + "=" * 70)
    print("✓ Test completado.")
    print(f"Audios de prueba en: {TMP_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
