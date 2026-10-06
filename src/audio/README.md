# `src/audio/` — Módulo de audio (Cristian)

Capa de modelos IA para **Speech-to-Text** y **Text-to-Speech**.
Pertenece a la **capa de modelos IA** del proyecto (ver `CLAUDE.md`):
expone funciones puras que la capa de API (`src/api/`) orquesta.
**No contiene endpoints HTTP** — eso es responsabilidad de FastAPI.

---

## Casos de uso

El módulo soporta dos flujos:

1. **Voz del usuario ↔ voz del sistema** (core del MVP multimodal).
   - User habla → `transcribe()` → query → agentes → respuesta texto → `synthesize()` → audio.
   - Diferenciador clave vs. un dashboard financiero tradicional.

2. **Transcripción de earnings calls** (opcional, nice-to-have).
   - Audio crudo de una call → `transcribe()` con timestamps → segmentos navegables desde la UI.
   - Nota: las earnings calls ya suelen tener transcripción publicada (Seeking Alpha, Motley Fool).
     El valor aquí está en los *timestamps alineados* y en poder procesar calls recientes
     antes de que exista transcripción oficial.

---

## Stack elegido

| Componente | Elección | Motivo |
|---|---|---|
| **STT** | Groq API (`whisper-large-v3-turbo`) | Multilingüe (97 idiomas), ~216x realtime, free tier generoso. |
| **TTS** | Kokoro ONNX (local, 82M params) | Gratis, suena natural, multilingüe (EN/ES/FR/IT/PT/HI/JA/ZH), corre en CPU. |

### 🔧 Roadmap TTS (pendiente)

Está previsto añadir un **backend TTS alternativo basado en servicio externo** (candidatos: OpenAI TTS, ElevenLabs, Azure Speech) para acelerar la inferencia en producción, sobre todo cuando el servidor no tenga GPU. La API pública (`synthesize(...)`) se mantendrá idéntica; el cambio será seleccionable vía variable de entorno (p.ej. `TTS_BACKEND=kokoro|openai|elevenlabs`).

---

## API pública

```python
from src.audio import transcribe, synthesize, list_voices

# --- STT ---
t = transcribe("call.mp3", language="en")  # language=None para auto-detect
print(t.text)                              # transcripción completa
for seg in t.segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")

# --- TTS ---
result = synthesize("Hello, this is a test.", language="en")
result.save("output.wav")

result_es = synthesize("Hola, esto es una prueba.", language="es")
result_es.save("output_es.wav")

# Listar voces disponibles
print(list_voices("es"))  # voces en español
```

---

## Variables de entorno

Añadir en `.env` (en la raíz del proyecto):

```
GROQ_API_KEY=gsk_...           # Obligatorio para STT. Obtener en https://console.groq.com/
KOKORO_MODEL_DIR=~/.cache/kokoro  # Opcional. Dónde cachear el modelo de TTS (~350 MB total).
```

---

## Instalación

```bash
pip install -r requirements.txt
```

La primera llamada a `synthesize()` **descargará automáticamente** el modelo Kokoro (~325 MB) y el fichero de voces (~27 MB) en `KOKORO_MODEL_DIR`.

---

## Pruebas

```bash
python scripts/test_audio.py
```

Hace un round-trip: sintetiza una frase, la graba a disco, la vuelve a transcribir y compara.

### API de desarrollo (temporal)

Para probar los endpoints vía HTTP antes de que exista la API oficial:

```bash
uvicorn scripts.dev_api:app --reload --port 8000
```

- `http://localhost:8000/` → UI web mínima con formularios para los 3 endpoints.
- `http://localhost:8000/docs` → Swagger UI auto-generada.

Endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/transcribe` | Audio file → JSON con transcripción y segmentos. |
| `POST` | `/synthesize` | Texto + idioma → audio WAV. |
| `POST` | `/voice-chat` | Audio del user → STT → **mock LLM** → TTS → audio de respuesta. |

El mock del LLM en `/voice-chat` se sustituirá por los agentes de `src/extraction/` cuando estén listos (ver el TODO marcado en `scripts/dev_api.py`).

⚠️ `scripts/dev_api.py` es temporal — la API oficial del proyecto vive en `src/api/`.

---

## Formatos soportados

- **STT (entrada)**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`, `.mpga`. Límite: 25 MB por archivo (free tier de Groq).
- **TTS (salida)**: `.wav` nativo. `.mp3` requiere `pydub` + `ffmpeg` en PATH.

---

## Decisiones de arquitectura

- **Singletons perezosos** para el cliente Groq y el modelo Kokoro: evitan reinicializar en cada llamada.
- **Dataclasses** (`Transcription`, `SynthesisResult`) en lugar de dicts crudos: contratos claros con la capa de API.
- **Sin lógica de negocio**: no se interpretan los textos ni se decide qué hacer con ellos.
  Eso corresponde a `src/extraction/` (Dani) y `src/api/` (Marco).
