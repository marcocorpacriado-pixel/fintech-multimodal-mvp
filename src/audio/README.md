# `src/audio/` — módulo de audio

Capa de modelos para **Speech-to-Text (STT)** y **Text-to-Speech (TTS)**. Expone
funciones puras que FastAPI orquesta; el módulo no define endpoints HTTP ni
interpreta información financiera.

## Alcance actual

- **STT:** Groq `whisper-large-v3-turbo` convierte voz del usuario en una
  pregunta para filing chat.
- **TTS local (default):** Kokoro v1.0 ONNX 82M genera audio del executive
  summary o de una respuesta de chat.
- **TTS remoto opcional:** Groq
  `canopylabs/orpheus-v1-english` sintetiza respuestas en inglés cuando el
  usuario selecciona ese motor.
- **Fuera de alcance actual:** el audio de earnings calls no alimenta el
  loader, BM25 ni el pipeline de análisis SEC. Esa integración es futura.

## API pública de Python

```python
from src.audio import list_voices, synthesize, synthesize_groq, transcribe

transcription = transcribe("question.wav", language="en")
print(transcription.text)

speech = synthesize("Revenue increased in the period.", language="en")
speech.save("summary.wav")

remote_speech = synthesize_groq(
    "Revenue increased in the period.", voice="troy"
)

print(list_voices("es"))
```

`transcribe(...)` acepta una ruta, `Path` o bytes y devuelve texto, idioma,
duración y segmentos cuando el proveedor los facilita. `synthesize(...)` usa
Kokoro local. `synthesize_groq(...)` usa Orpheus remoto y requiere texto en
inglés previamente normalizado. Ambos devuelven `SynthesisResult`.

## API HTTP integrada

La API oficial vive en `src/api/main.py`:

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/api/v1/audio/voices?provider=local|groq` | Voces del proveedor seleccionado |
| `POST` | `/api/v1/audio/summary` | Texto → audio WAV mediante `provider=local|groq` |
| `POST` | `/api/v1/audio/transcribe` | Cuerpo de audio crudo → texto para chat |
| `POST` | `/api/v1/chat` | Chat contextual en streaming; no pertenece al módulo de audio |

`provider="local"` es el default. La UI deja elegir el motor para respuestas
de chat y envía una voz compatible. El backend normaliza Markdown, citas y
símbolos financieros antes de ambos motores. Si detecta español, usa Kokoro y
una voz española incluso cuando se solicitó Groq, porque el modelo Orpheus
integrado solo admite inglés. Los headers `X-TTS-Provider` y `X-TTS-Voice`
indican qué motor y voz se usaron realmente.

Este routing por idioma es explícito; no es un fallback ante fallos. Si Groq
falla, la API devuelve un error 503 seguro y no repite la síntesis con Kokoro.
No existe un endpoint temporal `scripts/dev_api.py` ni un flujo de mock LLM
que forme parte del producto actual.

## Configuración

```text
GROQ_API_KEY=<configure-in-environment>   # Obligatoria para STT y Groq TTS
KOKORO_MODEL_DIR=<optional-local-path>    # Caché/assets locales de Kokoro
KOKORO_THREADS=<optional-positive-int>    # Override de hilos ONNX
```

No se deben versionar `.env`, credenciales, grabaciones ni transcripciones
personales.

En desarrollo local, la primera llamada Kokoro puede descargar el modelo y
las voces si no están presentes en `KOKORO_MODEL_DIR`. La imagen Docker los
incorpora durante el build, por lo que Kokoro funciona localmente/offline en
runtime después de construir la imagen. STT y Groq TTS requieren acceso al
proveedor y comparten `GROQ_API_KEY`. Cuando se elige Groq TTS, el texto
normalizado se envía al proveedor externo; Kokoro mantiene la síntesis local.

## Formatos y límites

- STT acepta los formatos soportados por Groq a través de la API de Python. El
  endpoint integrado recibe un cuerpo de audio crudo y limita la carga a
  25 MB.
- TTS produce WAV. La conversión MP3 de la API Python Kokoro requiere `pydub`
  y FFmpeg.
- Groq TTS admite inglés, limita cada petición a 200 caracteres y el módulo
  divide el texto por frases/cláusulas. Ejecuta como máximo cuatro chunks en
  paralelo y concatena los WAV con pausas breves.
- Audios de más de 25 MB necesitan preprocesamiento o chunking, que no está
  implementado en el flujo actual.

## Validación

Los contratos HTTP, selección/routing de proveedor, errores seguros y el flujo
Dash están cubiertos por `tests/test_api.py` y
`tests/test_dash_app.py`. Los tests específicos de audio cubren chunking
Groq, concatenación WAV, normalización hablable y selección de hilos Kokoro.
No realizan llamadas pagadas.

El script siguiente es un smoke manual Kokoro → Groq STT con servicios/modelos
reales:

```powershell
python scripts/test_audio.py
```

El script escribe WAV temporales en `data/tmp/` y requiere `GROQ_API_KEY`; no
forma parte de la suite offline ordinaria.

## Decisiones de arquitectura

- Clientes/modelos singleton perezosos evitan reinicialización por llamada.
- Kokoro ajusta los hilos ONNX a la cuota cgroup/CPU, salvo override explícito.
- Groq TTS reutiliza el cliente autenticado de STT y limita el paralelismo.
- `Transcription` y `SynthesisResult` son contratos tipados, no diccionarios
  ad hoc.
- FastAPI es la frontera HTTP y Dash consume esa API.
- El módulo no calcula métricas, no llama al pipeline financiero y no conserva
  historial de chat.
