# CLAUDE.md — contexto operativo del proyecto

## Proyecto

MVP académico FinTech multimodal (Taller B5-T4). Analiza filings SEC 10-K/10-Q
y prepara la incorporación futura de earnings calls. La salida estable es un
análisis financiero estructurado, grounded y verificado, consumible por una
API/UI y por TTS.

## Equipo y límites

- **Dani:** `src/extraction/`, `src/integration/` y sus tests.
- **Cristian:** `src/audio/` y sus tests.
- **Marco:** `src/api/`, `src/visualization/`, `app/` y sus tests.

No modificar módulos de otro responsable ni hacer merges, commits o pushes sin
coordinación explícita. No introducir llamadas a modelos en UI o visualización.

## Capas

1. `src/extraction/`: SEC ingestion, loader, chunking, BM25, XBRL, métricas,
   grounded LLM y verifier.
2. `src/integration/`: DTOs serializables y errores seguros para API/UI/audio.
3. `src/audio/`: STT/TTS de Cristian. El STT existe como módulo, pero aún no
   forma parte del pipeline de análisis.
4. `src/api/`: orquestación FastAPI de Marco (demo explícito, real sin fallback).
5. `src/visualization/` y `app/`: presentación Dash/Plotly de Marco, que
   consume FastAPI solo por HTTP.

En real mode, `filing_date` selecciona la presentación SEC. El `period` del
handoff representa exclusivamente el periodo financiero reportado. El
discovery ligero `GET /api/v1/filings/{ticker}` no ejecuta XBRL ni LLM.

Las cinco capas están integradas en `integration/final-mvp`.

La UI debe consumir `AnalysisHandoff`: nunca recalcula métricas, interpreta
XBRL, ejecuta retrieval o llama directamente al LLM.

## Stack validado

- Python 3.14
- edgartools 5.21.1
- pandas, NumPy y PyArrow
- Pydantic 2
- rank-bm25
- httpx para OpenRouter
- pytest
- Audio: groq, kokoro-onnx, soundfile, pydub, python-dotenv
- API y presentación: FastAPI, Uvicorn, Dash (gunicorn en producción), Plotly

Todas las dependencias están declaradas en `requirements.txt`. No versionar
`.env`, datos SEC, modelos o audio.

## Configuración principal

- `EDGAR_IDENTITY`: SEC real.
- `OPENROUTER_API_KEY` y `OPENROUTER_MODEL`: análisis real.
- `GROQ_API_KEY`: STT (módulo `src/audio/stt.py`).
- `KOKORO_MODEL_DIR`: caché TTS opcional.

Nunca imprimir secretos, headers de autorización, prompts completos o filings.

## Comandos de verificación

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m compileall src app
git diff --check
git status --short
```

Para arquitectura, ejecución y limitaciones, consultar `README.md`. Para el
contrato entre módulos, consultar `docs/integration/D8C_HANDOFF.md`.
