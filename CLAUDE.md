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
3. `src/audio/`: STT/TTS, actualmente en `feature/audio_integration`.
4. `src/api/`: futura orquestación FastAPI de Marco.
5. `src/visualization/` y `app/`: futura presentación de Marco.

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

Las dependencias de audio y presentación se reconciliarán al integrar sus
ramas. No versionar `.env`, datos SEC, modelos o audio.

## Configuración principal

- `EDGAR_IDENTITY`: SEC real.
- `OPENROUTER_API_KEY` y `OPENROUTER_MODEL`: análisis real.
- `GROQ_API_KEY`: STT cuando se integre audio.
- `KOKORO_MODEL_DIR`: caché TTS opcional.

Nunca imprimir secretos, headers de autorización, prompts completos o filings.

## Comandos de verificación

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m compileall src
git diff --check
git status --short
```

Para arquitectura, ejecución y limitaciones, consultar `README.md`. Para el
contrato entre módulos, consultar `docs/integration/D8C_HANDOFF.md`.
