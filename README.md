# FinTech Multimodal MVP

MVP de una startup FinTech basada en IA multimodal para el análisis automatizado de informes financieros (10-K / 10-Q) y earnings calls.

**Taller B5-T4 — Octubre 2026**

## 👥 Equipo

| Rol | Responsable | Módulo |
|---|---|---|
| Extracción y análisis de datos | **Dani** | `src/extraction/` |
| Audio (transcripción + síntesis) | **Cristian** | `src/audio/` |
| Visualización e interfaz web | **Marco** | `src/visualization/`, `src/api/`, `app/` |

## 🎯 Propuesta de valor

Un analista financiero automatizado que recibe informes 10-K/10-Q y earnings calls, y devuelve un informe visual con infografías comparativas y un resumen ejecutivo en audio.

## 🏗️ Arquitectura

_En construcción. Diagrama de flujo multimodal disponible en `docs/`._

## 📦 Estructura del repositorio

```text
fintech-multimodal-mvp/
├── .claude/              # Skills y configuración de Claude Code
├── app/                  # Frontend Streamlit
├── data/
│   ├── raw/
│   │   ├── txt/          # Corpus narrativo (135 archivos)
│   │   └── xbrl/         # Corpus financiero estructurado (656 Parquet)
│   └── processed/
├── docs/                 # Diagramas y documentación
├── src/
│   ├── api/              # Backend FastAPI
│   ├── audio/            # Módulo de audio
│   ├── extraction/       # Módulo de extracción
│   └── visualization/    # Módulo de infografías
├── tests/                # Tests
├── CLAUDE.md             # Contexto del proyecto para Claude Code
├── requirements.txt
└── README.md
```

## 🚀 Instalación y uso

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Arranca la API y el dashboard en dos terminales (ambas desde la raíz del repo y con el entorno activado):

```powershell
# Terminal 1 — API FastAPI (docs en http://localhost:8000/docs)
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Dashboard Streamlit (http://localhost:8501)
streamlit run app/streamlit_app.py
```

- **Modo demo** (por defecto): la API devuelve un fixture sintético (`src/api/demo_fixture.json`). No necesita claves ni red.
- **Modo real**: requiere en `.env` las variables `EDGAR_IDENTITY` (`"Nombre email@dominio"`), `OPENROUTER_API_KEY` y `OPENROUTER_MODEL`. La fecha que se pide es la *filing date* del 10-Q/10-K en la SEC.
- **Audio**: la primera síntesis descarga el modelo Kokoro (~350 MB) en `KOKORO_MODEL_DIR` (por defecto `~/.cache/kokoro`).
- Si la API no está en `http://localhost:8000`, define `API_URL` antes de lanzar Streamlit.

Tests:

```powershell
python -m pytest -q
```

## 📊 Corpus de datos

- **SEC filings**: 12 empresas del S&P 500, 2024-2026.
  - 135 documentos TXT (texto narrativo).
  - 656 archivos Parquet XBRL (datos financieros estructurados).
- **Earnings calls**: pendiente de integración.

## 📜 Licencia

Proyecto académico — Taller B5-T4.
