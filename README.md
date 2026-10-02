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

_En construcción._

## 📊 Corpus de datos

- **SEC filings**: 12 empresas del S&P 500, 2024-2026.
  - 135 documentos TXT (texto narrativo).
  - 656 archivos Parquet XBRL (datos financieros estructurados).
- **Earnings calls**: pendiente de integración.

## 📜 Licencia

Proyecto académico — Taller B5-T4.
