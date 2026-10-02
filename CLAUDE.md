# CLAUDE.md — Contexto del proyecto para Claude Code

## 🎯 Proyecto

MVP de una startup FinTech multimodal (Taller B5-T4). Análisis automatizado de informes financieros SEC (10-K / 10-Q) y earnings calls. Salida: informe visual con infografías + resumen ejecutivo en audio.

## 👥 Equipo y módulos

- **Dani** → `src/extraction/` (extracción de datos, agentes de análisis, chunks)
- **Cristian** → `src/audio/` (Whisper para transcripción, TTS para resumen)
- **Marco** → `src/visualization/`, `src/api/`, `app/` (infografías, FastAPI, Streamlit)

**Regla de oro**: cada uno trabaja en su carpeta. No modificar código de otro módulo sin avisar.

## 🏗️ Arquitectura

Separación estricta de capas:

1. **Capa de modelos IA** (`src/extraction/`, `src/audio/`): llamadas a modelos, orquestación, pre/postprocesamiento.
2. **Capa de lógica de negocio** (`src/api/`): FastAPI, endpoints, validación, orquestación del pipeline.
3. **Capa de presentación** (`app/`, `src/visualization/`): Streamlit, infografías, UX.

**Nunca** mezclar: no hacer llamadas a modelos directamente desde la UI.

## 🐍 Stack técnico

- Python 3.14
- `edgartools` (v5.21.1) para SEC filings
- FastAPI + Streamlit
- Pandas + PyArrow (Parquet)
- Modelos IA: pendiente de definir (open-source preferido por presupuesto)

## 📦 Datos disponibles

- `data/raw/txt/` — 135 archivos TXT de 10-K/10-Q (12 tickers, 2024-2026)
- `data/raw/xbrl/` — 656 Parquet con estados financieros (balance, income, cash flow, equity, comprehensive)

**No commitear datos** (están en `.gitignore`). Se regeneran con `src/extraction/download_xbrl.py`.

## ⚙️ Comandos frecuentes

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecutar script de descarga (si hace falta regenerar datos)
python src\extraction\download_xbrl.py