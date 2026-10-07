# syntax=docker/dockerfile:1.7
# =============================================================================
#  fintech-multimodal-mvp — imagen única (FastAPI + Streamlit)
# =============================================================================
#  Decisión clave: el modelo de Kokoro (~350 MB) se descarga DURANTE EL BUILD,
#  no en runtime. En Cloud Run cada cold start arranca un contenedor nuevo con
#  disco efímero: si el modelo se bajase al vuelo, el primer usuario esperaría
#  ~60 s en cada arranque en frío. Horneado en la imagen, arranca en segundos.
#
#  Además la descarga vive en su propia capa, ANTES de copiar el código, así
#  un cambio en src/ no obliga a volver a bajar los 350 MB en cada rebuild.
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1: builder — compila dependencias y descarga el modelo
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential sólo existe en esta etapa; no llega a la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Entorno virtual aislado: se copia entero al runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Capa de dependencias -----------------------------------------------------
# Sólo se invalida si cambia requirements.txt.
# torch se instala ANTES desde el índice CPU: el wheel por defecto de PyPI trae
# CUDA + librerías NVIDIA (~3 GB) inútiles en Cloud Run (sin GPU). Con torch ya
# satisfecho, `pip install -r` no vuelve a bajar la variante CUDA.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# --- Capa del modelo Kokoro ---------------------------------------------------
# Copiamos ÚNICAMENTE tts.py (no todo src/) para no invalidar esta capa cada vez
# que se toque cualquier otro fichero del proyecto. Reutilizamos la función
# _ensure_model_files() del propio módulo para no duplicar las URLs.
#
# Además lanzamos una síntesis trivial en EN y ES para:
#   1. Forzar la descarga de cualquier diccionario perezoso de misaki / espeak-ng.
#   2. Dejar las voces pre-cargadas en /opt/kokoro.
#   3. Validar que el stack TTS realmente arranca.
# Hace falta instalar espeak-ng también en el builder para esta validación.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV KOKORO_MODEL_DIR=/opt/kokoro
COPY src/audio/tts.py /tmp/_tts.py
RUN python -c "\
import importlib.util, sys; \
spec = importlib.util.spec_from_file_location('_tts', '/tmp/_tts.py'); \
m = importlib.util.module_from_spec(spec); \
sys.modules['_tts'] = m; \
spec.loader.exec_module(m); \
paths = m._ensure_model_files(); \
print('[build] Modelo horneado en:', *paths); \
print('[build] Pre-cargando Kokoro en memoria y haciendo síntesis trivial...'); \
r1 = m.synthesize('Warm up.', language='en'); \
print('[build]   EN OK:', r1.duration, 's,', r1.voice); \
r2 = m.synthesize('Prueba de voz.', language='es'); \
print('[build]   ES OK:', r2.duration, 's,', r2.voice); \
print('[build] Kokoro warm-up terminado.')\
" && rm /tmp/_tts.py

# --- Capa del modelo FinBERT --------------------------------------------------
# Mismo motivo que Kokoro: en Cloud Run el disco es efímero (y en memoria), así
# que bajar ProsusAI/finbert (~440 MB) en cada cold start sería lento y gastaría
# RAM. Se hornea en /opt/hf y en runtime se fuerza modo offline. Reutilizamos
# sentiment.py (solo depende de stdlib + transformers) y validamos que clasifica.
# transformers también baja una copia safetensors desde un PR de conversión
# automática (~440 MB duplicados); en offline solo se usa el snapshot de `main`,
# así que el resto se elimina.
ENV HF_HOME=/opt/hf
COPY src/extraction/sentiment.py /tmp/_sentiment.py
RUN python -c "\
import importlib.util, sys; \
spec = importlib.util.spec_from_file_location('_sentiment', '/tmp/_sentiment.py'); \
m = importlib.util.module_from_spec(spec); \
sys.modules['_sentiment'] = m; \
spec.loader.exec_module(m); \
r = m.classify_financial_sentiment('Revenue increased strongly this quarter.'); \
assert r is not None, 'FinBERT failed to load'; \
print('[build] FinBERT OK:', r)\
" && rm /tmp/_sentiment.py \
    && python -c "\
import os, shutil; from pathlib import Path; \
hub = Path('/opt/hf/hub'); repo = hub / 'models--ProsusAI--finbert'; \
keep = (repo / 'refs' / 'main').read_text().strip(); \
[shutil.rmtree(s) for s in (repo / 'snapshots').iterdir() if s.name != keep]; \
shutil.rmtree(repo / 'refs' / 'refs', ignore_errors=True); \
live = {os.path.realpath(f) for f in (repo / 'snapshots' / keep).iterdir()}; \
[f.unlink() for f in (repo / 'blobs').iterdir() if os.path.realpath(f) not in live]; \
[f.unlink() for f in hub.glob('blobs/*/*') if str(f.resolve()) not in live]; \
shutil.rmtree('/opt/hf/xet', ignore_errors=True); \
print('[build] FinBERT cache pruned to snapshot', keep)\
"


# -----------------------------------------------------------------------------
# Stage 2: runtime — imagen final, sin toolchain de compilación
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libsndfile1 → soundfile (lectura/escritura WAV)
# espeak-ng    → fallback de fonemización de Kokoro para idiomas no ingleses (es, fr, it...)
# ffmpeg       → pydub, necesario sólo si se pide output_format="mp3"
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        espeak-ng \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios (buena práctica y requisito de varios entornos gestionados).
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/kokoro /opt/kokoro
COPY --from=builder /opt/hf /opt/hf

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Apunta a los pesos ya horneados: tts.py los encuentra y NO descarga nada.
    KOKORO_MODEL_DIR=/opt/kokoro \
    # FinBERT horneado: offline para que nunca intente descargar en runtime.
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Cloud Run inyecta $PORT; 8080 es el default razonable en local.
    PORT=8080 \
    # Puerto interno de FastAPI (no se expone al exterior).
    API_PORT=8000

WORKDIR /app

COPY --chown=appuser:appuser docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY --chown=appuser:appuser . /app

# `data/` no está en git (y .dockerignore excluye sus subcarpetas), así que en
# un build desde CI no existe; `/app` lo crea WORKDIR como root y appuser no
# podría crearla en runtime (sec_ingestion escribe en data/processed/...).
RUN install -d -o appuser -g appuser /app/data

USER appuser

EXPOSE 8080

# Cloud Run ignora HEALTHCHECK, pero es útil en local y en docker-compose.
# Comprobamos $PORT (lo ocupa Streamlit, o la API en modo fallback), así vale
# para ambos modos sin saber cuál está activo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import socket,os; socket.create_connection(('127.0.0.1', int(os.environ['PORT'])), timeout=4).close()" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
