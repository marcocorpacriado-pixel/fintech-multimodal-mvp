#!/usr/bin/env bash
# =============================================================================
#  Entrypoint: arranca FastAPI (interno) + Dash/gunicorn (expuesto en $PORT).
# =============================================================================
#  Un solo contenedor con dos procesos. La separación de capas del CLAUDE.md se
#  mantiene a nivel de CÓDIGO (la UI Dash nunca llama a src/audio/ directamente,
#  siempre pasa por la API); compartir contenedor es sólo una decisión de
#  empaquetado para abaratar el despliegue del MVP.
#
#  Dash sirve callbacks HTTP cortos (sin WebSocket): Cloud Run sólo factura
#  mientras se atiende un callback, no mientras hay una pestaña abierta.
#
#  Mientras src/api/ (Marco) y app/ (Marco) no existan, se usa como fallback
#  scripts/dev_api.py, que ya trae su propia UI web. Así la imagen es
#  construible y probable HOY, y migra sola cuando los módulos aparezcan.
# =============================================================================

set -euo pipefail

PORT="${PORT:-8080}"
API_PORT="${API_PORT:-8000}"

# --- Elegir módulo de API -----------------------------------------------------
if [[ -f /app/src/api/main.py ]]; then
    API_MODULE="src.api.main:app"      # API oficial (Marco)
else
    API_MODULE="scripts.dev_api:app"   # API temporal (Cristian)
    echo "[entrypoint] src/api/main.py no existe todavía → usando ${API_MODULE}"
fi

# --- Elegir UI ----------------------------------------------------------------
UI_MODULE=""
if [[ -f /app/app/dash_app.py ]]; then
    UI_MODULE="app.dash_app:server"    # objeto WSGI (Flask) de la app Dash
fi

# --- Apagado limpio -----------------------------------------------------------
pids=()
shutdown() {
    echo "[entrypoint] Señal recibida, parando procesos..."
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}
trap shutdown SIGTERM SIGINT

# --- Caso A: no hay UI todavía → la API ocupa $PORT ---------------------------
if [[ -z "$UI_MODULE" ]]; then
    echo "[entrypoint] No se encontró app/dash_app.py."
    echo "[entrypoint] Sirviendo sólo ${API_MODULE} en el puerto ${PORT}."
    exec uvicorn "$API_MODULE" --host 0.0.0.0 --port "$PORT"
fi

# --- Caso B: ambos procesos ---------------------------------------------------
echo "[entrypoint] API → ${API_MODULE} en 127.0.0.1:${API_PORT}"
echo "[entrypoint] UI  → ${UI_MODULE} en 0.0.0.0:${PORT}"

uvicorn "$API_MODULE" --host 127.0.0.1 --port "$API_PORT" &
api_pid=$!
pids+=($api_pid)

# Esperar a que la API responda ANTES de abrir $PORT. El startup probe de Cloud
# Run es TCP sobre $PORT: si gunicorn arranca primero, el probe pasa al instante,
# entra tráfico y la UI devuelve API_UNAVAILABLE mientras uvicorn sigue
# importando torch/Kokoro/FinBERT. Además, con cpu_idle=true, tras el probe la
# CPU se estrangula y esa importación en segundo plano tarda minutos.
echo "[entrypoint] Esperando a la API en 127.0.0.1:${API_PORT}/health..."
until python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)" 2>/dev/null; do
    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "[entrypoint] La API terminó durante el arranque. Cerrando."
        exit 1
    fi
    sleep 1
done
echo "[entrypoint] API lista."

# La UI descubre la API por esta variable; nunca hardcodear la URL en el código.
# app/api_client.py lee `API_URL`; exportamos ambas por si otra UI usa el
# nombre alternativo.
export API_URL="http://127.0.0.1:${API_PORT}"
export API_BASE_URL="$API_URL"

# 1 worker + hilos: los callbacks son I/O contra la API local (análisis hasta
# 300 s), así que los hilos bastan. --timeout cubre el peor caso del análisis.
gunicorn "$UI_MODULE" \
    --bind "0.0.0.0:${PORT}" \
    --worker-class gthread \
    --workers 1 \
    --threads 8 \
    --timeout 330 \
    --graceful-timeout 30 &
pids+=($!)

# Si cualquiera de los dos muere, tumbamos el contenedor para que Cloud Run
# lo reinicie en vez de quedarse medio vivo.
wait -n
echo "[entrypoint] Un proceso terminó inesperadamente. Cerrando."
shutdown
