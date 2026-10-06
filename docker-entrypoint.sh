#!/usr/bin/env bash
# =============================================================================
#  Entrypoint: arranca FastAPI (interno) + Streamlit (expuesto en $PORT).
# =============================================================================
#  Un solo contenedor con dos procesos. La separación de capas del CLAUDE.md se
#  mantiene a nivel de CÓDIGO (Streamlit nunca llama a src/audio/ directamente,
#  siempre pasa por la API); compartir contenedor es sólo una decisión de
#  empaquetado para abaratar el despliegue del MVP.
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

# --- Elegir app de Streamlit --------------------------------------------------
# streamlit_app.py es el nombre real que entregó Marco en feature/marco-ui;
# el resto se mantiene como fallback por si alguna rama lo renombra.
STREAMLIT_APP=""
for candidate in /app/app/streamlit_app.py /app/app/main.py /app/app/app.py /app/app/Home.py; do
    if [[ -f "$candidate" ]]; then
        STREAMLIT_APP="$candidate"
        break
    fi
done

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

# --- Caso A: no hay Streamlit todavía → la API ocupa $PORT --------------------
if [[ -z "$STREAMLIT_APP" ]]; then
    echo "[entrypoint] No se encontró app/ de Streamlit."
    echo "[entrypoint] Sirviendo sólo ${API_MODULE} en el puerto ${PORT}."
    exec uvicorn "$API_MODULE" --host 0.0.0.0 --port "$PORT"
fi

# --- Caso B: ambos procesos ---------------------------------------------------
echo "[entrypoint] API      → ${API_MODULE} en 127.0.0.1:${API_PORT}"
echo "[entrypoint] Streamlit→ ${STREAMLIT_APP} en 0.0.0.0:${PORT}"

uvicorn "$API_MODULE" --host 127.0.0.1 --port "$API_PORT" &
pids+=($!)

# La UI descubre la API por esta variable; nunca hardcodear la URL en el código.
# app/streamlit_app.py lee `API_URL`; exportamos ambas por si otra UI usa el
# nombre alternativo.
export API_URL="http://127.0.0.1:${API_PORT}"
export API_BASE_URL="$API_URL"

streamlit run "$STREAMLIT_APP" \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false &
pids+=($!)

# Si cualquiera de los dos muere, tumbamos el contenedor para que Cloud Run
# lo reinicie en vez de quedarse medio vivo.
wait -n
echo "[entrypoint] Un proceso terminó inesperadamente. Cerrando."
shutdown
