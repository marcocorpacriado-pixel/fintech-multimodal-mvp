#!/usr/bin/env bash
# =============================================================================
#  build_and_push.sh — build + push + deploy nueva revisión a Cloud Run.
# =============================================================================
#  Uso:
#      ./scripts/build_and_push.sh                 # tag auto (timestamp)
#      TAG=v0.1.0 ./scripts/build_and_push.sh      # tag concreto
#      SKIP_DEPLOY=1 ./scripts/build_and_push.sh   # solo build + push
#
#  Variables de entorno (con defaults razonables para este proyecto):
#      PROJECT_ID   (ml-miax)
#      REGION       (europe-southwest1)
#      BASE_NAME    (fintech-mvp)
#      TAG          (timestamp UTC)
#      SKIP_DEPLOY  (sin valor = deploy; "1" = no deploy)
#
#  Requisitos previos:
#      - gcloud auth login + `gcloud config set project ml-miax`
#      - Terraform aplicado al menos una vez (crea Artifact Registry y servicio)
#      - Docker en local
# =============================================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ml-miax}"
REGION="${REGION:-europe-southwest1}"
BASE_NAME="${BASE_NAME:-fintech-mvp}"
TAG="${TAG:-$(date -u +%Y%m%d-%H%M%S)}"

REPO="${BASE_NAME}-images"
HOST="${REGION}-docker.pkg.dev"
IMAGE="${HOST}/${PROJECT_ID}/${REPO}/${BASE_NAME}:${TAG}"
IMAGE_LATEST="${HOST}/${PROJECT_ID}/${REPO}/${BASE_NAME}:latest"

# El módulo compone el nombre del servicio como `${base_name}`
# (ver locals.tf del módulo: m_base_name = var.base_name).
SERVICE_NAME="${BASE_NAME}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "────────────────────────────────────────────────────────────"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE_NAME}"
echo "  Image   : ${IMAGE}"
echo "────────────────────────────────────────────────────────────"

# --- 0. Pre-chequeos ---------------------------------------------------------
command -v docker  >/dev/null || { echo "✗ docker no está instalado"; exit 1; }
command -v gcloud  >/dev/null || { echo "✗ gcloud no está instalado"; exit 1; }

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "✗ No estás autenticado en gcloud. Lanza: gcloud auth login"
    exit 1
fi

# --- 1. Verificar que el repo existe antes de intentar nada ------------------
echo "→ Verificando que el repo '${REPO}' existe en ${REGION}/${PROJECT_ID}"
if ! gcloud artifacts repositories describe "${REPO}" \
        --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "✗ El Artifact Registry '${REPO}' no existe en ${REGION}/${PROJECT_ID}."
    echo "  Ejecuta primero: cd iac && terraform apply"
    exit 1
fi

# --- 2. Autenticar Docker contra Artifact Registry ---------------------------
# Usamos docker login con access token en vez del credential helper de gcloud:
# el helper puede fallar en Arch si el daemon no encuentra el binario de gcloud.
# Esto crea un auth directo en ~/.docker/config.json que SIEMPRE funciona.
echo "→ Autenticando Docker contra ${HOST} (vía access token)"
gcloud auth print-access-token \
    | docker login -u oauth2accesstoken --password-stdin "https://${HOST}" >/dev/null

# --- 3. Build (forzamos amd64: Cloud Run no admite arm64) --------------------
echo "→ Build de la imagen"
docker build \
    --platform=linux/amd64 \
    --tag "${IMAGE}" \
    --tag "${IMAGE_LATEST}" \
    .

# --- 4. Push -----------------------------------------------------------------
echo "→ Push a Artifact Registry"
docker push "${IMAGE}"
docker push "${IMAGE_LATEST}"

echo "✓ Imagen disponible: ${IMAGE}"

# --- 4. Deploy nueva revisión a Cloud Run ------------------------------------
if [[ -n "${SKIP_DEPLOY:-}" ]]; then
    echo "ℹ SKIP_DEPLOY activo: no se crea nueva revisión."
    exit 0
fi

if ! gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo
    echo "⚠ El servicio '${SERVICE_NAME}' aún no existe en Cloud Run."
    echo "  Ejecuta primero:"
    echo "      cd iac && terraform init && terraform apply"
    echo "  Luego vuelve a lanzar este script."
    exit 1
fi

echo "→ Desplegando revisión en Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet

URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" --project="${PROJECT_ID}" \
        --format='value(status.url)')

echo
echo "✓ Deploy OK → ${URL}"
