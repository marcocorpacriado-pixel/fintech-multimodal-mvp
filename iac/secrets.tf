# =============================================================================
#  Secret Manager — creamos los recursos vacíos; los valores los subes a mano
#  desde la consola de GCP (o con `gcloud secrets versions add ...`).
# =============================================================================
#  Terraform NO guarda las versiones (ni valores) de los secrets: solo crea el
#  contenedor y los permisos. Los lifecycle ignore protegen de que un apply
#  posterior intente borrar la versión que subiste a mano.
# =============================================================================

resource "google_project_service" "secretmanager" {
  project            = var.gcp_project_id
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

locals {
  secret_ids = {
    openrouter_api_key = "${var.base_name}-openrouter-api-key"
    groq_api_key       = "${var.base_name}-groq-api-key"
    edgar_identity     = "${var.base_name}-edgar-identity"
  }
}

resource "google_secret_manager_secret" "secrets" {
  for_each = local.secret_ids

  project   = var.gcp_project_id
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}
