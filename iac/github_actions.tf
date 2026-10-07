# =============================================================================
#  GitHub Actions → GCP sin claves (Workload Identity Federation).
# =============================================================================
#  El workflow `.github/workflows/deploy-cloudrun.yml` se autentica con un
#  token OIDC de GitHub que GCP canjea por credenciales de la SA `deployer`.
#  No hay JSON keys que rotar ni guardar en GitHub Secrets.
#
#  Solo el repo `var.github_repository` puede suplantar a la SA (condición
#  en el provider + binding por `attribute.repository`).
# =============================================================================

resource "google_project_service" "github_actions" {
  for_each = toset([
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "iam.googleapis.com",
  ])

  project            = var.gcp_project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.gcp_project_id
  workload_identity_pool_id = "${var.base_name}-gh"
  display_name              = "GitHub Actions (${var.base_name})"

  depends_on = [google_project_service.github_actions]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.gcp_project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Rechaza tokens de cualquier otro repo aunque alguien conozca el pool.
  attribute_condition = "assertion.repository == \"${var.github_repository}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# --- SA que usa el workflow ---------------------------------------------------
resource "google_service_account" "deployer" {
  project      = var.gcp_project_id
  account_id   = "${var.base_name}-deployer"
  display_name = "GitHub Actions deployer (${var.base_name})"
}

resource "google_service_account_iam_member" "deployer_wif" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# Push de imágenes solo a nuestro repo de Artifact Registry.
resource "google_artifact_registry_repository_iam_member" "deployer_writer" {
  project    = var.gcp_project_id
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

# Crear revisiones del servicio Cloud Run.
resource "google_cloud_run_v2_service_iam_member" "deployer_developer" {
  project  = var.gcp_project_id
  location = var.region_name
  name     = module.cloudrun.cloudrun_name
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# `gcloud run deploy` necesita actAs sobre la SA de runtime del servicio.
resource "google_service_account_iam_member" "deployer_act_as_runtime" {
  service_account_id = "projects/${var.gcp_project_id}/serviceAccounts/${module.cloudrun.sa_email}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
