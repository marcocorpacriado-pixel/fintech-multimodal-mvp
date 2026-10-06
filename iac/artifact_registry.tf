# =============================================================================
#  Artifact Registry — repo Docker donde el script bash sube la imagen.
# =============================================================================

resource "google_project_service" "artifactregistry" {
  project            = var.gcp_project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  project       = var.gcp_project_id
  location      = var.region_name
  repository_id = "${var.base_name}-images"
  description   = "Docker images for ${var.base_name}"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}
