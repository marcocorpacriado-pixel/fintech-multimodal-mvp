output "service_url" {
  description = "URL pública del servicio Cloud Run."
  value       = module.cloudrun.cloudrun_service_url
}

output "service_name" {
  description = "Nombre del servicio Cloud Run (lo necesita el script de deploy)."
  value       = module.cloudrun.cloudrun_name
}

output "service_account_email" {
  description = "SA que ejecuta Cloud Run; con permisos para leer los secrets."
  value       = module.cloudrun.sa_email
}

output "artifact_registry_repo" {
  description = "Hostname + path del repo Docker (sin tag)."
  value = format(
    "%s-docker.pkg.dev/%s/%s",
    var.region_name,
    var.gcp_project_id,
    google_artifact_registry_repository.images.repository_id,
  )
}

output "image_latest_path" {
  description = "Imagen con tag :latest. Úsala en el script bash."
  value = format(
    "%s-docker.pkg.dev/%s/%s/%s:latest",
    var.region_name,
    var.gcp_project_id,
    google_artifact_registry_repository.images.repository_id,
    var.base_name,
  )
}

output "secret_ids" {
  description = "IDs de los secrets creados; súbeles valor desde la consola."
  value       = { for k, s in google_secret_manager_secret.secrets : k => s.secret_id }
}
