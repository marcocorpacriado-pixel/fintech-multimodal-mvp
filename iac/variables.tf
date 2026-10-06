variable "gcp_project_id" {
  type        = string
  description = "GCP project where the service lives."
  default     = "ml-miax"
}

variable "gcp_project_sys_id" {
  type        = string
  description = <<-EOT
    GCP project for cross-project resources (networking/monitoring host).
    En este MVP no tenemos proyecto SYS separado, apuntamos al mismo proyecto.
  EOT
  default     = "ml-miax"
}

variable "region_name" {
  type        = string
  description = "Región de Cloud Run y Artifact Registry."
  default     = "europe-southwest1"
}

variable "base_name" {
  type        = string
  description = "Nombre base; el servicio Cloud Run se llamará igual."
  default     = "fintech-mvp"
}

variable "openrouter_model" {
  type        = string
  description = "Modelo de OpenRouter (debe soportar JSON schema estructurado)."
  default     = "anthropic/claude-3.5-sonnet"
}

variable "repo_iac_url" {
  type        = string
  description = <<-EOT
    Identificador del repo IaC. El módulo lo mete como `user_label` de las
    alertas de monitoring, así que GCP exige el formato de label:
    minúsculas / dígitos / guiones / underscores, máx 63 chars. No admite
    URLs ni mayúsculas.
  EOT
  default     = "fintech-multimodal-mvp"

  validation {
    condition     = can(regex("^[a-z0-9_-]{1,63}$", var.repo_iac_url))
    error_message = "Solo minúsculas, dígitos, '-' y '_'; máximo 63 caracteres."
  }
}
