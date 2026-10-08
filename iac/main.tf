# =============================================================================
#  Cloud Run — invoca el módulo local tf-gcp-cloudrun.
# =============================================================================
#  El módulo hace `ignore_changes` sobre `template[0].containers[*].image`:
#  Terraform fija SOLO la imagen inicial (bootstrap). Las revisiones
#  posteriores las crea `scripts/build_and_push.sh` con `gcloud run deploy`.
# =============================================================================

locals {
  # Imagen DUMMY para que el primer `terraform apply` funcione aunque todavía
  # no hayamos subido nada a Artifact Registry. Es la imagen "hello" pública de
  # Google (siempre disponible, sin autenticación). El módulo tiene
  # `ignore_changes` sobre `template[0].containers[*].image`, así que Terraform
  # NO sobrescribirá esta imagen después: cada deploy real crea una revisión
  # nueva vía `gcloud run deploy --image=...` desde `scripts/build_and_push.sh`.
  bootstrap_image = "gcr.io/cloudrun/hello"

  # vCPUs del contenedor. KOKORO_THREADS se deriva de aquí: Cloud Run gen2 no
  # expone la cuota de CPU al proceso (os.sched_getaffinity devuelve 5-10
  # según el host), y ONNX Runtime con más hilos que vCPUs sufre throttling.
  cpu_count  = 4
  memory_gib = 4
}

module "cloudrun" {
  source = "../../../../../repos/holafly/holafly-terraform-repos/tf-gcp-cloudrun"

  base_name          = var.base_name
  gcp_project_id     = var.gcp_project_id
  gcp_project_sys_id = var.gcp_project_sys_id
  region_name        = var.region_name

  # Creamos SA interna (6-30 chars).
  service_account = {
    name = "${var.base_name}-run"
  }

  # Público: cualquiera puede invocar el endpoint. Para restringir a
  # usuarios/grupos concretos, cambia "allUsers" por "user:correo@..." o
  # "group:grupo@...".
  permissions = {
    "public_invoker" = {
      account     = "allUsers"
      permissions = ["roles/run.invoker"]
    }
  }

  cloudrun = {
    # Público (no detrás de Load Balancer interno).
    ingress     = "INGRESS_TRAFFIC_ALL"
    concurrency = 4
    timeout     = "600s"

    autoscaling = {
      min = 0
      # max=1 evita el problema de Streamlit sin session affinity: todas las
      # conexiones WebSocket caen en la misma instancia. Suficiente para la
      # demo del máster; sube si esperas carga concurrente.
      max = 1
    }

    containers = [{
      name  = "app"
      image = local.bootstrap_image

      ports = {
        container_port = 8080
      }

      resources = {
        cpu               = "${local.cpu_count * 1000}m"
        memory            = "${local.memory_gib * 1024}Mi" # Kokoro + Streamlit + FastAPI
        startup_cpu_boost = true
        cpu_idle          = true
      }

      # PORT lo inyecta Cloud Run automáticamente (no se puede sobrescribir).
      envs = [
        { name = "API_PORT", value = "8000" },
        { name = "KOKORO_MODEL_DIR", value = "/opt/kokoro" },
        { name = "OPENROUTER_MODEL", value = var.openrouter_model },
        { name = "KOKORO_THREADS", value = tostring(local.cpu_count) },
        { name = "IDLE_TIMEOUT_SECONDS", value = tostring(var.idle_timeout_seconds) },
        { name = "IDLE_WARNING_SECONDS", value = tostring(var.idle_warning_seconds) },
        # Service size, used to cost Cloud Run time in the Performance tab.
        { name = "CLOUD_RUN_VCPU", value = tostring(local.cpu_count) },
        { name = "CLOUD_RUN_MEMORY_GIB", value = tostring(local.memory_gib) },
      ]

      secrets = [
        {
          name   = "OPENROUTER_API_KEY"
          secret = google_secret_manager_secret.secrets["openrouter_api_key"].secret_id
        },
        {
          name   = "GROQ_API_KEY"
          secret = google_secret_manager_secret.secrets["groq_api_key"].secret_id
        },
        {
          name   = "EDGAR_IDENTITY"
          secret = google_secret_manager_secret.secrets["edgar_identity"].secret_id
        },
      ]
    }]
  }

  monitoring = {
    repo_iac = var.repo_iac_url
  }

  depends_on = [
    google_artifact_registry_repository.images,
    google_secret_manager_secret.secrets,
  ]
}
