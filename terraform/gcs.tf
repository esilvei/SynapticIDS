resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}
