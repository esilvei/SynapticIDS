# GKE Cluster
resource "google_container_cluster" "primary" {
  name = var.gke_cluster_name
  location = var.region
  initial_node_count = 1
    node_config {
        machine_type = "e2-medium"
        disk_size_gb = 50
        oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        ]
  }
}

# Cloud SQL for PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = var.db_instance_name
  database_version = "POSTGRES_14"
  region           = var.region

  settings {
    tier = "db-f1-micro"
    edition = "ENTERPRISE"
  }
}

resource "google_sql_database" "database" {
  name     = var.db_name
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "user" {
    name     = var.db_user
    instance = google_sql_database_instance.main.name
    password = var.db_password
}

# Memorystore for Redis
resource "google_redis_instance" "cache" {
  name           = var.redis_instance_name
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region
}

resource "google_sql_database" "mlflow_db" {
  name     = "mlflowdb"
  instance = google_sql_database_instance.main.name
}
