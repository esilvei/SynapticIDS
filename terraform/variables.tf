variable "project_id" {
  description = "The project ID to host the resources in"
  type        = string
}

variable "region" {
  description = "The region to host the resources in"
  type        = string
  default     = "us-central1"
}

variable "gke_cluster_name" {
  description = "The name for the GKE cluster"
  type        = string
  default     = "synapticids-cluster"
}

variable "db_instance_name" {
  description = "The name for the Cloud SQL instance"
  type        = string
  default     = "synapticids-db-instance"
}

variable "db_name" {
  description = "The name of the database"
  type        = string
  default     = "synapticids_db"
}

variable "db_user" {
  description = "The username for the database"
  type        = string
  default     = "synapticids_user"
}

variable "db_password" {
  description = "The password for the database user"
  type        = string
  sensitive   = true
}

variable "redis_instance_name" {
  description = "The name for the Memorystore Redis instance"
  type        = string
  default     = "synapticids-redis"
}

variable "mlflow_db_name" {
  description = "The name of the database for MLflow"
  type        = string
  default     = "mlflowdb"
}
