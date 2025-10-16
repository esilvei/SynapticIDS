output "gke_cluster_name" {
  value = google_container_cluster.primary.name
}

output "gke_cluster_endpoint" {
  value = google_container_cluster.primary.endpoint
}

output "db_instance_connection_name" {
  value = google_sql_database_instance.main.connection_name
}

output "redis_instance_host" {
  value = google_redis_instance.cache.host
}
