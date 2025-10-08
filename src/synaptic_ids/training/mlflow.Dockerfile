# Use a imagem oficial do MLflow como base
FROM ghcr.io/mlflow/mlflow:v2.14.1

# Instale o driver do PostgreSQL que falta
RUN pip install psycopg2-binary
