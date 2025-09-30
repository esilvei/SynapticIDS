# SynapticIDS: A Hybrid Deep Learning Intrusion Detection System 🛡️

## 🚀 Overview

SynapticIDS is an end-to-end Intrusion Detection System (IDS) that leverages a hybrid deep learning architecture to identify and classify security threats in computer networks. The model, trained on the UNSW-NB15 dataset, is served via a high-performance **FastAPI** backend, providing a complete solution from training to real-time inference.

This project uses **MLflow** 📈 for robust management of the entire model lifecycle. For development, this project uses **UV** ⚡ for high-speed dependency management and enforces code quality with **Pre-Commit** hooks ✅, including `ruff` and `pylint`.

***

## 🧠 Model Architecture

The core of SynapticIDS is a hybrid model that processes network data in two parallel branches:

1.  **🖼️ 2D Convolutional Branch**: Features from network traffic are transformed into a 2D image-like representation. A Convolutional Neural Network (CNN) then extracts spatial patterns.
2.  **📉 Sequential Branch (GRU)**: The same features are treated as a time series. A Gated Recurrent Unit (GRU) network captures temporal dependencies in the packet flow.

The outputs from both branches are merged using a **Transformer Fusion** mechanism before being passed through dense layers for the final classification, determining whether the traffic is normal or malicious.

***

## ✨ Project Features

* **📦 End-to-End ML Pipeline**: Complete scripts for data setup, feature engineering, model training, and evaluation.
* **🚀 High-Performance API**: A RESTful API built with **FastAPI** to serve the trained model for real-time, batch predictions.
* **📊 Experiment Management**: Full integration with **MLflow** to track experiments, manage model versions, and deploy models seamlessly from the registry.
* **🗄️ Database Integration**: All predictions are automatically logged to a **SQLite** database using SQLAlchemy for auditing and retrieval.
* **⚡ High-Speed Dependency Management**: Uses **UV** for fast and efficient virtual environment and package management.
* **✅ Code Quality**: Pre-commit hooks are configured to enforce linting and code formatting standards automatically.

***

## 🏁 Getting Started

### Prerequisites

* Python 3.10+
* **UV** ⚡ (for package and environment management)

### 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/esilvei/SynapticIDS/](https://github.com/esilvei/SynapticIDS/)
    cd SynapticIDS
    ```

2.  **Create a virtual environment and install dependencies using UV:**
    ```bash
    # Create a virtual environment
    uv venv

    # Activate the environment
    # On Windows (PowerShell/CMD)
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate

    # Install/sync dependencies from requirements.txt
    uv sync
    ```

3.  **Set up the Pre-Commit Hooks:**
    * This ensures that code quality checks are run automatically before each commit.
    ```bash
    pre-commit install
    ```

### 🚀 How to Run

Follow these steps to train the model and run the API.

**1. Train the Model:**
Before running the API, you must train the model. This script will run the full pipeline and register the final model artifact in the local MLflow registry (`mlruns/`).
```bash
python src/synaptic_ids/training/run_training.py
```

**2. Run the API Server:**
Once the model is trained, start the API server using Uvicorn.
```bash
uvicorn src.synaptic_ids.api.main:app --reload
```
The server will start on `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

***

## 🔌 API Usage

The API is designed to receive a list of traffic records and return a prediction for each.

### API Endpoints

| Method | Endpoint                    | Description                                         |
| :----- | :-------------------------- | :-------------------------------------------------- |
| `GET`  | `/`                         | Health check to confirm the API is running.         |
| `POST` | `/predict/`                 | Submits one or more traffic records for classification. |
| `GET`  | `/predictions/`             | Retrieves a paginated list of all stored predictions. |
| `GET`  | `/predictions/{prediction_id}` | Retrieves a single prediction by its unique ID.     |

### Example: Making a Prediction with `curl`

You can send a `POST` request to the `/predict/` endpoint with a list of records.

**Example of an "Attack" record:**
```bash
curl -X POST "[http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)" -H "Content-Type: application/json" -d '{
  "records": [
    {
      "proto": "tcp", "state": "FIN", "dur": 0.088159, "sbytes": 568, "dbytes": 3130,
      "sttl": 62, "dttl": 252, "sloss": 7, "dloss": 5, "service": "-",
      "sload": 45554.402344, "dload": 243763.421875, "spkts": 18, "dpkts": 14,
      "smean": 32, "dmean": 224, "sjit": 23.477833, "djit": 1.400333,
      "stime": 1421927414, "ltime": 1421927414, "sinpkt": 5.185823, "dinpkt": 6.209231,
      "is_sm_ips_ports": 0, "ct_srv_src": 3, "ct_srv_dst": 2, "ct_dst_ltm": 1,
      "ct_src_ltm": 1, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1,
      "rate": 351.638062, "response_body_len": 0, "tcprtt": 0.057393, "synack": 0.013233,
      "ackdat": 0.04416, "ct_state_ttl": 0, "ct_flw_http_mthd": 0, "is_ftp_login": 0,
      "ct_ftp_cmd": 0, "trans_depth": 0, "swin": 255, "dwin": 255,
      "stcpb": 133588210, "dtcpb": 357039015
    }
  ]
}'
```
**Expected Response:**
```json
{
  "predictions": [
    {
      "label": "Attack",
      "prediction": 1,
      "confidence": 0.9619,
      "probabilities": null
    }
  ]
}
```

***

## 🖥️ Using the MLflow UI

To visualize experiment results, run the MLflow UI pointing to the project's database:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
Access the dashboard at `http://127.0.0.1:5000`.

***

## 🗺️ Roadmap & Next Steps

This section outlines the plan for evolving SynapticIDS into a fully-fledged, production-ready system.

### Phase 1: Enhance Testing and Caching
-   **Goal**: Increase system reliability and performance.
-   **Tasks**:
    1.  **Unit Tests for API**: Implement a suite of unit tests for the API layer, covering endpoints, CRUD operations, and business logic to ensure robustness.
    2.  **Integrate Redis for Caching**: Use **Redis** as a high-speed, in-memory cache for recent predictions. This will reduce database load for repeated queries and speed up response times.

### Phase 2: Real-time Sequential Analysis
-   **Goal**: Leverage the full power of the sequential model by analyzing true, time-ordered sequences of traffic.
-   **Tasks**:
    1.  **Stateful Session Tracking with Redis**: For each unique source IP or session, store the last `N` traffic records in a Redis List or Stream.
    2.  **Adapt API for True Sequences**: Evolve the API so that when a new record arrives, it retrieves the recent records from Redis for that session, forms a true sequence, and sends it to the model. This will enable more context-aware and accurate predictions.

### Phase 3: Containerization & Local Orchestration
-   **Goal**: Ensure consistent and portable deployments by packaging the application.
-   **Tasks**:
    1.  **Finalize `Dockerfile`**: Create a multi-stage `Dockerfile` to build a lean, optimized production image for the FastAPI application.
    2.  **Enhance `docker-compose.yaml`**: Update the `docker-compose.yaml` to orchestrate a multi-service environment including the API, Redis, and a **PostgreSQL** database as a more robust production alternative to SQLite.

### Phase 4: Cloud Deployment with IaC and Kubernetes
-   **Goal**: Automate infrastructure provisioning and deploy the application in a scalable, resilient cloud environment.
-   **Tasks**:
    1.  **Infrastructure as Code (IaC) with Terraform**: Write **Terraform** scripts to automatically provision a managed Kubernetes cluster (e.g., GKE on GCP), a managed Redis instance, and a managed PostgreSQL database.
    2.  **Orchestration with Kubernetes (K8s)**:
        * Write Kubernetes manifest files (`Deployment`, `Service`, `Ingress`) to deploy the containerized services.
        * Implement a Horizontal Pod Autoscaler (HPA) to automatically scale the API based on traffic load.
    3.  **CI/CD Pipeline**: Set up a CI/CD pipeline (e.g., using GitHub Actions) to automate testing, building Docker images, and deploying updates to the Kubernetes cluster.

### Phase 5: Advanced MLOps and Data Management
-   **Goal**: Implement advanced MLOps practices for data versioning and continuous model improvement.
-   **Tasks**:
    1.  **Data and Model Versioning with DVC**: Integrate **DVC (Data Version Control)** to work alongside Git. This will allow for versioning of large datasets and model files, ensuring full reproducibility of experiments.
    2.  **Automated Retraining Pipeline**: Design and implement a system for automated model retraining. This pipeline would:
        * Monitor for new, labeled data (e.g., from a production feedback loop).
        * Trigger the training script automatically when a sufficient amount of new data is available.
        * Evaluate the newly trained model against a holdout dataset.
        * If the new model shows improved performance, automatically register it in the MLflow Model Registry and potentially flag it for promotion to production.