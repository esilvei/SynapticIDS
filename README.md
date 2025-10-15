# SynapticIDS: A Hybrid Deep Learning Intrusion Detection System 🛡️

## 🚀 Overview

SynapticIDS is an end-to-end Intrusion Detection System (IDS) that leverages a hybrid deep learning architecture to identify and classify security threats in computer networks. The model, trained on the UNSW-NB15 dataset, is served via a high-performance FastAPI backend and orchestrated with Docker, providing a complete, portable solution from training to real-time inference.

This project uses MLflow 📈 for robust management of the entire model lifecycle. For development, this project uses UV ⚡ for high-speed dependency management and enforces code quality with Pre-Commit hooks ✅, including `ruff` and `pylint`.

-----

## 🧠 Model Architecture

The core of SynapticIDS is a hybrid model that processes network data in two parallel branches:

  - 🖼️ **2D Convolutional Branch**: Features from network traffic are transformed into a 2D image-like representation. A Convolutional Neural Network (CNN) then extracts spatial patterns.
  - 📉 **Sequential Branch (GRU)**: The same features are treated as a time series. A Gated Recurrent Unit (GRU) network captures temporal dependencies in the packet flow.

The outputs from both branches are merged using a **Transformer Fusion mechanism** before being passed through dense layers for the final classification, determining whether the traffic is normal or malicious.

-----

## ✨ Project Features

  - 🐳 **Containerized Environment**: Full application suite orchestrated with Docker Compose, including the API, MLflow, Redis, and a PostgreSQL database for a consistent development and production setup.
  - 🚀 **High-Performance API**: A RESTful API built with FastAPI to serve the trained model for real-time, batch predictions.
  - 📊 **Experiment Management**: Full integration with MLflow to track experiments, manage model versions, and deploy models seamlessly from the registry.
  - 🧠 **Real-time Sequential Analysis with Redis**: The system uses Redis for stateful session tracking. This allows the model to analyze true, time-ordered sequences of traffic for a given session, enabling more context-aware and accurate predictions.
  - 🗄️ **Production-Ready Database**: All predictions are logged to a PostgreSQL database using SQLAlchemy for auditing and retrieval.
  - ✅ **Robust Testing Suite**: Includes unit and integration tests (pytest) for the API layer, ensuring code reliability and correctness.
  - ⚡ **High-Speed Dependency Management**: Uses UV for fast and efficient virtual environment and package management during development.
  - ✅ **Code Quality**: Pre-commit hooks are configured to enforce linting and code formatting standards automatically.

-----

## 📈 Model Performance

The model achieves state-of-the-art performance in network traffic classification. The results were evaluated on the test set of the UNSW-NB15 dataset.

| Metric | Score  |
| :--- |:-------|
| Accuracy | 91.13% |
| Precision | 0.9140 |


-----

## 🏁 Getting Started with Docker (Recommended)

This is the easiest and most reliable way to run the entire application stack.

### Prerequisites

  * Docker and Docker Compose
  * **Kaggle API Credentials**: To download the dataset, you must have your `kaggle.json` file in `~/.kaggle/`. You can find instructions [here](https://www.kaggle.com/docs/api).

### 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/esilvei/SynapticIDS/
    cd SynapticIDS
    ```
2.  **Create an environment file:**
    Create a file named `.env` in the project root and add the following content. This will configure the database credentials.
    ```env
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password
    POSTGRES_DB=synaptic
    ```

### 🚀 How to Run

The workflow is separated into a one-time training step and the main application execution.

1.  **Train the Model (One-Time Task)**
    This command builds the Docker image, starts the necessary services (DB and MLflow), and runs the training script. The final model is automatically registered in MLflow.
    ```bash
    docker-compose run --build training
    ```
2.  **Run the Application**
    Once the model is trained, this command starts the API, MLflow UI, Redis, and PostgreSQL database. It will not run the training again.
    ```bash
    docker-compose up
    ```

The services are now available at:

  * **SynapticIDS API**: `http://localhost:8000`
  * **API Docs (Swagger UI)**: `http://localhost:8000/docs`
  * **MLflow Dashboard**: `http://localhost:5000`

-----

## 🔌 API Usage

The API is designed to receive a list of traffic records and return a prediction for each. For sequential analysis, a `session_id` should be provided.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check to confirm the API is running. |
| `POST` | `/predictions/` | Submits one or more traffic records for classification. |
| `GET` | `/predictions/` | Retrieves a paginated list of all stored predictions. |
| `GET` | `/predictions/{prediction_id}` | Retrieves a single prediction by its unique ID. |

### Example: Making a Prediction with `curl`

```bash
curl -X POST "http://localhost:8000/predictions/" \
-H "Content-Type: application/json" \
-d '{
  "session_id": "user123_session",
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
      "probabilities": { "Normal": 0.0381, "Attack": 0.9619 }
    }
  ]
}
```

-----

## 🗺️ Roadmap

This section outlines the future plan for evolving SynapticIDS into a fully-fledged, production-ready system.

### Phase 1: Automation with CI/CD 🤖

**Goal**: Automate testing and image building to ensure code quality and create deployment artifacts.

**Tasks**:

  * **Implement GitHub Actions Workflows**:
      * **Continuous Integration (CI)**: Create a workflow that triggers on every push and pull request to the `main` branch. This workflow will automatically run linting, formatting checks, and the full `pytest` suite.
      * **Docker Build & Push**: Create a second workflow that, upon a successful merge to `main`, automatically builds the production Docker image and pushes it to a container registry (e.g., GitHub Container Registry or Docker Hub).

### Phase 2: Cloud Deployment with IaC and Kubernetes ☁️

**Goal**: Automate infrastructure provisioning and deploy the application in a scalable, resilient cloud environment.

**Tasks**:

  * **Infrastructure as Code (IaC) with Terraform**: Write Terraform scripts to automatically provision a managed Kubernetes cluster (e.g., GKE on GCP), a managed Redis instance, and a managed PostgreSQL database.
  * **Orchestration with Kubernetes (K8s)**:
      * Write Kubernetes manifest files (Deployment, Service, Ingress) to deploy the containerized services.
      * Implement a Horizontal Pod Autoscaler (HPA) to automatically scale the API based on traffic load.
  * **Continuous Deployment (CD)**: Extend the GitHub Actions pipeline to automatically deploy the new Docker image version to the Kubernetes cluster after it passes all CI checks.

### Phase 3: Advanced MLOps and Data Management 📊

**Goal**: Implement advanced MLOps practices for data versioning and continuous model improvement.

**Tasks**:

  * **Data and Model Versioning with DVC**: Integrate DVC (Data Version Control) to work alongside Git, enabling versioning of large datasets and models for full experiment reproducibility.
  * **Automated Retraining Pipeline**: Design a system for automated model retraining that monitors for new data, triggers training jobs, evaluates new models, and flags them for promotion in the MLflow Model Registry if performance improves.

### Phase 4: Security Hardening and Continuous Improvement 🔐

**Goal**: Enhance API security, implement access controls, and create a data feedback loop for model retraining.

**Tasks**:

  * **Implement API Security**:
      * **Authentication** 🔑: Integrate OAuth2 with JWT tokens to secure all endpoints. This will ensure that only authenticated clients can interact with the API.
      * **Authorization** 🛡️: Implement role-based access control (RBAC). For instance, restrict access to the `GET /predictions/` endpoints to users with "analyst" or "admin" roles, preventing unauthorized data exposure.
  * **Establish a Data Feedback Loop** 🔄:
      * **Log Production Inputs** 📝: Create a secure and efficient mechanism to capture and store the traffic records sent to the prediction endpoint.
      * **Data Labeling Interface** 🏷️: Develop a simple internal tool or process for security analysts to review and label the captured traffic, especially for ambiguous or novel threats. This labeled data will be crucial for the next generation of model training.
