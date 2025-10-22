# SynapticIDS: A Cloud-Native Intrusion Detection System 🛡️

## 🚀 Overview

SynapticIDS is an end-to-end Intrusion Detection System (IDS) deployed on the **Google Cloud Platform (GCP)**. It leverages a hybrid deep learning architecture to identify and classify security threats in real-time.

The entire infrastructure is provisioned as code with **Terraform** and deployed on the **Google Kubernetes Engine (GKE)**, creating a scalable, production-ready solution. The project features a complete MLOps workflow with a centralized **MLflow Tracking Server** for robust experiment management and a **GitHub Actions CI/CD pipeline** for full automation.

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Backend & API** | Python, FastAPI, SQLAlchemy |
| **ML & MLOps** | TensorFlow, MLflow, Scikit-learn, Pandas |
| **Containerization** | Docker, Kubernetes (GKE) |
| **Database & Caching**| PostgreSQL (Cloud SQL), Redis (Memorystore) |
| **Infrastructure (IaC)**| Terraform, Google Cloud Platform (GCP) |
| **CI/CD** | GitHub Actions, Google Artifact Registry |

## ✨ Project Features

* **☁️ Cloud-Native Deployment**: Fully orchestrated on **Google Kubernetes Engine (GKE)** for high availability and scalability.
* **🏗️ Infrastructure as Code (IaC)**: All cloud resources (GKE, Cloud SQL, Memorystore, GCS) are managed declaratively using **Terraform** for reproducible environments.
* **🤖 Automated CI/CD Pipeline**: A **GitHub Actions** workflow automates testing, Docker image builds, pushing to **Google Artifact Registry**, and deploying to GKE.
* **📊 Centralized MLOps**: A dedicated **MLflow Tracking Server** on GKE uses **Cloud SQL** for metadata and **Cloud Storage (GCS)** for artifact storage, enabling a full model lifecycle management.
* **🚀 High-Performance API**: A RESTful API built with **FastAPI** serves the trained model for real-time, stateful predictions.
* **🧠 Real-time Sequential Analysis**: Uses **Memorystore (Redis)** for stateful session tracking, allowing the model to analyze time-ordered sequences of traffic.
* **🗄️ Managed Database**: Predictions are logged to a managed **Google Cloud SQL (PostgreSQL)** instance for auditing and retrieval.
* **🐳 Containerized Environment**: All services are containerized with **Docker**, ensuring consistency across local development and cloud production environments.

## 🧠 Model Architecture

The core of SynapticIDS is a hybrid model that processes network data in two parallel branches:

-   🖼️ **2D Convolutional Branch**: Features from network traffic are transformed into a 2D image-like representation. A Convolutional Neural Network (CNN) then extracts spatial patterns.
-   📉 **Sequential Branch (GRU)**: The same features are treated as a time series. A Gated Recurrent Unit (GRU) network captures temporal dependencies in the packet flow.

The outputs from both branches are merged using a **Transformer Fusion mechanism** before being passed through dense layers for the final classification.

## 📈 Model Performance

The model achieves state-of-the-art performance in network traffic classification. The results were evaluated on the test set of the UNSW-NB15 dataset.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 91.13% |
| **Precision**| 0.9140 |

## 🏁 Getting Started

### ☁️ Option 1: Deploy to the Cloud with Terraform and Kubernetes (Production)

This method provisions the entire infrastructure on GCP and deploys the application.

**Prerequisites:**

* Google Cloud SDK (`gcloud`) installed and authenticated.
* Terraform installed.
* `kubectl` installed.

**Steps:**

1.  **Provision the Infrastructure**:
    Navigate to the `terraform/` directory and run the following commands to create the GKE cluster, Cloud SQL database, and other resources.
    ```bash
    cd terraform
    terraform init
    terraform apply
    ```
2.  **Connect to the GKE Cluster**:
    Configure `kubectl` to communicate with your new cluster.
    ```bash
    gcloud container clusters get-credentials [YOUR_CLUSTER_NAME] --region [YOUR_REGION]
    ```
3.  **Create the Database Secret**:
    Before deploying, you must create the Kubernetes secret that the application will use to connect to the Cloud SQL database.
    ```bash
    kubectl create secret generic db-credentials \
      --from-literal=username=[YOUR_DB_USER] \
      --from-literal=password=[YOUR_DB_PASSWORD]
    ```
4.  **Deploy the Application**:
    From the project root, apply the Kubernetes manifests to deploy all services (API, MLflow, etc.).
    ```bash
    kubectl apply -f k8s/
    ```
5.  **Access the Services**:
    Find the external IP address of your API's LoadBalancer service to start making predictions.
    ```bash
    kubectl get services
    ```

### 🐳 Option 2: Run Locally with Docker (Development)

This is the easiest way to run the entire application stack on your local machine.

**Prerequisites:**

* Docker and Docker Compose.
* Kaggle API Credentials: Your `kaggle.json` file must be in `~/.kaggle/`.

**Steps:**

1.  **Clone the repository and set up the environment**:
    ```bash
    git clone [https://github.com/esilvei/SynapticIDS/](https://github.com/esilvei/SynapticIDS/)
    cd SynapticIDS
    cp .env.example .env # Create .env and customize if needed
    ```
2.  **Train the Model (One-Time Task)**:
    This command builds the Docker image and runs the training script. The final model is registered in the local MLflow instance.
    ```bash
    docker-compose run --build training
    ```
3.  **Run the Application**:
    This starts the API, MLflow UI, Redis, and PostgreSQL database.
    ```bash
    docker-compose up
    ```

**Local Endpoints:**

* **SynapticIDS API**: `http://localhost:8000`
* **API Docs (Swagger UI)**: `http://localhost:8000/docs`
* **MLflow Dashboard**: `http://localhost:5000`

---

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
curl -X POST "http://[YOUR_API_IP_OR_LOCALHOST]:8000/predictions/" \
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

## 🗺️ Roadmap

This section outlines the future plan for evolving SynapticIDS into a fully autonomous, self-improving security system.

### Phase 1: Advanced MLOps and Data Management 📊

- **Data and Model Versioning with DVC:** Integrate DVC (Data Version Control) to work alongside Git, enabling versioning of large datasets and models for full experiment reproducibility.
- **Automated Retraining Pipeline:** Design a system for automated model retraining that monitors for new data, triggers training jobs on GKE, evaluates new models, and flags them for promotion in the MLflow Model Registry if performance improves.

---

### Phase 2: Security Hardening and Continuous Improvement 🔐

- **Implement API Security:**
  - **Authentication 🔑:** Integrate OAuth2 with JWT tokens to secure all endpoints.
  - **Authorization 🛡️:** Implement role-based access control (RBAC) to restrict access to sensitive endpoints.
- **Establish a Data Feedback Loop 🔄:**
  - **Log Production Inputs 📝:** Create a secure mechanism to capture and store the traffic records sent to the prediction endpoint.
  - **Data Labeling Interface 🏷️:** Develop a simple internal tool for security analysts to review and label the captured traffic, which will be used for the next generation of model training.

---

### Phase 3: Continuous Learning and Automated Data Pipelines 🚀

- **Automated Network Data Ingestion:** Develop a service that automatically detects and captures live network traffic, preparing it for the processing pipeline.
- **Data Monitoring and Orchestration with Airflow:** Implement Apache Airflow to create, schedule, and monitor data engineering pipelines, managing the flow of data from ingestion to storage.
- **Continuous Learning Loop:** Connect the automated data pipelines with the retraining system. New, labeled data will automatically trigger model evaluation and potential redeployment, allowing the IDS to adapt to emerging threats without manual intervention.
