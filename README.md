# DeepHybrid-IDS: AI-Powered Cyber-Attack Classification with MLOps on GCP

This project implements an end-to-end system to classify network cyber-attacks using a hybrid Deep Learning model. The solution transforms an analytical prototype (initially developed in a Jupyter Notebook) into a robust, scalable, and automated web service on the Google Cloud Platform (GCP), following modern MLOps best practices.

**Workflow:**
1.  Code is pushed to the **GitHub** repository.
2.  **GitHub Actions** (CI/CD) is triggered, building the application's Docker image.
3.  The image is pushed to **Artifact Registry**.
4.  The new image is deployed to **Cloud Run**.
5.  The Cloud Run service loads the trained model from **Cloud Storage** upon startup.
6.  Clients send prediction requests to the API endpoint.
7.  Prediction requests and results are logged to **Cloud SQL** for monitoring and analysis.

## ✨ Features

* **Prediction API:** A RESTful API built with **FastAPI** for real-time predictions.
* **Hybrid Model:** Leverages a Deep Learning model for high-accuracy classification of network traffic.
* **Infrastructure as Code (IaC):** The entire GCP infrastructure is managed by **Terraform**, ensuring consistency and reproducibility.
* **Automated CI/CD:** A continuous integration and delivery pipeline with **GitHub Actions** for automated builds, tests, and deployments.
* **Containerization:** The application is packaged with **Docker**, ensuring a consistent and portable execution environment.
* **Persistence & Logging:** Predictions are logged to a **PostgreSQL (Cloud SQL)** database for auditing and performance analysis.
* **Serverless & Scalable:** The API is hosted on **Cloud Run**, which automatically scales with demand, including scaling to zero to save costs.

## 🚀 Tech Stack

* **Language:** Python 3.12.7
* **API Framework:** FastAPI
* **Machine Learning:** TensorFlow/Keras, Scikit-learn, Pandas
* **Containerization:** Docker
* **Cloud:** Google Cloud Platform (GCP)
    * **API Service:** Cloud Run
    * **Image Registry:** Artifact Registry
    * **Database:** Cloud SQL (PostgreSQL)
    * **Artifact Storage:** Cloud Storage
* **CI/CD:** GitHub Actions
* **Infrastructure as Code:** Terraform
