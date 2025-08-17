# SynapticIDS: A Hybrid Deep Learning Intrusion Detection System 🛡️

## 🚀 Overview

SynapticIDS is an Intrusion Detection System (IDS) that leverages a hybrid deep learning architecture to identify and classify (binary or multiclass) security threats in computer networks. The model is designed to learn from complex network traffic data, combining different feature modalities to achieve high accuracy and robustness in attack detection.

This project uses **MLflow** 📈 for robust management of the entire model lifecycle. For development, this project uses **UV** ⚡ for high-speed dependency management and enforces code quality with **Pre-Commit** hooks ✅, including `ruff`, `pylint`, and `pytest`.

***

## 🧠 Model Architecture

The core of SynapticIDS is a hybrid model that processes network data in two parallel branches:

1.  **🖼️ 2D Convolutional Branch**: Selected features from network traffic are transformed into a 2D representation, similar to an image. A Convolutional Neural Network (CNN) is used to extract spatial patterns from this representation.
2.  **📉 Sequential Branch (GRU)**: Another set of features is treated as a time series. A Recurrent Neural Network, specifically using Gate Recurrent Unit (GRU), is used to capture temporal dependencies and sequential patterns in the packet flow.

The outputs from both branches are then merged using a **Transformer Fusion** mechanism before being passed through dense layers for the final classification, determining whether the traffic is benign or malicious.

***
## 🏆 Model Performance

The model was evaluated on the UNSW-NB15 test set, achieving strong results in binary classification (Normal vs. Attack). The key metrics are summarized below:

| Metric         | Score |
| :------------- | :---- |
| **Accuracy** | 90.7% |
| **F1-Score** | 90.8% |
| **Precision** | 91.2% |
| **Recall** | 90.7% |

These results demonstrate the model's high effectiveness in distinguishing between benign and malicious network traffic.

***

## ✨ Implemented Features

* **📦 Complete Data Pipeline**: Scripts for loading, preprocessing, feature engineering, and transforming data from the UNSW-NB15 dataset.
* **🤖 Hybrid Model (CNN+GRU)**: A custom deep learning architecture implemented in Keras/TensorFlow.
* **📊 Experiment Management**: Full integration with **MLflow** to track parameters, metrics, artifacts (plots, reports), and save models from each run.
* **🗂️ Model Registry**: Trained models are registered in the MLflow Model Registry, allowing for versioning and easy access for deployment.
* **📈 Results Analysis**: Automatic generation of classification reports, confusion matrices, and training history plots.

***

## 🏁 Getting Started

### Prerequisites

* Python 3.10+
* **UV** ⚡ (for package and environment management)

### 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/esilvei/SynapticIDS/
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

    # Install dependencies from requirements.txt
    uv sync
    ```

3.  **Set up the Pre-Commit Hooks:**
    * This ensures that code quality checks (with `ruff`, `pylint`) and tests (with `pytest`) are run automatically before each commit.
    ```bash
    pre-commit install
    ```

4.  **Set up the dataset:**
    * Download the UNSW-NB15 dataset (training and testing sets).
    * Place the `.parquet` files into the `data/raw/` directory.

### 🏋️‍♀️ Running the Training

To start a full cycle of training, evaluation, and model registration, run the following command from the project root:

```bash
python src/synaptic_ids/training/run_training.py
```

The script will execute the entire pipeline, and at the end, a new experiment will be registered in MLflow.

***

## 🖥️ Using the MLflow UI

To visualize the results of your experiments, such as comparing metrics, parameters, and generated artifacts, you can use the MLflow web interface.

Because this project is configured to use a SQLite database as its tracking backend, it is **essential** to specify the path to this database when launching the UI. This prevents "malformed experiment" errors.

Run the following command in the project's root directory:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

After running the command, open `http://127.0.0.1:5000` in your browser to see the MLflow dashboard.

***

## 🚀 Next Steps (Future Development)

* **🤖 Inference API**: Package the best-performing model from the MLflow Registry and deploy it as a REST API (using FastAPI or Flask) to allow for real-time inference.
* **☁️ Containerization and Cloud Deployment**: Containerize the inference API using **Docker** 🐳 and deploy it on a scalable, managed **Kubernetes** ☸️ cluster (such as Google Kubernetes Engine) on **Google Cloud Platform (GCP)**.
* **✅ Testing and CI/CD**: Expand the test suite and set up a Continuous Integration/Continuous Deployment (CI/CD) pipeline with GitHub Actions to automate testing and future deployments.
