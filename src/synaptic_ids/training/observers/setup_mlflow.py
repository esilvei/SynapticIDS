import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow_local():
    """
    Configures MLflow for a robust, serverless local setup.

    This function sets up a local SQLite database for tracking and ensures
    artifacts are stored correctly on the local filesystem.
    """
    # Find project root reliably
    project_root = next(
        p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )
    mlruns_path = project_root / "mlruns"
    os.makedirs(mlruns_path, exist_ok=True)

    # 1. Set tracking URI to a local SQLite database file
    db_path = mlruns_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # 2. Get the experiment and ensure its artifact location is a proper URI
    client = MlflowClient()
    experiment_name = "SynapticIDS"
    experiment = client.get_experiment_by_name(experiment_name)

    # --- THE CRUCIAL FIX IS HERE ---
    # Use .as_uri() to create a guaranteed correct file URI
    correct_artifact_location = mlruns_path.resolve().as_uri()

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Creating it now.")
        # Create the experiment with the correctly formatted artifact location
        client.create_experiment(
            name=experiment_name, artifact_location=correct_artifact_location
        )
        print(f"Experiment created with artifact location: {correct_artifact_location}")

    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: '{experiment_name}'")
