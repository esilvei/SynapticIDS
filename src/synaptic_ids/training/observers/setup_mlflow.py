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
    tracking_uri_from_env = os.getenv("MLFLOW_TRACKING_URI")
    client = MlflowClient()
    experiment_name = "SynapticIDS"

    if tracking_uri_from_env:
        # Docker Configuration
        mlflow.set_tracking_uri(tracking_uri_from_env)
        print(f"MLflow tracking URI set from environment: {tracking_uri_from_env}")

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found. Creating it now.")
            client.create_experiment(name=experiment_name)
    else:
        # Local Configuration
        project_root = next(
            p
            for p in Path(__file__).resolve().parents
            if (p / "pyproject.toml").exists()
        )
        mlruns_path = project_root / "mlruns"
        os.makedirs(mlruns_path, exist_ok=True)

        db_path = mlruns_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.resolve()}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set for local: {tracking_uri}")

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found. Creating it now.")
            correct_artifact_location = mlruns_path.resolve().as_uri()
            client.create_experiment(
                name=experiment_name, artifact_location=correct_artifact_location
            )
            print(
                f"Experiment created with artifact location: {correct_artifact_location}"
            )
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: '{experiment_name}'")
