import os
from pathlib import Path
import mlflow


def setup_mlflow_local():
    """
    Configures MLflow for local, test, or server environments.

    This function prioritizes the MLFLOW_TRACKING_URI environment variable,
    making it ideal for isolated test runs (e.g., pre-commit hooks, CI/CD).
    If the variable is not set, it falls back to a default local setup
    using a SQLite database for tracking.
    """
    experiment_name = "SynapticIDS"

    # Priority 1: Use MLFLOW_TRACKING_URI from environment (for tests and containers).
    tracking_uri_from_env = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri_from_env:
        mlflow.set_tracking_uri(tracking_uri_from_env)
        print(f"MLflow tracking URI set from environment: {tracking_uri_from_env}")

        # For file-based or HTTP-based storage, set_experiment will create the
        # experiment if it doesn't exist. This avoids a direct client call
        # that can hang during tests.
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: '{experiment_name}'")
        return

    # Priority 2: Default local setup (for developers running training manually).
    try:
        project_root = next(
            p
            for p in Path(__file__).resolve().parents
            if (p / "pyproject.toml").exists()
        )
        mlruns_path = project_root / "mlruns"
        mlruns_path.mkdir(exist_ok=True)

        db_path = mlruns_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.resolve()}"

        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set for local development: {tracking_uri}")

        if not mlflow.get_experiment_by_name(experiment_name):
            print(f"Experiment '{experiment_name}' not found. Creating it now.")
            artifact_location = mlruns_path.resolve().as_uri()
            mlflow.create_experiment(
                name=experiment_name, artifact_location=artifact_location
            )

        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: '{experiment_name}'")

    except StopIteration:
        print(
            "Error: Could not find project root (pyproject.toml). MLflow not configured."
        )
