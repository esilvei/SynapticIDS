from unittest.mock import MagicMock, ANY
import pytest
import mlflow

from src.synaptic_ids.training.observers.setup_mlflow import setup_mlflow_local


@pytest.fixture
def mock_mlflow_dependencies(monkeypatch, tmp_path):
    """
    Pytest fixture to mock all external dependencies:
    - Simulates a fake project structure using tmp_path.
    - Replaces MLflow and file system functions.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    project_root = tmp_path
    (project_root / "pyproject.toml").touch()

    fake_script_path = project_root / "src/observers/setup_mlflow.py"
    fake_script_path.parent.mkdir(parents=True, exist_ok=True)
    fake_script_path.touch()
    monkeypatch.setattr(
        "src.synaptic_ids.training.observers.setup_mlflow.__file__",
        str(fake_script_path),
    )

    # Mock the top-level MLflow functions
    mock_set_tracking_uri = MagicMock()
    mock_set_experiment = MagicMock()
    mock_get_experiment_by_name = MagicMock()
    mock_create_experiment = MagicMock()

    monkeypatch.setattr(mlflow, "set_tracking_uri", mock_set_tracking_uri)
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_experiment)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", mock_get_experiment_by_name)
    monkeypatch.setattr(mlflow, "create_experiment", mock_create_experiment)

    return {
        "project_root": project_root,
        "set_tracking_uri": mock_set_tracking_uri,
        "set_experiment": mock_set_experiment,
        "get_experiment_by_name": mock_get_experiment_by_name,
        "create_experiment": mock_create_experiment,
    }


def test_setup_mlflow_creates_experiment_if_not_exists(mock_mlflow_dependencies):
    """
    Verifies that the function creates the MLflow directory and experiment
    when it does not exist.
    """
    # --- Arrange ---
    mock_mlflow_dependencies["get_experiment_by_name"].return_value = None

    # --- Act ---
    setup_mlflow_local()

    # --- Assert ---
    project_root = mock_mlflow_dependencies["project_root"]
    assert (project_root / "mlruns").is_dir()

    mock_mlflow_dependencies["set_tracking_uri"].assert_called_once()
    call_args = mock_mlflow_dependencies["set_tracking_uri"].call_args[0]
    assert call_args[0].startswith("sqlite:///")
    assert "mlflow.db" in call_args[0]

    mock_mlflow_dependencies["get_experiment_by_name"].assert_called_once_with(
        "SynapticIDS"
    )

    mock_mlflow_dependencies["create_experiment"].assert_called_once_with(
        name="SynapticIDS",
        artifact_location=ANY,
    )
    create_call_kwargs = mock_mlflow_dependencies["create_experiment"].call_args[1]
    assert create_call_kwargs["artifact_location"].startswith("file://")

    mock_mlflow_dependencies["set_experiment"].assert_called_once_with("SynapticIDS")


def test_setup_mlflow_uses_existing_experiment(mock_mlflow_dependencies):
    """
    Verifies that the function uses an existing experiment and does not try
    to create it again.
    """
    # --- Arrange ---
    mock_experiment = MagicMock()
    mock_mlflow_dependencies["get_experiment_by_name"].return_value = mock_experiment

    # --- Act ---
    setup_mlflow_local()

    # --- Assert ---
    assert (mock_mlflow_dependencies["project_root"] / "mlruns").is_dir()
    mock_mlflow_dependencies["set_tracking_uri"].assert_called_once()

    mock_mlflow_dependencies["get_experiment_by_name"].assert_called_once_with(
        "SynapticIDS"
    )

    mock_mlflow_dependencies["create_experiment"].assert_not_called()

    mock_mlflow_dependencies["set_experiment"].assert_called_once_with("SynapticIDS")
