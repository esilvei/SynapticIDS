from unittest.mock import MagicMock, ANY
import pytest


from src.synaptic_ids.training.observers.setup_mlflow import setup_mlflow_local


@pytest.fixture
def mock_mlflow_dependencies(monkeypatch, tmp_path):
    """
    Pytest fixture to mock all external dependencies:
    - Simulates a fake project structure using tmp_path.
    - Replaces MLflow and file system functions.
    """
    # 1. Create a fake project structure for the function to find the "root"
    project_root = tmp_path
    (project_root / "pyproject.toml").touch()

    # Make the function think it's being run from inside the fake project.
    # This robustly handles the Path(__file__).resolve().parents logic.
    fake_script_path = project_root / "src/observers/setup_mlflow.py"
    fake_script_path.parent.mkdir(parents=True, exist_ok=True)
    fake_script_path.touch()
    monkeypatch.setattr(
        "src.synaptic_ids.training.observers.setup_mlflow.__file__",
        str(fake_script_path),
    )

    # 2. Mock the MLflow functions
    mock_set_tracking_uri = MagicMock()
    mock_set_experiment = MagicMock()
    mock_mlflow_client_class = MagicMock()

    monkeypatch.setattr("mlflow.set_tracking_uri", mock_set_tracking_uri)
    monkeypatch.setattr("mlflow.set_experiment", mock_set_experiment)
    monkeypatch.setattr(
        "src.synaptic_ids.training.observers.setup_mlflow.MlflowClient",
        mock_mlflow_client_class,
    )

    # Return the mocks so the tests can inspect them
    return {
        "project_root": project_root,
        "set_tracking_uri": mock_set_tracking_uri,
        "set_experiment": mock_set_experiment,
        "client_class": mock_mlflow_client_class,
    }


def test_setup_mlflow_creates_experiment_if_not_exists(mock_mlflow_dependencies):
    """
    Verifies that the function creates the MLflow directory and experiment
    when it does not exist.
    """
    # --- Arrange ---
    # Configure the MlflowClient mock to simulate that the experiment was not found
    mock_client_instance = mock_mlflow_dependencies["client_class"].return_value
    mock_client_instance.get_experiment_by_name.return_value = None

    # --- Act ---
    setup_mlflow_local()

    # --- Assert ---
    # Check if the mlruns directory was created
    project_root = mock_mlflow_dependencies["project_root"]
    assert (project_root / "mlruns").is_dir()

    # Check if the tracking URI was set correctly
    mock_mlflow_dependencies["set_tracking_uri"].assert_called_once()
    call_args = mock_mlflow_dependencies["set_tracking_uri"].call_args[0]
    assert call_args[0].startswith("sqlite:///")
    assert "mlflow.db" in call_args[0]

    # Check if the client tried to get the experiment
    mock_client_instance.get_experiment_by_name.assert_called_once_with("SynapticIDS")

    # **Check that the experiment was created**
    mock_client_instance.create_experiment.assert_called_once_with(
        name="SynapticIDS",
        artifact_location=ANY,  # We only care that it was called
    )
    # Check the artifact location format
    create_call_kwargs = mock_client_instance.create_experiment.call_args[1]
    assert create_call_kwargs["artifact_location"].startswith("file://")

    # Check that the active experiment was set at the end
    mock_mlflow_dependencies["set_experiment"].assert_called_once_with("SynapticIDS")


def test_setup_mlflow_uses_existing_experiment(mock_mlflow_dependencies):
    """
    Verifies that the function uses an existing experiment and does not try
    to create it again.
    """
    # --- Arrange ---
    # Configure the MlflowClient mock to simulate that the experiment WAS found
    mock_client_instance = mock_mlflow_dependencies["client_class"].return_value
    mock_experiment = MagicMock()  # A fake experiment object
    mock_client_instance.get_experiment_by_name.return_value = mock_experiment

    # --- Act ---
    setup_mlflow_local()

    # --- Assert ---
    # Check that the directory and URI were configured as before
    assert (mock_mlflow_dependencies["project_root"] / "mlruns").is_dir()
    mock_mlflow_dependencies["set_tracking_uri"].assert_called_once()

    # Check that the client tried to get the experiment
    mock_client_instance.get_experiment_by_name.assert_called_once_with("SynapticIDS")

    # **Check that the create_experiment function was NOT called**
    mock_client_instance.create_experiment.assert_not_called()

    # Check that the active experiment was set at the end
    mock_mlflow_dependencies["set_experiment"].assert_called_once_with("SynapticIDS")
