from unittest.mock import ANY, patch, AsyncMock

import numpy as np
import pytest

from src.synaptic_ids.training.run_training import (
    build_and_train_model,
    main,
    prepare_data,
    setup_and_load_data,
)


@patch("src.synaptic_ids.training.run_training.DataLoader")
@patch("src.synaptic_ids.training.run_training.DataSetup")
def test_setup_and_load_data(mock_data_setup, mock_data_loader):
    """
    Verifies that the function correctly instantiates and calls DataSetup and DataLoader.
    """
    # Arrange: Configure the behavior of the mocks
    mock_setup_instance = mock_data_setup.return_value
    mock_setup_instance.setup_dataset.return_value = "/fake/dataset/path"

    mock_loader_instance = mock_data_loader.return_value
    mock_loader_instance.load_and_split.return_value = ("train_df", "val_df", "test_df")

    # Act: Execute the function
    train, val, test = setup_and_load_data()

    # Assert: Check if the classes were called as expected
    mock_data_setup.assert_called_once_with(dataset_name=ANY, download_path=ANY)
    mock_setup_instance.setup_dataset.assert_called_once()

    mock_data_loader.assert_called_once_with(
        dataset_dir="/fake/dataset/path",
        target_col=ANY,
        test_size=ANY,
        val_size=ANY,
    )
    mock_loader_instance.load_and_split.assert_called_once()

    assert (train, val, test) == ("train_df", "val_df", "test_df")


# --- Test for prepare_data ---
@pytest.mark.asyncio
@patch("src.synaptic_ids.training.run_training.DataPreparer")
@patch("src.synaptic_ids.training.run_training.UNSWNB15FeatureEngineer")
async def test_prepare_data(mock_feature_engineer, mock_data_preparer):
    """
    Verifies that the function orchestrates the FeatureEngineer and DataPreparer.
    """
    # Arrange
    mock_preparer = mock_data_preparer.return_value
    mock_preparer.prepare_data = AsyncMock(
        side_effect=["train_data", "val_data", "test_data"]
    )

    # Act
    train, val, test, preparer = await prepare_data("train_df", "val_df", "test_df")

    # Assert
    mock_feature_engineer.assert_called_once_with(
        mode=ANY, target_col=ANY, selected_features=ANY
    )
    mock_data_preparer.assert_called_once_with(
        feature_engineer=mock_feature_engineer.return_value, mode=ANY
    )

    mock_preparer.fit.assert_called_once_with("train_df")
    assert mock_preparer.prepare_data.call_count == 3

    assert (train, val, test, preparer) == (
        "train_data",
        "val_data",
        "test_data",
        mock_preparer,
    )


# --- Test for build_and_train_model ---
@patch("src.synaptic_ids.training.run_training.ModelTrainer")
@patch("src.synaptic_ids.training.run_training.SynapticIDSBuilder")
def test_build_and_train_model_multiclass(mock_builder, _mock_trainer, monkeypatch):
    """
    Verifies the model building logic for the MULTICLASS case.
    """
    # Arrange
    monkeypatch.setattr("src.synaptic_ids.config.settings.training.mode", "multiclass")

    fake_train_data = {
        "images": np.zeros((100, 32, 32, 1)),
        "sequences": np.zeros((100, 5)),
        "labels": np.zeros((100, 10)),  # Labels are one-hot encoded (10 classes)
    }

    # Act
    build_and_train_model(fake_train_data, "val_data", "preparer")

    # Assert: Check that num_classes is correctly calculated as 10
    mock_builder.assert_called_once_with(
        image_shape=(32, 32, 1), sequence_shape=(5,), num_classes=10, mode="multiclass"
    )


@patch("src.synaptic_ids.training.run_training.ModelTrainer")
@patch("src.synaptic_ids.training.run_training.SynapticIDSBuilder")
def test_build_and_train_model_binary(mock_builder, _mock_trainer, monkeypatch):
    """
    Verifies the model building logic for the BINARY case.
    """
    # Arrange
    monkeypatch.setattr("src.synaptic_ids.config.settings.training.mode", "binary")

    fake_train_data = {
        "images": np.zeros((100, 32, 32, 1)),
        "sequences": np.zeros((100, 5)),
        "labels": np.zeros(100),  # Labels are flat (0s and 1s)
    }

    # Act
    build_and_train_model(fake_train_data, "val_data", "preparer")

    # Assert: Check that num_classes is correctly calculated as 1
    mock_builder.assert_called_once_with(
        image_shape=(32, 32, 1), sequence_shape=(5,), num_classes=1, mode="binary"
    )


def test_build_and_train_model_no_data():
    """Verifies that the function returns None if there is no training data."""
    # Arrange
    fake_train_data = {"images": []}

    # Act
    result = build_and_train_model(fake_train_data, "val_data", "preparer")

    # Assert
    assert result == (None, None)


# --- Test for the main flow ---
@patch("src.synaptic_ids.training.run_training.setup_and_load_data")
@patch("src.synaptic_ids.training.run_training.prepare_data")
@patch("src.synaptic_ids.training.run_training.build_and_train_model")
@patch("src.synaptic_ids.training.run_training.evaluate_and_log_results")
@patch("src.synaptic_ids.training.run_training.MLflowObserver")
@patch("mlflow.start_run")
@patch("mlflow.set_experiment")
def test_main_pipeline_flow(
    _mock_set_exp,
    _mock_start_run,
    mock_observer,
    mock_evaluate,
    mock_build,
    mock_prepare,
    mock_setup,
):
    """
    Tests the end-to-end flow of the main function, checking that all steps
    are called in the correct order.
    """
    # Arrange
    mock_setup.return_value = ("train_df", "val_df", "test_df")
    mock_prepare.return_value = ("train_data", "val_data", "test_data", "preparer")
    mock_build.return_value = ("trainer", "history")

    # Act
    main()

    # Assert
    mock_setup.assert_called_once()
    mock_prepare.assert_called_once_with("train_df", "val_df", "test_df")
    mock_build.assert_called_once_with("train_data", "val_data", "preparer")
    mock_evaluate.assert_called_once_with(
        "trainer",
        "history",
        "test_data",
        "preparer",
        "test_df",
        mock_observer.return_value,
    )
