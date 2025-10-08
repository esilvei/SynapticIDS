from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.synaptic_ids.processing.data_transformer.data_preparer import DataPreparer

# --- Fixtures to set up the test environment ---


@pytest.fixture
def mock_feature_engineer():
    """Mocks the UNSWNB15FeatureEngineer with essential attributes and methods."""
    engineer = MagicMock()
    # Mock attributes that are set after fitting
    engineer.final_selected_features = ["feat1", "feat2", "feat3"]
    engineer.label_encoder.classes_ = [0, 1, 2]  # For multiclass testing
    # Mock the transform method to return predictable data
    x_eng = pd.DataFrame(np.random.rand(20, 3), columns=["feat1", "feat2", "feat3"])
    y_eng = pd.Series(np.random.randint(0, 3, 20))
    engineer.transform.return_value = (x_eng, y_eng)
    return engineer


@pytest.fixture
def sample_dataframe():
    """Provides a basic DataFrame for fitting and preparing."""
    return pd.DataFrame({"featureA": range(20), "featureB": range(20, 40)})


# --- Test Suite ---


## 1. Testing the `fit` method
def test_fit_method_trains_dependencies(mock_feature_engineer, sample_dataframe):
    """
    Tests that fit() correctly trains the feature engineer and initializes the image transformer.
    """
    # Arrange
    preparer = DataPreparer(feature_engineer=mock_feature_engineer)

    # Act
    preparer.fit(sample_dataframe)

    # Assert
    mock_feature_engineer.fit.assert_called_once_with(sample_dataframe)
    assert preparer.image_transformer is not None
    assert preparer.image_transformer.feature_names == ["feat1", "feat2", "feat3"]

    assert preparer.is_fitted is True


## 2. Testing the `prepare_data` method
@patch("src.synaptic_ids.processing.data_transformer.data_preparer.SequenceGenerator")
async def test_prepare_data_happy_path(
    mock_sequence_generator, mock_feature_engineer, sample_dataframe
):
    """
    Tests the main data preparation pipeline in 'binary' mode.
    """
    # Arrange
    mock_seq_gen_instance = mock_sequence_generator.return_value
    mock_sequences = np.random.rand(16, 5, 3)  # (num_seq, len_seq, num_feat)
    mock_labels = np.random.randint(0, 2, 16)
    mock_indices = np.arange(4, 20)
    mock_seq_gen_instance.generate_offline.return_value = (
        mock_sequences,
        mock_labels,
        mock_indices.tolist(),
    )

    preparer = DataPreparer(feature_engineer=mock_feature_engineer, mode="binary")
    preparer.fit(sample_dataframe)

    preparer.image_transformer = MagicMock(
        return_value=tf.random.uniform((16, 32, 32, 1))
    )

    # Act
    result = await preparer.prepare_data(sample_dataframe, is_training=True)

    # Assert
    assert mock_feature_engineer.transform.call_args[1]["is_inference"] is False
    mock_seq_gen_instance.generate_offline.assert_called_once()
    preparer.image_transformer.assert_called_once()

    # Verify output dictionary
    assert "images" in result
    assert "sequences" in result
    assert "labels" in result

    # Verify shapes
    assert result["images"].shape == (16, 32, 32, 1)
    assert result["sequences"].shape == (16, 5, 3)
    assert result["labels"].shape == (16,)  # Binary mode, no one-hot encoding


@patch("src.synaptic_ids.processing.data_transformer.data_preparer.SequenceGenerator")
async def test_prepare_data_inference_mode(
    mock_sequence_generator, mock_feature_engineer, sample_dataframe
):
    """
    Tests the data preparation pipeline for inference, where no labels are provided.
    """
    # --- Arrange ---
    # 1. Simulate a preparer that has been fitted during training
    preparer = DataPreparer(feature_engineer=mock_feature_engineer, mode="multiclass")
    preparer.fit(sample_dataframe)  # Fit to populate internal artifacts
    preparer.image_transformer = MagicMock(
        return_value=tf.random.uniform((16, 32, 32, 1))
    )

    # 2. Configure mocks for the inference flow
    x_eng_inference = pd.DataFrame(
        np.random.rand(19, 3), columns=["feat1", "feat2", "feat3"]
    )
    mock_feature_engineer.transform.return_value = (x_eng_inference, None)

    # The sequence generator will also return labels=None
    mock_seq_gen_instance = mock_sequence_generator.return_value
    mock_sequences = np.random.rand(15, 5, 3)
    mock_indices = np.arange(4, 19).tolist()
    mock_seq_gen_instance.generate_offline.return_value = (
        mock_sequences,
        None,
        mock_indices,
    )

    # 3. Create an inference DataFrame
    inference_df = sample_dataframe.copy()

    # --- Act ---
    result = await preparer.prepare_data(inference_df, is_training=False)

    # --- Assert ---
    # Verify that the transform method was called with the correct arguments
    # `call_args` gets the arguments of the last call to the mock
    # The first positional argument (index 0) is the DataFrame
    pd.testing.assert_frame_equal(
        mock_feature_engineer.transform.call_args[0][0], inference_df
    )
    # The keyword argument 'is_inference' should be True
    assert mock_feature_engineer.transform.call_args[1]["is_inference"] is True

    # Verify the output does not contain labels
    assert "labels" in result
    assert result["labels"] is None

    # Verify that features were generated correctly
    assert result["images"].shape == (16, 32, 32, 1)
    assert result["sequences"].shape == (15, 5, 3)


async def test_prepare_data_multiclass_encoding(
    mock_feature_engineer, sample_dataframe
):
    """
    Tests that labels are correctly one-hot encoded in 'multiclass' mode.
    """
    # Arrange
    preparer = DataPreparer(feature_engineer=mock_feature_engineer, mode="multiclass")
    preparer.fit(sample_dataframe)
    preparer.image_transformer = MagicMock(return_value=tf.zeros((1, 32, 32, 1)))

    # Act
    result = await preparer.prepare_data(sample_dataframe, is_training=True)

    # Assert
    # The label encoder in the mock has 3 classes, so shape should be (num_labels, 3)
    # The exact number of labels depends on the random data, so we just check the second dimension.
    assert result["labels"].shape[1] == 3


## 3. Testing Edge Cases and Error Handling
async def test_prepare_data_raises_error_if_not_fitted(mock_feature_engineer):
    """
    Tests that calling prepare_data before fit raises a RuntimeError.
    """
    # Arrange
    preparer = DataPreparer(feature_engineer=mock_feature_engineer)

    # Act & Assert
    with pytest.raises(RuntimeError, match="DataPreparer must be fitted"):
        await preparer.prepare_data(pd.DataFrame())


@patch("src.synaptic_ids.processing.data_transformer.data_preparer.SequenceGenerator")
async def test_prepare_data_handles_no_sequences(
    mock_sequence_generator, mock_feature_engineer, sample_dataframe
):
    """
    Tests that the method returns empty arrays if the sequence generator produces no sequences.
    """
    # Arrange
    mock_seq_gen_instance = mock_sequence_generator.return_value
    mock_seq_gen_instance.generate_offline.return_value = (np.array([]), None, [])

    preparer = DataPreparer(feature_engineer=mock_feature_engineer)
    preparer.fit(sample_dataframe)

    # Act
    result = await preparer.prepare_data(sample_dataframe)

    # Assert
    assert result["images"].size == 0
    assert result["sequences"].size == 0
    assert result["labels"] is None
