import pytest
import pandas as pd
import numpy as np

# Import the class to be tested
from src.synaptic_ids.processing.data_transformer.to_sequence import SequenceGenerator


@pytest.fixture
def sequence_generator_instance():
    """Provides a standard instance of the SequenceGenerator."""
    return SequenceGenerator(sequence_length=5)


@pytest.fixture
def sample_batch_data():
    """Provides sample data for offline/batch processing (10 rows)."""
    # Define which temporal features will be in our test data
    temporal_features = ["dur", "sbytes", "dbytes", "rate"]
    other_features = ["other_feature_1", "other_feature_2"]

    # Create a DataFrame with 10 rows
    data = pd.DataFrame(
        np.arange(10 * len(temporal_features + other_features)).reshape(10, -1),
        columns=temporal_features + other_features,
    )
    # Create corresponding labels
    labels = pd.Series(np.arange(100, 110))  # Labels from 100 to 109
    return data, labels


## Test Suite for Offline (Batch) Mode


def test_generate_offline_correct_shape_and_count(
    sequence_generator_instance, sample_batch_data
):
    """
    Tests that offline generation produces the correct number and shape of sequences.
    """
    # Arrange
    generator = sequence_generator_instance
    x_data, y_data = sample_batch_data

    # Act
    sequences, labels, valid_indices = generator.generate(x_data, y_data)

    # Assert
    # For 10 data points and sequence_length=5, we expect 10 - 5 + 1 = 6 sequences.
    assert sequences.shape == (
        6,
        5,
        4,
    )  # (num_sequences, seq_length, num_temporal_features)
    assert labels.shape == (6,)
    assert valid_indices.shape == (6,)
    assert sequences.dtype == np.float32
    assert labels.dtype == np.int32


def test_generate_offline_correct_labeling(
    sequence_generator_instance, sample_batch_data
):
    """
    Tests that the label for each sequence corresponds to its last element.
    """
    # Arrange
    generator = sequence_generator_instance
    x_data, y_data = sample_batch_data  # Original labels are 100, 101, ..., 109

    # Act
    _, labels, _ = generator.generate(x_data, y_data)

    # Assert
    # The first sequence is from index 0-4, so its label should be y[4] = 104.
    # The second sequence is from index 1-5, so its label should be y[5] = 105.
    # ...and so on.
    expected_labels = np.array([104, 105, 106, 107, 108, 109])
    np.testing.assert_array_equal(labels, expected_labels)


def test_generate_offline_with_insufficient_data(sequence_generator_instance):
    """
    Tests that generation returns empty arrays if data is shorter than sequence length.
    """
    # Arrange
    generator = sequence_generator_instance  # sequence_length is 5
    # Create a DataFrame with only 4 rows
    x_short = pd.DataFrame(np.zeros((4, 3)), columns=["dur", "sbytes", "dbytes"])
    y_short = pd.Series([0, 1, 0, 1])

    # Act
    sequences, labels, valid_indices = generator.generate(x_short, y_short)

    # Assert
    assert sequences.size == 0
    assert labels.size == 0
    assert valid_indices.size == 0


## Test Suite for Online (Single Instance) Mode


def test_generate_online_correct_shape_and_replication(sequence_generator_instance):
    """
    Tests that online mode replicates the single data point to form a sequence.
    """
    # Arrange
    generator = sequence_generator_instance
    # Create a single-row DataFrame
    x_single = pd.DataFrame(
        [[1, 2, 3, 4]], columns=["dur", "sbytes", "dbytes", "rate"], index=[99]
    )
    y_single = pd.Series([1], index=[99])  # Label is 1, index is 99

    # Act
    sequences, labels, valid_indices = generator.generate(x_single, y_single)

    # Assert
    # The output should have a batch size of 1
    assert sequences.shape == (1, 5, 4)  # (batch_size, seq_length, num_features)
    assert labels.shape == (1,)
    assert valid_indices.shape == (1,)

    # Check that the sequence is just the input row repeated 5 times
    expected_sequence_contents = np.tile([1, 2, 3, 4], (5, 1))
    np.testing.assert_array_equal(sequences[0], expected_sequence_contents)
    assert labels[0] == 1
    assert valid_indices[0] == 99


## Test for Feature Filtering


def test_generate_handles_missing_temporal_features(sequence_generator_instance):
    """
    Tests that the generator only uses available temporal features and doesn't fail.
    """
    # Arrange
    generator = sequence_generator_instance
    # Create data with only 2 of the generator's defined temporal features
    x_data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["dur", "sload"])
    y_data = pd.Series(np.zeros(10))

    # Act
    sequences, _, _ = generator.generate(x_data, y_data)

    # Assert
    # The number of features in the output sequence should be 2, not the full list.
    assert sequences.shape == (6, 5, 2)
