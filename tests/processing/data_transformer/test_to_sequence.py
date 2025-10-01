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
    """
    Provides a sample DataFrame (x) and Series (y) with 10 rows.
    'dur', 'sbytes', 'dbytes', 'rate' are temporal features.
    """
    x = pd.DataFrame(
        {
            "dur": range(10),
            "sbytes": range(1, 11),
            "dbytes": range(2, 12),
            "rate": range(3, 13),
            "other_feature_1": range(4, 14),
            "other_feature_2": range(5, 15),
        }
    )
    y = pd.Series(range(100, 110))
    return x, y


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
    assert len(valid_indices) == 6
    np.testing.assert_array_equal(labels, [104, 105, 106, 107, 108, 109])
    np.testing.assert_array_equal(valid_indices, [4, 5, 6, 7, 8, 9])


def test_generate_offline_correct_labeling(
    sequence_generator_instance, sample_batch_data
):
    """
    Tests that the label for each sequence corresponds to its last element.
    """
    # Arrange
    generator = sequence_generator_instance
    x_data, y_data = sample_batch_data

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


def test_generate_online_mode_with_batch_no_labels(
    sequence_generator_instance, sample_batch_data
):
    """
    Tests that online generation mode (triggered by y=None) works for a batch.
    """
    # Arrange
    generator = sequence_generator_instance
    x_data, _ = sample_batch_data

    # Act: Call the function with y=None to trigger online mode
    sequences, labels, valid_indices = generator.generate(x_data, y=None)

    # Assert
    # In online mode, each of the 10 samples is replicated into a sequence.
    assert sequences.shape == (10, 5, 4)
    # The labels output should be None
    assert labels is None
    # Valid indices should correspond to all 10 samples
    assert len(valid_indices) == 10
    np.testing.assert_array_equal(valid_indices, np.arange(10))


def test_generate_online_correct_shape_and_replication(sequence_generator_instance):
    """
    Tests that online mode replicates a single data point to form a sequence.
    """
    # Arrange
    generator = sequence_generator_instance
    x_single = pd.DataFrame([[1, 2, 3, 4]], columns=["dur", "sbytes", "dbytes", "rate"])

    # Act
    # Call generate with y=None to trigger online mode for a single instance
    sequences, labels, valid_indices = generator.generate(x_single, y=None)

    # Assert
    # The output should have a batch size of 1
    assert sequences.shape == (1, 5, 4)  # (batch_size, seq_length, num_features)
    assert labels is None  # In online mode (y=None), labels should be None
    assert valid_indices.shape == (1,)

    # Check that the sequence is the input row repeated 5 times
    expected_sequence_contents = np.tile([1, 2, 3, 4], (5, 1))
    np.testing.assert_array_equal(sequences[0], expected_sequence_contents)
    # The current implementation returns np.arange(n_samples), so the index will be 0.
    assert valid_indices[0] == 0


## Test for Feature Filtering


def test_generate_handles_missing_temporal_features(sequence_generator_instance):
    """
    Tests that the generator only uses available temporal features and doesn't fail.
    """
    # Arrange
    generator = sequence_generator_instance
    x_data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["dur", "sload"])
    y_data = pd.Series(np.zeros(10))

    # Act
    sequences, _, _ = generator.generate(x_data, y_data)

    # Assert
    # The number of features in the output sequence should be 2.
    assert sequences.shape == (6, 5, 2)
