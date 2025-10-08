import json
from unittest.mock import MagicMock, AsyncMock

import pytest
import pandas as pd
import numpy as np

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
    generator = sequence_generator_instance
    x_data, y_data = sample_batch_data
    sequences, labels, valid_indices = generator.generate_offline(x_data, y_data)

    assert sequences.shape == (
        10,
        5,
        4,
    )  # (num_sequences, seq_length, num_temporal_features)
    assert labels.shape == (10,)
    assert len(valid_indices) == 10
    np.testing.assert_array_equal(labels, y_data.values)
    np.testing.assert_array_equal(valid_indices, np.arange(10))


def test_generate_offline_correct_padding(
    sequence_generator_instance, sample_batch_data
):
    """
    Tests that padding is applied correctly to the initial sequences.
    """
    generator = sequence_generator_instance
    x_data, y_data = sample_batch_data

    sequences, _, _ = generator.generate_offline(x_data, y_data)

    expected_first_sequence = np.zeros((5, 4))
    expected_first_sequence[4, :] = (
        x_data[["dur", "sbytes", "dbytes", "rate"]].iloc[0].values
    )
    np.testing.assert_array_equal(sequences[0], expected_first_sequence)

    expected_last_sequence = (
        x_data[["dur", "sbytes", "dbytes", "rate"]].iloc[5:10].values
    )
    np.testing.assert_array_equal(sequences[9], expected_last_sequence)


def test_generate_offline_with_insufficient_data(sequence_generator_instance):
    """
    Tests that generation returns empty arrays if data is shorter than sequence length.
    """
    generator = sequence_generator_instance
    x_short = pd.DataFrame(np.zeros((4, 3)), columns=["dur", "sbytes", "dbytes"])
    y_short = pd.Series([0, 1, 0, 1])
    sequences, labels, valid_indices = generator.generate_offline(x_short, y_short)

    assert sequences.shape == (4, 5, 3)
    assert labels.shape == (4,)
    assert len(valid_indices) == 4


@pytest.mark.asyncio
async def test_generate_online_sequence_correct_shape_and_padding(
    sequence_generator_instance,
):
    """
    Tests the online generation for a single instance, checking shape and padding.
    """
    # Arrange
    generator = sequence_generator_instance
    x_single = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]], columns=["dur", "sbytes", "dbytes", "rate"]
    )
    session_id = "test_session_1"

    mock_redis_client = MagicMock()
    mock_pipeline = AsyncMock()

    mock_pipeline.lpush = MagicMock()
    mock_pipeline.ltrim = MagicMock()
    mock_pipeline.lrange = MagicMock()

    new_record = x_single.iloc[0].to_dict()
    redis_return_value = [json.dumps(new_record).encode("utf-8")]
    mock_pipeline.execute.return_value = [None, None, redis_return_value]
    mock_redis_client.pipeline.return_value.__aenter__.return_value = mock_pipeline

    sequences = await generator.generate_online_sequence(
        x_single, mock_redis_client, session_id
    )

    assert sequences.shape == (1, 5, 4)
    expected_sequence = np.zeros((5, 4))
    expected_sequence[4, :] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(sequences[0], expected_sequence)


def test_generate_handles_missing_temporal_features(sequence_generator_instance):
    """
    Tests that the generator only uses available temporal features and doesn't fail.
    """
    generator = sequence_generator_instance
    x_data = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["dur", "sload"])
    y_data = pd.Series(np.zeros(10))

    sequences, _, _ = generator.generate_offline(x_data, y_data)

    assert sequences.shape == (10, 5, 2)
