import pytest
import pandas as pd
import numpy as np

# Import the REAL components to be integrated
from src.synaptic_ids.processing.feature_engineer import UNSWNB15FeatureEngineer
from src.synaptic_ids.processing.data_transformer.data_preparer import DataPreparer


@pytest.fixture
def small_real_dataframe():
    """
    Creates a small, realistic DataFrame to simulate the data flow.
    Includes categorical features, numerical features, and the target.
    """
    data = {
        # Numerical features
        "dur": np.random.rand(25),
        "spkts": np.random.randint(1, 100, 25),
        "sbytes": np.random.randint(100, 1000, 25),
        "rate": np.random.rand(25) * 100,
        # Categorical features
        "proto": ["tcp", "udp"] * 12 + ["tcp"],
        "service": ["dns", "-", "http", "smtp", "ftp"] * 5,
        "state": ["FIN", "CON", "INT"] * 8 + ["FIN"],
        # Target
        "attack_cat": ["Normal", "Exploits", "DoS", "Reconnaissance", "Generic"] * 5,
        "label": [0, 1, 1, 1, 1] * 5,
    }
    return pd.DataFrame(data)


def test_data_preparer_integration_flow(small_real_dataframe):  # pylint: disable=redefined-outer-name
    """
    Tests the end-to-end flow of the DataPreparer with real components.
    1. Instantiates the real FeatureEngineer.
    2. Instantiates the DataPreparer with the real engineer.
    3. Executes fit() and prepare_data().
    4. Verifies the output for correct shape, type, and consistency.
    """
    # --- Arrange ---
    # 1. Initialize the real feature engineer
    # We'll use a subset of features to make the test faster.
    selected_features = ["dur", "spkts", "sbytes", "rate", "proto", "service", "state"]
    feature_engineer = UNSWNB15FeatureEngineer(
        mode="multiclass", target_col="attack_cat", selected_features=selected_features
    )

    # 2. Initialize the DataPreparer with the real component
    data_preparer = DataPreparer(feature_engineer=feature_engineer, mode="multiclass")

    # --- Act ---
    # 3. Execute the full fit and prepare flow
    data_preparer.fit(small_real_dataframe)
    prepared_data = data_preparer.prepare_data(small_real_dataframe, is_training=True)

    # --- Assert ---
    # 4. Check the integrity of the output

    # Check that the output dictionary was created correctly
    assert "images" in prepared_data
    assert "sequences" in prepared_data
    assert "labels" in prepared_data

    # Calculate the expected number of sequences
    # 25 rows of data, sequence length is 5 (SequenceGenerator default)
    # Expected: 25 - 5 + 1 = 21 sequences
    expected_num_sequences = 21

    # Check the shapes
    assert prepared_data["sequences"].shape[0] == expected_num_sequences
    assert prepared_data["images"].shape[0] == expected_num_sequences
    assert prepared_data["labels"].shape[0] == expected_num_sequences

    # Check the sequence feature dimension
    # The number of temporal features in our test dataframe is 4.
    assert prepared_data["sequences"].shape[1] == 5  # sequence_length
    assert prepared_data["sequences"].shape[2] == 5  # num_temporal_features

    # Check the image dimension (default is 32x32x1)
    assert prepared_data["images"].shape[1:] == (32, 32, 1)

    # Check the labels dimension (one-hot encoded for 5 classes)
    assert prepared_data["labels"].shape[1] == 5

    # Check the data types (dtypes)
    assert prepared_data["sequences"].dtype == "float32"
    assert prepared_data["images"].dtype == "float32"
    assert prepared_data["labels"].dtype == "float32"
