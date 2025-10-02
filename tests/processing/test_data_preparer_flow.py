import pytest
import pandas as pd
import numpy as np
from cloudpickle import cloudpickle

from src.synaptic_ids.config import settings

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
    # We'll use a subset of features to make the test faster.
    selected_features = ["dur", "spkts", "sbytes", "rate", "proto", "service", "state"]
    feature_engineer = UNSWNB15FeatureEngineer(
        mode="multiclass", target_col="attack_cat", selected_features=selected_features
    )

    data_preparer = DataPreparer(feature_engineer=feature_engineer, mode="multiclass")

    data_preparer.fit(small_real_dataframe)
    prepared_data = data_preparer.prepare_data(small_real_dataframe, is_training=True)

    assert "images" in prepared_data
    assert "sequences" in prepared_data
    assert "labels" in prepared_data

    # Expected: 25 - 5 + 1 = 21 sequences
    expected_num_sequences = 21

    # Check the shapes
    assert prepared_data["sequences"].shape[0] == expected_num_sequences
    assert prepared_data["images"].shape[0] == expected_num_sequences
    assert prepared_data["labels"].shape[0] == expected_num_sequences

    assert prepared_data["sequences"].shape[1] == 5  # sequence_length
    assert prepared_data["sequences"].shape[2] == 5  # num_temporal_features

    assert prepared_data["images"].shape[1:] == (32, 32, 1)

    assert prepared_data["labels"].shape[1] == 5

    assert prepared_data["sequences"].dtype == "float32"
    assert prepared_data["images"].dtype == "float32"
    assert prepared_data["labels"].dtype in ("float32", "float64")


def test_data_preparer_inference_flow(small_real_dataframe):
    """
    Tests the end-to-end flow for INFERENCE.
    1. Fits the preparer with training data.
    2. Simulates saving and loading the fitted object.
    3. Transforms new data (without labels) using the loaded preparer.
    4. Verifies the output is correct for the model and contains no labels.
    """
    feature_engineer = UNSWNB15FeatureEngineer(
        mode=settings.training.mode,
        target_col=settings.training.target_column,
        selected_features=settings.features.selected,
    )
    preparer_for_training = DataPreparer(
        feature_engineer=feature_engineer, mode=settings.training.mode
    )
    preparer_for_training.fit(small_real_dataframe)

    saved_preparer = cloudpickle.dumps(preparer_for_training)
    loaded_preparer_for_inference = cloudpickle.loads(saved_preparer)

    inference_df = small_real_dataframe.head(10).drop(
        columns=[settings.training.target_column]
    )

    prepared_data = loaded_preparer_for_inference.prepare_data(
        inference_df, is_training=False
    )

    assert "images" in prepared_data
    assert "sequences" in prepared_data
    assert prepared_data["labels"] is None

    assert prepared_data["images"].shape[0] == prepared_data["sequences"].shape[0]
    assert prepared_data["images"].shape[0] > 0

    assert prepared_data["images"].shape[1] == 32
