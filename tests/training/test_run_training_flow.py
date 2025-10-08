from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.synaptic_ids.processing.data_loader import DataLoader
from src.synaptic_ids.processing.data_setup import DataSetup
from src.synaptic_ids.training.run_training import build_and_train_model, prepare_data


@pytest.fixture
def small_real_dataframe():
    """Creates a small DataFrame (30 rows) with realistic data."""
    data = {
        "dur": np.random.rand(30),
        "spkts": np.random.randint(1, 100, 30),
        "sbytes": np.random.randint(100, 1000, 30),
        "rate": np.random.rand(30) * 100,
        "proto": ["tcp", "udp", "arp"] * 10,
        "service": ["dns", "-", "http", "smtp", "ftp", "ssh"] * 5,
        "state": ["FIN", "CON", "INT"] * 10,
        "attack_cat": [
            "Normal",
            "Exploits",
            "DoS",
            "Reconnaissance",
            "Generic",
            "Fuzzers",
        ]
        * 5,
        "label": [0, 1, 1, 1, 1, 1] * 5,
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_data_pipeline_integration(small_real_dataframe, monkeypatch):
    """Integration test verifying DataLoader -> FeatureEngineer -> DataPreparer."""
    # Arrange
    monkeypatch.setattr(DataSetup, "setup_dataset", lambda self: "fake_path")
    monkeypatch.setattr(
        DataLoader,
        "load_and_split",
        lambda self: (
            small_real_dataframe,
            small_real_dataframe.head(10),
            small_real_dataframe.head(10),
        ),
    )

    # --- Act ---
    loader = DataLoader(dataset_dir="", target_col="label", test_size=0.2, val_size=0.1)
    train_df, val_df, test_df = loader.load_and_split()
    train_data, _, _, _ = await prepare_data(train_df, val_df, test_df)

    # --- Assert ---
    assert isinstance(train_data, dict)
    assert all(k in train_data for k in ["images", "sequences", "labels"])
    assert train_data["images"].shape[0] > 0
    assert train_data["images"].shape[0] == train_data["sequences"].shape[0]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_smoke_test(small_real_dataframe, monkeypatch):
    """A full smoke test that runs the pipeline end-to-end for 1 epoch."""
    # --- Arrange ---
    monkeypatch.setattr(DataSetup, "setup_dataset", lambda self: "fake_path")
    monkeypatch.setattr(
        DataLoader,
        "load_and_split",
        lambda self: (
            small_real_dataframe,
            small_real_dataframe.head(10),
            small_real_dataframe.head(10),
        ),
    )
    monkeypatch.setattr(
        "src.synaptic_ids.training.run_training.evaluate_and_log_results", MagicMock()
    )

    # --- Act ---
    # 1. Prepare data
    loader = DataLoader(dataset_dir="", target_col="label", test_size=0.2, val_size=0.1)
    train_df, val_df, test_df = loader.load_and_split()
    train_data, val_data, _, preparer = await prepare_data(train_df, val_df, test_df)

    # 2. Build and train for 1 epoch
    monkeypatch.setattr("src.synaptic_ids.config.settings.training.epochs", 1)
    trainer, history = build_and_train_model(train_data, val_data, preparer)

    # --- Assert ---
    # 3. Final checks
    assert trainer is not None
    assert history is not None
    assert "accuracy" in history.history
