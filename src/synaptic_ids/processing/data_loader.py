# In scripts/data_loader.py

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Responsible for loading the prepared raw dataset and splitting it into
    training, validation, and test sets for the pipeline.
    """

    def __init__(
        self, dataset_dir: Path, target_col: str, test_size: float, val_size: float
    ):
        """
        Initializes the DataLoader.

        Args:
            dataset_dir (Path): Path to the directory with the dataset files.
            target_col (str): The name of the column to use for stratified splitting.
            test_size (float): Proportion of the dataset for the test split.
            val_size (float): Proportion of the training data for the validation split.
        """
        if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
            raise ValueError(
                "test_size and val_size must be floats "
                "between 0 and 1, and their sum must be less than 1."
            )

        self.dataset_dir = dataset_dir
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads data from parquet files, concatenates them, and performs a
        stratified split based on the target column.
        """
        print("Loading datasets for splitting...")
        # Assuming the parquet files have a consistent naming scheme
        all_parquet_files = list(self.dataset_dir.glob("*.parquet"))
        if not all_parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {self.dataset_dir}")

        full_df = pd.concat(
            (pd.read_parquet(f) for f in all_parquet_files), ignore_index=True
        )

        print("Splitting data into train, validation, and test sets...")

        # Use self.target_col for stratification
        stratify_col = full_df[self.target_col]

        # First split: separate the test set
        train_val_df, test_df = train_test_split(
            full_df, test_size=self.test_size, stratify=stratify_col, random_state=42
        )

        # Second split: separate the validation set from the remaining data
        relative_val_size = self.val_size / (1.0 - self.test_size)
        stratify_col_train_val = train_val_df[self.target_col]

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=stratify_col_train_val,
            random_state=42,
        )

        print(
            f"Data split complete. Train: {len(train_df)},"
            f" Val: {len(val_df)}, Test: {len(test_df)} rows."
        )
        return train_df, val_df, test_df
