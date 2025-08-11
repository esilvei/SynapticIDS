from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.data_setup import DataSetup


class DataLoader:
    """
    Responsible for loading the prepared raw dataset and splitting it into
    training, validation, and test sets for the pipeline.
    """

    def __init__(
        self, dataset_dir: Path, test_size: float = 0.2, val_size: float = 0.1
    ):
        """
        Initializes the DataLoader.

        Args:
            dataset_dir (Path): Path to the directory with the dataset files.
            test_size (float): Proportion of the dataset for the test split.
            val_size (float): Proportion of the training data for the validation split.
        """
        self.dataset_dir = dataset_dir
        self.test_size = test_size
        self.val_size = val_size

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads data from parquet files, concatenates them, and performs a
        stratified split based on the 'attack_cat' column.

        Returns:
            A tuple of (train_df, val_df, test_df).
        """
        print("Loading datasets for splitting...")
        train_path = self.dataset_dir / "UNSW_NB15_training-set.parquet"
        test_path = self.dataset_dir / "UNSW_NB15_testing-set.parquet"

        df_train_raw = pd.read_parquet(train_path)
        df_test_raw = pd.read_parquet(test_path)
        full_df = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

        stratify_col = full_df["attack_cat"]

        print("Splitting data into train, validation, and test sets...")
        train_val_df, test_df = train_test_split(
            full_df, test_size=self.test_size, stratify=stratify_col, random_state=42
        )

        relative_val_size = self.val_size / (1.0 - self.test_size)
        stratify_col_train_val = train_val_df["attack_cat"]

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=stratify_col_train_val,
            random_state=42,
        )

        print("Data split complete.")
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Step 1: Setup and download the data
    data_setup = DataSetup()
    local_dataset_path = data_setup.setup_dataset()

    # Step 2: Load and split the downloaded data
    data_loader = DataLoader(dataset_dir=local_dataset_path)
    train, val, test = data_loader.load_and_split()

    print("\n--- Sample Data ---")
    print("Training data head:")
    print(train.head())
