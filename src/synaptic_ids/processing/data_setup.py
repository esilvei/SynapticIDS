import subprocess
from pathlib import Path


class DataSetup:
    """
    Handles the initial download and setup of the dataset from Kaggle.
    Its single responsibility is to ensure the raw data is available locally.
    """

    def __init__(
        self, dataset_name: str = "dhoogla/unswnb15", download_path: str = "data/raw"
    ):
        self.dataset_name = dataset_name

        # Correctly determine the project root directory to save data
        # __file__ is the path to the current script (e.g., .../scripts/data_loader.py)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        self.download_path = project_root / download_path
        self.dataset_dir = self.download_path

    def setup_dataset(self) -> Path:
        """
        Downloads and unzips the dataset using the Kaggle CLI if it doesn't already exist.
        This method is more robust than using the kagglehub library directly for downloads.

        Returns:
            The path to the directory containing the dataset files.
        """
        print("Setting up dataset...")
        self.download_path.mkdir(parents=True, exist_ok=True)

        # Check if files already exist to avoid re-downloading
        train_file = self.download_path / "UNSW_NB15_training-set.parquet"
        if train_file.exists():
            print("Dataset files already exist locally.")
            return self.dataset_dir

        print(f"Downloading dataset '{self.dataset_name}' using Kaggle CLI...")

        # Command to download the dataset zip file
        download_command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.dataset_name,
            "-p",
            str(self.download_path),
            "--unzip",
        ]

        try:
            # Execute the command
            subprocess.run(download_command, check=True, capture_output=True, text=True)
            print("Dataset downloaded and unzipped successfully.")
        except FileNotFoundError:
            print("\n--- KAGGLE CLI ERROR ---")
            print("The 'kaggle' command was not found.")
            raise
        except subprocess.CalledProcessError as e:
            print("\n--- KAGGLE CLI ERROR ---")
            print(
                "The Kaggle CLI command failed"
                ". This might be due to an incorrect dataset name or authentication issues."
            )
            print(f"Dataset: {self.dataset_name}")
            print(f"Error: {e.stderr}")
            raise

        return self.dataset_dir
