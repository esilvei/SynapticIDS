import os
import sys
from pathlib import Path

# Ensure the root directory is in the Python path to find 'config' and 'src'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

# --- 1. CENTRALIZED CONFIGURATION IMPORT ---
from src.synaptic_ids.config import settings

# --- 2. COMPONENT IMPORTS ---
from src.synaptic_ids.processing.data_setup import DataSetup
from src.synaptic_ids.processing.data_loader import DataLoader
from src.synaptic_ids.processing.feature_engineer import UNSWNB15FeatureEngineer
from src.synaptic_ids.processing.data_transformer.data_preparer import DataPreparer
from src.synaptic_ids.training.model.model_builder import SynapticIDSBuilder
from src.synaptic_ids.training.model.model_trainer import ModelTrainer
from src.synaptic_ids.training.analysis.analysis import (
    evaluate_model,
    display_results,
    plot_training_history,
)

# Suppress TensorFlow informational messages for a cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    """
    Main orchestrator function for the entire model training and evaluation pipeline.
    This function reads all parameters from the central 'settings' object.
    """
    print("--- SYNAPTIC-IDS: STARTING END-TO-END TRAINING PIPELINE ---")

    # --- Step 1: Data Setup and Loading ---
    print("STEP 1: Setting up and loading data...")
    data_setup = DataSetup(
        dataset_name=settings.paths.dataset_name, download_path=settings.paths.raw_data
    )
    local_dataset_path = data_setup.setup_dataset()

    data_loader = DataLoader(
        dataset_dir=local_dataset_path,
        target_col=settings.training.target_column,
        test_size=settings.training.test_size,
        val_size=settings.training.val_size,
    )
    train_df, val_df, test_df = data_loader.load_and_split()

    # --- Step 2: Data Preparation ---
    print("\nSTEP 2: Preparing data for the model...")
    feature_engineer = UNSWNB15FeatureEngineer(
        mode=settings.training.mode,
        target_col=settings.training.target_column,
        selected_features=settings.features.selected,
    )
    data_preparer = DataPreparer(
        feature_engineer=feature_engineer, mode=settings.training.mode
    )
    data_preparer.fit(train_df)
    train_data = data_preparer.prepare_data(train_df, is_training=True)
    val_data = data_preparer.prepare_data(val_df)
    test_data = data_preparer.prepare_data(test_df)

    # --- Step 3: Model Architecture Construction ---
    print("\nSTEP 3: Building the model architecture...")
    if len(train_data["images"]) == 0:
        print("No training data generated. Exiting.")
        return

    image_shape = train_data["images"].shape[1:]
    sequence_shape = train_data["sequences"].shape[1:]
    num_classes = (
        train_data["labels"].shape[1] if settings.training.mode == "multiclass" else 1
    )

    builder = SynapticIDSBuilder(
        image_shape=image_shape,
        sequence_shape=sequence_shape,
        num_classes=num_classes,
        mode=settings.training.mode,
    )
    synaptic_model = builder.build()

    # --- Step 4: Model Training ---
    print("\nSTEP 4: Compiling and training the model...")
    trainer = ModelTrainer(
        model=synaptic_model, mode=settings.training.mode, preprocessor=data_preparer
    )
    steps_per_epoch = len(train_data["images"]) // settings.training.batch_size
    trainer.compile_model(
        steps_per_epoch=steps_per_epoch, epochs=settings.training.epochs
    )
    history = trainer.train(
        train_data,
        val_data,
        epochs=settings.training.epochs,
        batch_size=settings.training.batch_size,
        use_class_weights=settings.training.use_class_weights,
    )

    # --- Step 5: Evaluation and Analysis ---
    print("\nSTEP 5: Evaluating the model and analyzing results...")
    # --- FIX: Unpack the test_data dictionary into separate arguments ---
    # The evaluate_model function expects features (x_test) and labels (y_test) separately.
    x_test = [test_data["images"], test_data["sequences"]]
    y_test = test_data["labels"]

    results = evaluate_model(trainer, x_test=x_test, y_test=y_test)
    display_results(results, mode=settings.training.mode)
    plot_training_history(history)

    # --- Step 6: Save Final Model Artifact ---
    model_save_path = Path(settings.paths.model_save)
    print(f"\nSTEP 6: Saving final model to {model_save_path}...")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.model.save(model_save_path)
    print("Model artifact saved.")

    print("\n--- SYNAPTIC-IDS: TRAINING PIPELINE COMPLETE ---")


if __name__ == "__main__":
    main()
