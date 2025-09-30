import os
import sys
from datetime import datetime
import json
import mlflow

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
from src.synaptic_ids.training.model.pipeline import SynapticIDSPipeline
from src.synaptic_ids.training.analysis.analysis import (
    evaluate_model,
    display_results,
    plot_training_history,
)
from src.synaptic_ids.training.observers.mlflow_observer import MLflowObserver
from src.synaptic_ids.training.observers.setup_mlflow import setup_mlflow_local

# Suppress TensorFlow informational messages for a cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_and_load_data():
    """Sets up dataset paths and loads data into dataframes."""
    print("STEP 1: Setting up and loading data...")
    data_setup = DataSetup(
        dataset_name=settings.paths.dataset_name,
        download_path=settings.paths.raw_data,
    )
    local_dataset_path = data_setup.setup_dataset()

    data_loader = DataLoader(
        dataset_dir=local_dataset_path,
        target_col=settings.training.target_column,
        test_size=settings.training.test_size,
        val_size=settings.training.val_size,
    )
    return data_loader.load_and_split()


def prepare_data(train_df, val_df, test_df):
    """Prepares and transforms data for the model."""
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
    val_data = data_preparer.prepare_data(val_df, is_training=True)
    test_data = data_preparer.prepare_data(test_df, is_training=True)
    return train_data, val_data, test_data, data_preparer


def build_and_train_model(train_data, val_data, data_preparer):
    """Builds, compiles, and trains the SynapticIDS model."""
    print("\nSTEP 3 & 4: Building and training the model...")
    if len(train_data["images"]) == 0:
        print("No training data generated. Exiting.")
        return None, None

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

    trainer = ModelTrainer(
        model=synaptic_model,
        mode=settings.training.mode,
        preprocessor=data_preparer,
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
    return trainer, history


def evaluate_and_log_results(trainer, history, test_data, preparer, test_df,observer):
    """Evaluates the model and logs artifacts and metrics."""
    print("\nSTEP 5 & 6: Evaluating, analyzing, and logging results...")
    results_dir = os.path.join(settings.paths.processed_data, "results")
    os.makedirs(results_dir, exist_ok=True)
    x_test = [test_data["images"], test_data["sequences"]]
    y_test = test_data["labels"]

    results = evaluate_model(trainer, x_test=x_test, y_test=y_test)
    observer.on_metrics_logged(
        {"accuracy": results["accuracy"], "f1_score": results["f1_score"]}
    )

    # Log artifacts
    report_path = os.path.join(results_dir, "classification_report.json")
    history_plot_path = os.path.join(results_dir, "training_history.png")
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results["classification_report"], f, indent=4)

    plot_training_history(history, save_path=history_plot_path)
    display_results(results, mode=settings.training.mode, save_path=cm_plot_path)

    observer.on_artifact_logged(report_path, "reports")
    observer.on_artifact_logged(cm_plot_path, "plots")
    observer.on_artifact_logged(history_plot_path, "plots")

    # Log model
    pipeline_model = SynapticIDSPipeline(
        model = trainer.model,
        data_preparer=preparer
    )
    # Prepare a clean input example for MLflow signature inference
    input_example_df = test_df.head(1).copy()
    input_example_df = input_example_df.drop(columns=['attack_cat', 'label'], errors='ignore')
    # Ensure categorical columns are strings (avoid Pandas 'category' dtype issues)
    for col in input_example_df.select_dtypes(include=["category"]).columns:
        input_example_df[col] = input_example_df[col].astype(str)
    # Cast integer columns to float64 to avoid MLflow integer-missing-value caveat
    int_cols = input_example_df.select_dtypes(include=["int", "int32", "int64"]).columns
    for col in int_cols:
        input_example_df[col] = input_example_df[col].astype("float64")

    # Provide explicit pip requirements to avoid MLflow auto inference issues
    requirements_path = settings.paths.requirements

    observer.on_model_logged(
        model=pipeline_model,
        name="synaptic_ids_pipeline",
        registered_model_name="SynapticIDSPipeline",
        input_example=input_example_df,
        pip_requirements_path=requirements_path
    )

def main():
    """The main orchestrator function for the model training and evaluation pipeline."""
    print("--- SYNAPTIC-IDS: STARTING END-TO-END TRAINING PIPELINE ---")
    mlflow.set_experiment("SynapticIDS")
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as _:
        observer = MLflowObserver()
        observer.on_parameters_logged(
            {**vars(settings.training), **vars(settings.features)}
        )

        train_df, val_df, test_df = setup_and_load_data()
        train_data, val_data, test_data, preparer = prepare_data(
            train_df, val_df, test_df
        )
        trainer, history = build_and_train_model(train_data, val_data, preparer)

        if trainer and history:
            evaluate_and_log_results(trainer, history, test_data, preparer, test_df, observer)

        print("\n--- SYNAPTIC-IDS: TRAINING PIPELINE COMPLETE ---")


if __name__ == "__main__":
    setup_mlflow_local()
    main()
