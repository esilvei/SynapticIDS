import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(trainer, x_test, y_test):
    """
    Evaluates a trained model using PRE-PREPARED test data.

    This function's core responsibility is to calculate and structure the
    evaluation metrics using data that has already been processed.

    Args:
        trainer (ModelTrainer): The trained ModelTrainer instance.
        x_test (list): The prepared test data inputs (e.g., [images, sequences]).
        y_test (np.array): The prepared test labels.

    Returns:
        A dictionary containing all evaluation results.
    """
    print("Evaluating model on prepared test data...")
    # The data is already prepared, so we directly predict.
    y_true_raw = y_test
    y_pred_proba = trainer.model.predict(x_test)

    # The rest of the function remains the same logic for calculating metrics.
    if trainer.mode == "multiclass":
        y_true = (
            np.argmax(y_true_raw, axis=1)
            if y_true_raw.ndim > 1 and y_true_raw.shape[1] > 1
            else y_true_raw
        )
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:  # Binary mode
        y_true = y_true_raw.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Use zero_division=0 to prevent warnings for labels with no predictions.
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def display_results(results, mode="multiclass", save_path=None):
    """
    Prints key metrics and plots the confusion matrix from an evaluation results dictionary.
    """
    print("\n=== Key Metrics ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_score']:.4f}")

    plt.figure(figsize=(10, 8))
    cm = results["confusion_matrix"]

    if mode == "multiclass":
        # Safely get class names from the classification report
        class_names = [
            key
            for key in results["classification_report"].keys()
            if key not in ["accuracy", "macro avg", "weighted avg"]
        ]
    else:
        class_names = ["0", "1"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plots training & validation metrics from a Keras History object.
    """
    if not history or not hasattr(history, "history"):
        print("Training history is empty or invalid.")
        return

    # Filter out learning rate from the main plots
    metrics_to_plot = [
        key
        for key in history.history.keys()
        if not key.startswith("val_") and key != "lr"
    ]
    num_metrics = len(metrics_to_plot)

    plt.figure(figsize=(6 * num_metrics, 5))

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, num_metrics, i + 1)
        plt.plot(history.history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history.history:
            plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()
    plt.close()
