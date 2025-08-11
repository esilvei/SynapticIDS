# pylint: disable=no-member
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from scripts.lr_scheduler import OneCycleLR


class ModelTrainer:
    """
    Manages the lifecycle of a Keras model, including compilation,
    training, and prediction. It depends on a pre-built model and a
    preprocessor, separating the training workflow from the model's
    architecture and data preparation logic.
    """

    def __init__(self, model, mode="multiclass", preprocessor=None):
        """
        Initializes the ModelTrainer.

        Args:
            model (tf.keras.Model): The pre-built, uncompiled Keras model object.
            mode (str): The operational mode, either 'multiclass' or 'binary'.
            preprocessor: An object with a `prepare_data` method. This is a dependency.
        """
        if not hasattr(model, "fit"):
            raise TypeError("The 'model' object must be a compilable Keras model.")
        self.model = model
        self.mode = mode
        self.preprocessor = preprocessor

    def compile_model(self, steps_per_epoch, epochs):
        """
        Compiles the model with the specified optimizer and loss function.
        This logic is now separated from the model's architecture definition.
        """
        lr_schedule = OneCycleLR(
            max_lr=1e-4,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            div_factor=10.0,
            final_div_factor=1e3,
            three_phase=False,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.95,
            global_clipnorm=1.0,
        )

        loss = (
            tf.keras.losses.CategoricalFocalCrossentropy(
                name="loss", label_smoothing=0.2
            )
            if self.mode == "multiclass"
            else tf.keras.losses.BinaryFocalCrossentropy(
                name="loss", label_smoothing=0.2
            )
        )

        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Model compiled successfully.")

    # pylint: disable=too-many-arguments
    def train(
        self, x_train, y_train, x_val, y_val, epochs, batch_size, use_class_weights=True
    ):
        """
        Executes the model training process by first preparing data via the preprocessor.

        Args:
            x_train, y_train: Raw training data and labels.
            x_val, y_val: Raw validation data and labels.
            epochs (int): The number of epochs to train for.
            batch_size (int): The size of each training batch.
            use_class_weights (bool): Whether to apply class balancing.

        Returns:
            A Keras History object containing the training history.
        """
        # The trainer delegates data preparation to the preprocessor
        train_data = self.preprocessor.prepare_data(x_train, y_train)
        val_data = self.preprocessor.prepare_data(x_val, y_val)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=10, mode="max", restore_best_weights=True
            )
        ]
        class_weights = (
            self._compute_class_weights(train_data["labels"])
            if use_class_weights
            else None
        )

        history = self.model.fit(
            x=[train_data["images"], train_data["sequences"]],
            y=train_data["labels"],
            batch_size=batch_size,
            validation_data=(
                [val_data["images"], val_data["sequences"]],
                val_data["labels"],
            ),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        return history

    def predict(self, x_data):
        """
        Generates final class predictions for new, raw input data.
        """
        if self.preprocessor is None:
            raise ValueError(
                "A preprocessor must be provided to make predictions from raw data."
            )

        pred_data = self.preprocessor.prepare_data(x_data, y=None)
        predictions = self.model.predict([pred_data["images"], pred_data["sequences"]])

        if self.mode == "multiclass":
            return np.argmax(predictions, axis=1)

        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, x_data):
        """
        Generates class probabilities for new, raw input data.
        """
        if self.preprocessor is None:
            raise ValueError(
                "A preprocessor must be provided to make predictions from raw data."
            )

        pred_data = self.preprocessor.prepare_data(x_data, y=None)
        predictions = self.model.predict([pred_data["images"], pred_data["sequences"]])

        if self.mode == "binary":
            return np.column_stack([1 - predictions, predictions])
        return predictions

    def _compute_class_weights(self, y):
        """
        Calculates class weights for imbalanced datasets. This is a helper
        function specific to the training process.
        """
        y_integers = (
            np.argmax(y, axis=1)
            if y.ndim > 1 and self.mode == "multiclass"
            else y.flatten()
        )
        classes = np.unique(y_integers)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_integers)
        return dict(enumerate(class_weights))
