# pylint: disable=too-many-ancestors
"""
This module defines a custom Keras layer for transforming tabular data into an
image representation, specifically for the UNSW-NB15 dataset.
"""

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class UNSWNB15ToImage(keras.layers.Layer):
    """
    A TensorFlow layer that transforms tabular data into an image representation.
    This version is graph-compatible, linter-friendly, and type-safe.
    """

    def __init__(
        self,
        feature_names: List[str],
        image_size: Tuple[int, int] = (32, 32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.image_size = image_size

        # Defer weight creation to the build method for robustness
        self.feature_min: Optional[tf.Variable] = None
        self.feature_max: Optional[tf.Variable] = None
        self.feature_ordering: Optional[tf.Variable] = None
        self.is_initialized: Optional[tf.Variable] = None

    def build(self, input_shape):
        """Creates the layer's state. Called automatically on first use."""
        self.feature_min = self.add_weight(
            name="feature_min",
            shape=(self.n_features,),
            initializer="zeros",
            trainable=False,
        )
        self.feature_max = self.add_weight(
            name="feature_max",
            shape=(self.n_features,),
            initializer="ones",
            trainable=False,
        )
        self.feature_ordering = self.add_weight(
            name="feature_ordering",
            shape=(self.n_features,),
            initializer=keras.initializers.Constant(
                np.arange(self.n_features, dtype=np.int32)
            ),
            dtype=tf.int32,
            trainable=False,
        )
        self.is_initialized = self.add_weight(
            name="is_initialized",
            shape=(),
            initializer="zeros",
            dtype=tf.bool,
            trainable=False,
        )
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the layer."""
        config = super().get_config()
        config.update(
            {
                "feature_names": self.feature_names,
                "image_size": self.image_size,
            }
        )
        return config

    @tf.function
    def _initialize_and_update(self, features: tf.Tensor):
        """Initializes the layer on the first training call and sets initial stats."""
        # Feature ordering based on correlation
        features_centered = features - tf.reduce_mean(features, axis=0)
        features_std = tf.math.reduce_std(features_centered, axis=0)
        features_normalized = features_centered / (features_std + 1e-7)
        correlation = tf.matmul(
            features_normalized, features_normalized, transpose_a=True
        ) / tf.cast(tf.shape(features)[0], tf.float32)
        correlation_sum = tf.reduce_sum(tf.abs(correlation), axis=1)
        ordering = tf.argsort(correlation_sum, direction="DESCENDING")

        self.feature_ordering.assign(ordering)
        self.feature_min.assign(tf.reduce_min(features, axis=0))
        self.feature_max.assign(tf.reduce_max(features, axis=0))
        self.is_initialized.assign(True)

    @tf.function
    def _update_statistics(self, features: tf.Tensor):
        """Updates the min/max statistics on subsequent training calls."""
        self.feature_min.assign(
            tf.minimum(self.feature_min, tf.reduce_min(features, axis=0))
        )
        self.feature_max.assign(
            tf.maximum(self.feature_max, tf.reduce_max(features, axis=0))
        )

    @tf.function
    def _normalize_features(self, features: tf.Tensor, training: tf.bool) -> tf.Tensor:
        """Normalizes features to a [0, 1] scale in a graph-compatible way."""

        def update_stats_fn():
            tf.cond(
                self.is_initialized,
                lambda: self._update_statistics(features),
                lambda: self._initialize_and_update(features),
            )
            return tf.constant(True)  # Return a value for tf.cond

        # Only update statistics during training
        tf.cond(training, update_stats_fn, lambda: tf.constant(False))

        denominator = self.feature_max - self.feature_min
        denominator = tf.where(tf.equal(denominator, 0), 1.0, denominator)
        normalized = (features - self.feature_min) / denominator
        return tf.clip_by_value(normalized, 0.0, 1.0)

    @tf.function
    def _organize_features(self, features: tf.Tensor) -> tf.Tensor:
        """Reorders features based on learned correlation."""
        return tf.gather(features, self.feature_ordering, axis=1)

    @tf.function
    def _features_to_image(self, features: tf.Tensor) -> tf.Tensor:
        """Converts the flattened feature vector into a 2D image."""
        batch_size = tf.shape(features)[0]
        n_features_float = tf.cast(self.n_features, tf.float32)
        side_length = tf.cast(tf.math.ceil(tf.math.sqrt(n_features_float)), tf.int32)

        padding_size = side_length * side_length - self.n_features
        padded = tf.pad(features, [[0, 0], [0, padding_size]])
        images = tf.reshape(padded, [batch_size, side_length, side_length, 1])
        return tf.image.resize(images, self.image_size, method="bilinear")

    # pylint: disable=arguments-differ
    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Defines the forward pass logic for the layer."""
        if training is None:
            # Default to Keras' learning phase global flag
            training = keras.backend.learning_phase()

        features = tf.cast(inputs, tf.float32)
        # Ensure the boolean is a tensor for tf.cond
        is_training_tensor = tf.cast(training, tf.bool)

        normalized = self._normalize_features(features, is_training_tensor)
        organized = self._organize_features(normalized)
        images = self._features_to_image(organized)
        return images
