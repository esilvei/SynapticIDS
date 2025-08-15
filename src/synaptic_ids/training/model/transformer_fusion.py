# pylint: disable=no-member
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving


# pylint: disable=attribute-defined-outside-init
@saving.register_keras_serializable()
class TransformerFusion(keras.layers.Layer):
    """
    A custom layer to fuse multimodal features using a cross-attention mechanism.

    This layer takes feature tensors from different sources (e.g., a CNN branch
    and an LSTM branch) and projects them into a common dimension. It then uses
    Multi-Head Attention to allow the branches to enrich each other's representations
    before combining them into a final, fused output. This implementation follows
    modern Keras best practices by defining sub-layers within the build() method.
    """

    def __init__(self, dim=128, num_heads=8, dropout_rate=0.4, **kwargs):
        """
        Initializes the layer with its configuration.

        Args:
            dim (int): The dimensionality of the projection and attention space.
            num_heads (int): The number of attention heads.
            dropout_rate (float): The dropout rate for regularization.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """
        Creates the weights and sub-layers of this layer.

        This method is called automatically by Keras the first time the layer is
        used, based on the shape of the input tensors.
        """
        # --- All sub-layers are defined here ---
        self.cnn_proj = keras.layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.lstm_proj1 = keras.layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.lstm_proj2 = keras.layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )

        self.cnn_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_initializer="glorot_uniform",
        )
        self.lstm_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_initializer="glorot_uniform",
        )

        self.norm_cnn = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_lstm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_cross1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_cross2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.cnn_seq_proj = keras.layers.Dense(
            self.dim * 2, kernel_initializer="glorot_uniform"
        )

        self.pre_gate = keras.layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.gate = keras.layers.Dense(
            self.dim, activation="sigmoid", kernel_initializer="glorot_uniform"
        )

        self.combine = keras.layers.Dense(
            self.dim,
            activation="swish",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            kernel_initializer="glorot_uniform",
        )

        # It is crucial to call the parent's build method at the end.
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Defines the forward pass logic for the layer.

        Args:
            inputs (list or tuple): A list of three input tensors:
                                    [cnn_features, lstm_features_1, lstm_features_2].
            training (bool): A flag indicating if the layer should behave in
                             training mode (e.g., for dropout).

        Returns:
            A fused tensor representation.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError(
                "Expected a list or tuple of 3 input tensors. "
                f"Received {len(inputs)} inputs."
            )

        cnn_feat, lstm_feat1, lstm_feat2 = inputs

        # Project input features into the common dimension `self.dim`
        cnn_proj = self.norm_cnn(self.cnn_proj(cnn_feat))
        lstm_proj1 = self.norm_lstm(self.lstm_proj1(lstm_feat1))
        lstm_proj2 = self.norm_lstm(self.lstm_proj2(lstm_feat2))

        # Combine LSTM features to be used as key/value in attention
        lstm_combined = tf.stack([lstm_proj1, lstm_proj2], axis=1)

        # Prepare CNN features for attention mechanism
        cnn_seq = self.cnn_seq_proj(cnn_proj)
        cnn_seq = tf.reshape(cnn_seq, [-1, 2, self.dim])

        # 1. CNN branch attends to LSTM branch
        cnn_as_query = tf.expand_dims(cnn_proj, axis=1)
        cnn_attended = self.cnn_attention(
            query=cnn_as_query, key=lstm_combined, value=lstm_combined
        )
        # Add & Norm: Residual connection
        cnn_enhanced = self.norm_cross1(cnn_as_query + cnn_attended)

        # 2. LSTM branch attends to CNN branch
        lstm_attended = self.lstm_attention(
            query=lstm_combined, key=cnn_seq, value=cnn_seq
        )
        # Add & Norm: Residual connection
        lstm_enhanced = self.norm_cross2(lstm_combined + lstm_attended)

        # Flatten the enhanced features before concatenation
        cnn_flat = tf.reshape(cnn_enhanced, [-1, self.dim])
        lstm_flat = tf.reshape(lstm_enhanced, [-1, 2 * self.dim])

        # Concatenate all features for the final gating mechanism
        combined = tf.concat([cnn_flat, lstm_flat, cnn_proj], axis=-1)

        # Gating mechanism to control information flow
        pre_gated = self.pre_gate(combined)
        gate_weights = self.gate(pre_gated)
        gated = pre_gated * gate_weights

        # Final projection with dropout for regularization
        return self.combine(self.dropout(gated, training=training))

    def get_config(self):
        """Enables the layer to be serialized by Keras."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
