# pylint: disable=no-member
import tensorflow as tf

# pylint: disable=no-member
from tensorflow.keras import layers, regularizers


class TransformerFusion(layers.Layer):
    """
    A custom layer to fuse multimodal features using a cross-attention mechanism.

    This layer takes three feature tensors (one from a CNN branch and two from an
    LSTM branch) and projects them into a common dimension. It then uses
    Multi-Head Attention to allow the branches to enrich each other's representations
    before combining them into a final, fused output.
    """

    def __init__(self, dim=128, num_heads=8, dropout_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Layer definitions moved to __init__ to resolve pylint warnings
        self.cnn_proj = layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.lstm_proj1 = layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.lstm_proj2 = layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.cnn_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_initializer="glorot_uniform",
        )
        self.lstm_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_initializer="glorot_uniform",
        )
        self.norm_cnn = layers.LayerNormalization(epsilon=1e-6)
        self.norm_lstm = layers.LayerNormalization(epsilon=1e-6)
        self.norm_cross1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_cross2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.cnn_seq_proj = layers.Dense(
            self.dim * 2, kernel_initializer="glorot_uniform"
        )
        self.pre_gate = layers.Dense(
            self.dim, activation="swish", kernel_initializer="glorot_uniform"
        )
        self.gate = layers.Dense(
            self.dim, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self.combine = layers.Dense(
            self.dim,
            activation="swish",
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer="glorot_uniform",
        )

    # pylint: disable=too-many-locals
    def call(self, inputs, training=None):
        """Defines the forward pass logic for the layer."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("Expected 3 input tensors")

        cnn_feat, lstm_feat1, lstm_feat2 = inputs

        cnn_proj = self.norm_cnn(self.cnn_proj(cnn_feat))
        lstm_proj1 = self.norm_lstm(self.lstm_proj1(lstm_feat1))
        lstm_proj2 = self.norm_lstm(self.lstm_proj2(lstm_feat2))

        lstm_combined = tf.stack([lstm_proj1, lstm_proj2], axis=1)

        cnn_seq = self.cnn_seq_proj(cnn_proj)
        cnn_seq = tf.reshape(cnn_seq, [-1, 2, self.dim])

        cnn_as_query = tf.expand_dims(cnn_proj, axis=1)
        cnn_attended = self.cnn_attention(
            query=cnn_as_query, key=lstm_combined, value=lstm_combined
        )
        cnn_enhanced = self.norm_cross1(cnn_as_query + cnn_attended)

        lstm_attended = self.lstm_attention(
            query=lstm_combined, key=cnn_seq, value=cnn_seq
        )
        lstm_enhanced = self.norm_cross2(lstm_combined + lstm_attended)

        cnn_flat = tf.reshape(cnn_enhanced, [-1, self.dim])
        lstm_flat = tf.reshape(lstm_enhanced, [-1, 2 * self.dim])

        combined = tf.concat([cnn_flat, lstm_flat, cnn_proj], axis=-1)

        pre_gated = self.pre_gate(combined)
        gate_weights = self.gate(pre_gated)
        gated = pre_gated * gate_weights

        return self.combine(self.dropout(gated, training=training))

    def get_config(self):
        """Enables the layer to be serialized."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
