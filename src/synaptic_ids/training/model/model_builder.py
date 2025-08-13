# pylint: disable=no-member
import tensorflow as tf
from src.synaptic_ids.training.model.transformer_fusion import TransformerFusion


class SynapticIDSBuilder:
    """
    Constructs the SynapticIDS model architecture.

    This class's single responsibility is to define the model's structure by
    assembling the CNN and LSTM branches and fusing them. It separates the
    architectural blueprint from the training and data handling logic.
    """

    def __init__(self, image_shape, sequence_shape, num_classes, mode="multiclass"):
        """
        Initializes the builder with the required architectural parameters.

        Args:
            image_shape (tuple): The shape of the input image data.
            sequence_shape (tuple): The shape of the input sequence data.
            num_classes (int): The number of output classes for the classifier.
            mode (str): The operational mode, 'multiclass' or 'binary'.
        """
        self.image_shape = image_shape
        self.sequence_shape = sequence_shape
        self.num_classes = num_classes
        self.mode = mode

    def _adaptive_se_block(self, input_tensor):
        """Defines a Squeeze-and-Excitation block for channel-wise attention."""
        channels = input_tensor.shape[-1]
        ratio = max(4, channels // 8)

        squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        excitation = tf.keras.layers.Dense(ratio, activation="swish")(squeeze)
        excitation = tf.keras.layers.Dense(channels, activation="sigmoid")(excitation)
        return tf.keras.layers.Multiply()(
            [input_tensor, tf.keras.layers.Reshape((1, 1, channels))(excitation)]
        )

    def _multi_scale_cnn(self, input_tensor, block_index):
        """Defines a multi-scale convolutional block using dilated convolutions."""
        base_filters = 48
        l2_reg = 1e-5 if block_index < 2 else 2e-5

        def dilated_conv(x, dilation_rate):
            x = tf.keras.layers.DepthwiseConv2D(
                (3, 3),
                dilation_rate=dilation_rate,
                padding="same",
                depthwise_regularizer=tf.keras.regularizers.l2(l2_reg),
            )(x)
            x = tf.keras.layers.Conv2D(base_filters // 3, 1)(x)
            return x

        b1 = dilated_conv(input_tensor, 1)
        b2 = dilated_conv(input_tensor, 2)
        b3 = dilated_conv(input_tensor, 3)

        b1 = self._adaptive_se_block(b1)
        b2 = self._adaptive_se_block(b2)
        b3 = self._adaptive_se_block(b3)

        fused = tf.keras.layers.Concatenate()([b1, b2, b3])

        fused = tf.keras.layers.Conv2D(
            base_filters,
            1,
            groups=4,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        )(fused)

        if input_tensor.shape[-1] == base_filters:
            fused = tf.keras.layers.Add()([fused, input_tensor])

        return tf.keras.layers.MaxPooling2D(2)(fused)

    def _build_cnn_branch(self):
        """Constructs the complete CNN branch for image feature extraction."""
        input_layer = tf.keras.layers.Input(shape=self.image_shape)

        x = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(2e-5),
        )(input_layer)
        x = tf.keras.layers.BatchNormalization(epsilon=2e-5)(x)
        x = tf.keras.layers.Activation("swish")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

        x = self._multi_scale_cnn(x, block_index=1)
        x = tf.keras.layers.SpatialDropout2D(0.3)(x)

        x = self._multi_scale_cnn(x, block_index=2)
        x = tf.keras.layers.SpatialDropout2D(0.3)(x)

        x = tf.keras.layers.Conv2D(
            96,
            (3, 3),
            padding="same",
            dilation_rate=2,
            kernel_regularizer=tf.keras.regularizers.l2(2e-5),
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=2e-5)(x)
        x = tf.keras.layers.Activation("swish")(x)

        attention = tf.keras.layers.Conv2D(
            1, (1, 1), activation="sigmoid", kernel_initializer="glorot_normal"
        )(x)
        x = tf.keras.layers.Multiply()([x, attention])

        return tf.keras.models.Model(
            inputs=input_layer,
            outputs=[
                tf.keras.layers.GlobalAveragePooling2D()(x),
                tf.keras.layers.GlobalMaxPooling2D()(x),
            ],
        )

    def _build_lstm_branch(self, lstm_units=None, dropout_rate=0.3):
        """Constructs the complete LSTM branch for sequential feature extraction."""
        if lstm_units is None:
            lstm_units = [64, 32]

        input_layer = tf.keras.layers.Input(shape=self.sequence_shape)
        x = input_layer

        for _, units in enumerate(lstm_units):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True)
            )(x)
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)

        def se_block_1d(input_tensor):
            channels = input_tensor.shape[-1]
            squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
            excitation = tf.keras.layers.Dense(
                max(4, channels // 16), activation="swish"
            )(squeeze)
            excitation = tf.keras.layers.Dense(channels, activation="sigmoid")(
                excitation
            )
            return tf.keras.layers.Multiply()(
                [input_tensor, tf.keras.layers.Reshape((1, channels))(excitation)]
            )

        x = se_block_1d(x)

        feature1 = tf.keras.layers.GlobalAveragePooling1D()(x)
        feature2 = tf.keras.layers.GlobalMaxPooling1D()(x)

        return tf.keras.models.Model(inputs=input_layer, outputs=[feature1, feature2])

    def build(self):
        """
        Assembles the final model from the CNN and LSTM branches and returns the
        uncompiled Keras model.

        Returns:
            A tf.keras.Model instance representing the complete architecture.
        """
        input_cnn = tf.keras.layers.Input(shape=self.image_shape, name="image_input")
        input_lstm = tf.keras.layers.Input(
            shape=self.sequence_shape, name="sequence_input"
        )

        cnn_model = self._build_cnn_branch()
        lstm_model = self._build_lstm_branch()

        cnn_outputs = cnn_model(input_cnn)
        lstm_outputs = lstm_model(input_lstm)

        fusion_layer = TransformerFusion()
        fused_features = [
            fusion_layer([cnn_feat, lstm_outputs[0], lstm_outputs[1]])
            for cnn_feat in cnn_outputs
        ]

        z = tf.keras.layers.Concatenate()(fused_features)
        z = tf.keras.layers.LayerNormalization(epsilon=1e-6)(z)
        z = tf.keras.layers.Dense(128, activation="swish")(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Dropout(0.4)(z)
        z = tf.keras.layers.Dense(
            64, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Dropout(0.3)(z)

        if self.mode == "binary":
            output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(z)
        else:
            output = tf.keras.layers.Dense(
                self.num_classes, activation="softmax", name="output"
            )(z)

        final_model = tf.keras.models.Model(
            inputs=[input_cnn, input_lstm], outputs=output
        )
        print("SynapticIDS model built successfully.")
        final_model.summary()

        return final_model
