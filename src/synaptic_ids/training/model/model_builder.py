# pylint: disable=no-member
from tensorflow import keras
from src.synaptic_ids.training.model.transformer_fusion import TransformerFusion


# ToDO: Simplify CNN architecture and TransformerFusion and analyse results
class SynapticIDSBuilder:
    """
    Constructs the SynapticIDS model architecture.

    This class's single responsibility is to define the model's structure by
    assembling the CNN and Sequence branches and fusing them. It separates the
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

        squeeze = keras.layers.GlobalAveragePooling2D()(input_tensor)
        excitation = keras.layers.Dense(ratio, activation="swish")(squeeze)
        excitation = keras.layers.Dense(channels, activation="sigmoid")(excitation)
        return keras.layers.Multiply()(
            [input_tensor, keras.layers.Reshape((1, 1, channels))(excitation)]
        )

    def _multi_scale_cnn(self, input_tensor, block_index):
        """Defines a multi-scale convolutional block using dilated convolutions."""
        base_filters = 12
        l2_reg = 1e-5 if block_index < 2 else 2e-5

        def dilated_conv(x, dilation_rate):
            x = keras.layers.DepthwiseConv2D(
                (3, 3),
                dilation_rate=dilation_rate,
                padding="same",
                depthwise_regularizer=keras.regularizers.l2(l2_reg),
            )(x)
            x = keras.layers.Conv2D(base_filters // 3, 1)(x)
            return x

        b1 = dilated_conv(input_tensor, 1)
        b2 = dilated_conv(input_tensor, 2)
        b3 = dilated_conv(input_tensor, 3)

        b1 = self._adaptive_se_block(b1)
        b2 = self._adaptive_se_block(b2)
        b3 = self._adaptive_se_block(b3)

        fused = keras.layers.Concatenate()([b1, b2, b3])

        fused = keras.layers.Conv2D(
            base_filters,
            1,
            groups=4,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )(fused)

        if input_tensor.shape[-1] == base_filters:
            fused = keras.layers.Add()([fused, input_tensor])

        return keras.layers.MaxPooling2D(2)(fused)

    def _build_cnn_branch(self):
        """Constructs the complete CNN branch for image feature extraction."""
        input_layer = keras.layers.Input(shape=self.image_shape)

        x = keras.layers.Conv2D(
            16,
            (3, 3),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(2e-5),
        )(input_layer)
        x = keras.layers.BatchNormalization(epsilon=2e-5)(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.MaxPooling2D(2)(x)

        x = self._multi_scale_cnn(x, block_index=1)
        x = keras.layers.SpatialDropout2D(0.3)(x)

        x = self._multi_scale_cnn(x, block_index=2)
        x = keras.layers.SpatialDropout2D(0.3)(x)

        x = keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            dilation_rate=2,
            kernel_regularizer=keras.regularizers.l2(2e-5),
        )(x)
        x = keras.layers.BatchNormalization(epsilon=2e-5)(x)
        x = keras.layers.Activation("swish")(x)

        attention = keras.layers.Conv2D(
            1, (1, 1), activation="sigmoid", kernel_initializer="glorot_normal"
        )(x)
        x = keras.layers.Multiply()([x, attention])

        return keras.models.Model(
            inputs=input_layer,
            outputs=[
                keras.layers.GlobalAveragePooling2D()(x),
                keras.layers.GlobalMaxPooling2D()(x),
            ],
        )

    def _build_sequence_branch(self, sequence_units=None, dropout_rate=0.3):
        """Constructs the complete Sequence branch for sequential feature extraction."""
        if sequence_units is None:
            sequence_units = [64, 32]

        input_layer = keras.layers.Input(shape=self.sequence_shape)
        x = input_layer

        for _, units in enumerate(sequence_units):
            x = keras.layers.Bidirectional(
                keras.layers.GRU(units, return_sequences=True)
            )(x)
            x = keras.layers.LayerNormalization()(x)
            x = keras.layers.SpatialDropout1D(dropout_rate)(x)

        def se_block_1d(input_tensor):
            channels = input_tensor.shape[-1]
            squeeze = keras.layers.GlobalAveragePooling1D()(input_tensor)
            excitation = keras.layers.Dense(max(4, channels // 16), activation="swish")(
                squeeze
            )
            excitation = keras.layers.Dense(channels, activation="sigmoid")(excitation)
            return keras.layers.Multiply()(
                [input_tensor, keras.layers.Reshape((1, channels))(excitation)]
            )

        x = se_block_1d(x)

        feature1 = keras.layers.GlobalAveragePooling1D()(x)
        feature2 = keras.layers.GlobalMaxPooling1D()(x)

        return keras.models.Model(inputs=input_layer, outputs=[feature1, feature2])

    def build(self):
        """
        Assembles the final model from the CNN and Sequence branches and returns the
        uncompiled Keras model.

        Returns:
            A tf.keras.Model instance representing the complete architecture.
        """
        input_cnn = keras.layers.Input(shape=self.image_shape, name="image_input")
        input_sequence = keras.layers.Input(
            shape=self.sequence_shape, name="sequence_input"
        )

        cnn_model = self._build_cnn_branch()
        sequence_model = self._build_sequence_branch()

        cnn_outputs = cnn_model(input_cnn)
        sequence_outputs = sequence_model(input_sequence)

        fusion_layer = TransformerFusion()
        fused_features = [
            fusion_layer([cnn_feat, sequence_outputs[0], sequence_outputs[1]])
            for cnn_feat in cnn_outputs
        ]

        z = keras.layers.Concatenate()(fused_features)
        z = keras.layers.LayerNormalization(epsilon=1e-6)(z)
        z = keras.layers.Dense(64, activation="swish")(z)
        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.Dropout(0.4)(z)
        z = keras.layers.Dense(
            64, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-4)
        )(z)
        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.Dropout(0.3)(z)

        if self.mode == "binary":
            output = keras.layers.Dense(1, activation="sigmoid", name="output")(z)
        else:
            output = keras.layers.Dense(
                self.num_classes, activation="softmax", name="output"
            )(z)

        final_model = keras.models.Model(
            inputs=[input_cnn, input_sequence], outputs=output
        )
        print("SynapticIDS model built successfully.")
        final_model.summary()

        return final_model
