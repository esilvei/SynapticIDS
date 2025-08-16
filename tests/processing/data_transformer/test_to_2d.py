import pytest
import numpy as np
import tensorflow as tf

from src.synaptic_ids.processing.data_transformer.to_2d import UNSWNB15ToImage


@pytest.fixture
def sample_data():
    """Creates sample data for the tests."""
    # Batch of 4 samples, 5 features
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    return tf.constant(data)


@pytest.fixture
def layer_instance():
    """Creates an instance of the layer for testing."""
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    return UNSWNB15ToImage(feature_names=feature_names, image_size=(16, 16))


def test_layer_initialization(layer_instance):
    """Tests if the layer is initialized with the correct attributes."""
    assert layer_instance.n_features == 5
    assert layer_instance.image_size == (16, 16)
    assert not layer_instance.built


def test_layer_build_method(layer_instance, sample_data):
    """Tests if the layer creates its weights correctly after being built."""
    # Calling the layer triggers the build() method
    _ = layer_instance(sample_data, training=True)

    assert layer_instance.built
    assert layer_instance.feature_min.shape == (5,)
    assert layer_instance.feature_max.shape == (5,)
    assert layer_instance.feature_ordering.shape == (5,)
    assert layer_instance.is_initialized.dtype == tf.bool


def test_first_training_call_initializes_stats(layer_instance, sample_data):
    """
    Tests if the first call with training=True correctly initializes
    the min/max statistics.
    """
    # Act
    _ = layer_instance(sample_data, training=True)

    # Assert
    expected_min = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    expected_max = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    # Compare the tensor values
    np.testing.assert_allclose(
        layer_instance.feature_min.numpy(), expected_min, rtol=1e-6
    )
    np.testing.assert_allclose(
        layer_instance.feature_max.numpy(), expected_max, rtol=1e-6
    )
    # Corrected the singleton comparison from '== True' to a direct assertion
    assert layer_instance.is_initialized.numpy()


def test_subsequent_training_call_updates_stats(layer_instance, sample_data):
    """
    Tests if a second call with training=True updates the statistics.
    """
    # Arrange: First call to initialize
    _ = layer_instance(sample_data, training=True)

    # Act: Second call with new data that should expand the min/max range
    new_data = tf.constant(np.array([[-1.0, 10.0, 1.0, 12.0, 3.0]], dtype=np.float32))
    _ = layer_instance(new_data, training=True)

    # Assert: Verify that the min/max values were updated
    expected_min = np.array([-1.0, 1.0, 1.0, 3.0, 3.0])
    expected_max = np.array([5.0, 10.0, 7.0, 12.0, 9.0])

    np.testing.assert_allclose(
        layer_instance.feature_min.numpy(), expected_min, rtol=1e-6
    )
    np.testing.assert_allclose(
        layer_instance.feature_max.numpy(), expected_max, rtol=1e-6
    )


def test_inference_call_does_not_update_stats(layer_instance, sample_data):
    """
    Tests that a call with training=False does NOT update the statistics.
    """
    # Arrange: "Train" the layer with the initial data
    _ = layer_instance(sample_data, training=True)
    initial_min = layer_instance.feature_min.numpy()
    initial_max = layer_instance.feature_max.numpy()

    # Act: Call the layer in inference mode with data that could change the stats
    inference_data = tf.constant(
        np.array([[-10.0, 100.0, -5.0, 50.0, 0.0]], dtype=np.float32)
    )
    _ = layer_instance(inference_data, training=False)

    # Assert: The statistics should not have changed
    np.testing.assert_allclose(
        layer_instance.feature_min.numpy(), initial_min, rtol=1e-6
    )
    np.testing.assert_allclose(
        layer_instance.feature_max.numpy(), initial_max, rtol=1e-6
    )


def test_output_shape_and_value_range(layer_instance, sample_data):
    """
    Tests if the output image has the correct shape and value range.
    """
    # Act
    output_images = layer_instance(sample_data, training=True)
    output_images_np = output_images.numpy()

    # Assert: Check the shape (batch_size, height, width, channels)
    assert output_images.shape == (4, 16, 16, 1)

    # Assert: Check that all values are in the [0, 1] range
    assert np.all(output_images_np >= 0.0)
    assert np.all(output_images_np <= 1.0)
