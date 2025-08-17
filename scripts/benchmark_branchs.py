import time
import os
import numpy as np
from src.synaptic_ids.training.model.model_builder import SynapticIDSBuilder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def benchmark_model_branches(
    batch_size=128,
    image_shape=(32, 32, 1),
    sequence_shape=(5, 48),
    num_classes=10,
    iterations=100,
):
    """
    Measures and compares the inference time of the model's CNN and sequence branches.

    Args:
        batch_size (int): The batch size for inference.
        image_shape (tuple): The shape of the input image data.
        sequence_shape (tuple): The shape of the input sequence data.
        num_classes (int): The number of output classes.
        iterations (int): The number of iterations to average the timing over.
    """
    print("--- STARTING MODEL BRANCH BENCHMARK ---")
    print(f"Batch Size: {batch_size}, Iterations: {iterations}\n")

    # 1. Create an instance of your model builder
    builder = SynapticIDSBuilder(
        image_shape=image_shape,
        sequence_shape=sequence_shape,
        num_classes=num_classes,
    )

    # 2. Build each model branch separately
    print("Building individual branches...")
    cnn_branch = builder._build_cnn_branch()  # pylint: disable=protected-access
    sequence_branch = builder._build_sequence_branch()  # pylint: disable=protected-access
    print("Branches built successfully.\n")

    # 3. Create dummy input data
    print("Generating dummy input data...")
    dummy_images = np.random.rand(batch_size, *image_shape).astype("float32")
    dummy_sequences = np.random.rand(batch_size, *sequence_shape).astype("float32")
    print("Data generated.\n")

    # --- Benchmarking the CNN Branch ---
    print("--- Benchmarking CNN Branch ---")
    # "Warm-up" - The first prediction is always slower
    print("Running warm-up for the CNN branch...")
    _ = cnn_branch.predict(dummy_images, verbose=0)

    print(f"Running {iterations} inference iterations...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = cnn_branch.predict(dummy_images, verbose=0)
    end_time = time.perf_counter()

    cnn_total_time = end_time - start_time
    cnn_avg_time_per_batch = (cnn_total_time / iterations) * 1000  # in milliseconds

    print(f"CNN Branch Total Time: {cnn_total_time:.4f} seconds")
    print(f"Average time per batch: {cnn_avg_time_per_batch:.2f} ms\n")

    # --- Benchmarking the Sequence (GRU) Branch ---
    print("--- Benchmarking Sequence (GRU) Branch ---")
    # "Warm-up"
    print("Running warm-up for the sequence branch...")
    _ = sequence_branch.predict(dummy_sequences, verbose=0)

    print(f"Running {iterations} inference iterations...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = sequence_branch.predict(dummy_sequences, verbose=0)
    end_time = time.perf_counter()

    sequence_total_time = end_time - start_time
    sequence_avg_time_per_batch = (
        sequence_total_time / iterations
    ) * 1000  # in milliseconds

    print(f"Sequence Branch Total Time: {sequence_total_time:.4f} seconds")
    print(f"Average time per batch: {sequence_avg_time_per_batch:.2f} ms\n")

    # --- Conclusion ---
    print("--- BENCHMARK CONCLUSION ---")
    if cnn_avg_time_per_batch > sequence_avg_time_per_batch:
        # Prevent division by zero if one branch is extremely fast
        if sequence_avg_time_per_batch > 0:
            ratio = cnn_avg_time_per_batch / sequence_avg_time_per_batch
            print(
                f"The CNN Branch is the bottleneck, being {ratio:.1f}x slower"
                f" than the sequence branch."
            )
        else:
            print("The CNN Branch is the bottleneck.")
    else:
        if cnn_avg_time_per_batch > 0:
            ratio = sequence_avg_time_per_batch / cnn_avg_time_per_batch
            print(
                f"The Sequence Branch is the bottleneck, "
                f"being {ratio:.1f}x slower than the CNN branch."
            )
        else:
            print("The Sequence Branch is the bottleneck.")


if __name__ == "__main__":
    benchmark_model_branches(sequence_shape=(5, 96))
