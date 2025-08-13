from typing import Tuple, List

import numpy as np
import pandas as pd


class SequenceGenerator:
    """
    Generates temporal sequences from tabular data. It handles both offline
    batch processing for training and online single-instance processing for prediction.
    """

    def __init__(self, sequence_length: int = 5, num_clusters: int = 10):
        self.sequence_length = sequence_length
        self.num_clusters = num_clusters
        # These features are based on the original notebook's logic
        self.temporal_features = [
            "dur",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "rate",
            "sload",
            "dload",
            "sloss",
            "dloss",
            "sinpkt",
            "dinpkt",
            "sjit",
            "djit",
            "swin",
            "stcpb",
            "dtcpb",
            "dwin",
            "tcprtt",
            "synack",
            "ackdat",
            "smean",
            "dmean",
            "trans_depth",
            "response_body_len",
            "duplicate_score",
            "sbytes_dbytes_ratio",
            "spkts_dpkts_ratio",
        ]

    def generate(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes engineered features and labels and generates sequences. Adapts
        its strategy based on the input data size.

        Args:
            x (pd.DataFrame): The input features.
            y (pd.Series): The corresponding labels.

        Returns:
            A tuple containing (sequences, labels, valid_indices_for_images).
        """
        print("Generating temporal sequences...")
        # Filters by available temporal features
        available_temporal = [f for f in self.temporal_features if f in x.columns]
        x_temporal = x[available_temporal].values
        y_np = y.values

        # Handles online (single prediction) vs. offline (batch) cases
        if x.shape[0] == 1:
            # Online case: replicate the single data point to form a sequence
            print("Online mode: replicating single data point for sequence.")
            sequences = np.repeat(x_temporal, self.sequence_length, axis=0)
            sequences = sequences.reshape(
                1, self.sequence_length, -1
            )  # Reshape to (batch_size, seq_len, n_features)
            labels = y_np
            valid_indices = x.index.values
        else:
            # Offline case: use a sliding window to generate sequences, which is
            # more efficient and realistic than K-Means for sequential data.
            print("Offline mode: using sliding window to generate sequences.")
            indices_list: List[List[int]] = []
            for i in range(len(x_temporal) - self.sequence_length + 1):
                indices_list.append(list(range(i, i + self.sequence_length)))

            if not indices_list:
                return np.array([]), np.array([]), np.array([])

            indices_np = np.array(indices_list)

            sequences = np.array([x_temporal[idx] for idx in indices_np])
            # The label for a sequence is the label of its last element
            labels = y_np[indices_np[:, -1]]
            valid_indices = indices_np[:, -1]

        print(f"Generated {len(sequences)} sequences.")
        return sequences.astype("float32"), labels.astype("int32"), valid_indices
