from typing import Tuple, List, Optional

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
            self, x: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes engineered features and labels and generates sequences. Adapts
        its strategy based on whether labels are provided (training vs. inference).

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
        y_np = y.values if y is not None else None
        n_samples = x.shape[0]

        if y is not None:
            print("Offline mode: using sliding window to generate sequences.")
            if n_samples < self.sequence_length:
                return np.array([]), np.array([]), np.array([])

            indices_list: List[List[int]] = []
            for i in range(n_samples - self.sequence_length + 1):
                indices_list.append(list(range(i, i + self.sequence_length)))

            indices_np = np.array(indices_list)
            sequences = np.array([x_temporal[idx] for idx in indices_np])
            # A label de uma sequência é a label do seu último elemento.
            labels = y_np[indices_np[:, -1]]
            valid_indices = indices_np[:, -1]
        else:
            print(f"Online mode: handling batch of {n_samples} samples.")

            sequences_temp = x_temporal[:, np.newaxis, :]

            sequences = np.repeat(sequences_temp, self.sequence_length, axis=1)

            labels = None
            valid_indices = np.arange(n_samples)

        print(f"Generated {len(sequences)} sequences.")
        final_sequences = sequences.astype("float32")
        final_labels = labels.astype("int32") if labels is not None else None

        return final_sequences, final_labels, valid_indices