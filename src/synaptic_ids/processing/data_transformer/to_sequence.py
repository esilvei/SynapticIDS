from typing import Tuple
import json
import numpy as np
import pandas as pd
import redis.asyncio


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

    async def generate_online_sequence(
        self, x: pd.DataFrame, redis_client: redis.Redis, session_id: str
    ) -> np.ndarray:
        """
        Handles online sequence generation for a BATCH of new records using Redis.
        It processes each record sequentially to build the correct temporal context.
        """
        print(f"Online mode: handling {len(x)} samples for session {session_id}.")

        available_temporal = [f for f in self.temporal_features if f in x.columns]
        redis_key = f"session:{session_id}"

        generated_sequences = []

        # Iterate over each record in the input DataFrame
        for _, row in x[available_temporal].iterrows():
            new_record_dict = row.to_dict()

            # Get current sequence history from Redis
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.lpush(redis_key, json.dumps(new_record_dict))
                pipe.ltrim(redis_key, 0, self.sequence_length - 1)
                pipe.lrange(redis_key, 0, self.sequence_length - 1)
                results = await pipe.execute()
                current_sequence_json = results[2]

            # Build the sequence for the current record
            current_sequence_records = [
                json.loads(r) for r in reversed(current_sequence_json)
            ]
            sequence_df = pd.DataFrame(
                current_sequence_records, columns=available_temporal
            ).fillna(0)
            sequence_np = sequence_df.values

            # Apply padding if the sequence is not yet full
            current_len = len(sequence_np)
            if current_len < self.sequence_length:
                padding_size = self.sequence_length - current_len
                padding = np.zeros((padding_size, sequence_np.shape[1]))
                sequence_np = np.vstack([padding, sequence_np])

            generated_sequences.append(sequence_np)

        # Stack all generated sequences into a single numpy array for the batch
        final_sequences_batch = np.array(generated_sequences).astype("float32")

        print(
            f"Generated a batch of {final_sequences_batch.shape[0]} "
            f"sequences of shape {final_sequences_batch.shape} for prediction."
        )
        return final_sequences_batch

    def generate_offline(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handles offline batch processing for training.
        it also simulates session starts with zero-padding to match online behavior.
        """
        print("Offline mode: generating sequences with padding for consistency...")
        available_temporal = [f for f in self.temporal_features if f in x.columns]
        x_temporal = x[available_temporal].values
        n_samples, n_features = x_temporal.shape

        if n_samples < 1:
            return np.array([]), np.array([]), np.array([])

        padded_x = np.zeros((n_samples + self.sequence_length - 1, n_features))
        padded_x[self.sequence_length - 1 :] = x_temporal

        indices_list = [
            list(range(i, i + self.sequence_length)) for i in range(n_samples)
        ]
        indices_np = np.array(indices_list)

        sequences = np.array([padded_x[idx] for idx in indices_np])
        labels = None
        if y is not None:
            labels = y.values.astype("int32")
        valid_indices = np.arange(n_samples)

        print(f"Generated {len(sequences)} sequences.")

        return sequences.astype("float32"), labels, valid_indices
