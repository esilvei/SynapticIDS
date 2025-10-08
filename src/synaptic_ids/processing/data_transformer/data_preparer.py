from typing import Dict, Optional

import numpy as np
import pandas as pd
import redis
import tensorflow as tf

from .to_2d import UNSWNB15ToImage
from .to_sequence import SequenceGenerator
from ..feature_engineer import UNSWNB15FeatureEngineer


class DataPreparer:
    """
    Orchestrates the feature engineering and data transformation steps,
    providing a clean interface to the training pipeline.
    """

    def __init__(
        self, feature_engineer: UNSWNB15FeatureEngineer, mode: str = "multiclass"
    ):
        """
        Initializes the DataPreparer.

        Args:
            feature_engineer: A pre-initialized instance of the feature engineer.
            mode (str): The operational mode, 'multiclass' or 'binary'.
        """
        self.feature_engineer = feature_engineer
        self.sequence_generator = SequenceGenerator()
        self.mode = mode

        self.image_transformer: Optional[UNSWNB15ToImage] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the entire data preparation pipeline, including the feature engineer
        and the subsequent image transformer.
        """
        print("Fitting Data Preparer...")
        self.feature_engineer.fit(df)

        # Get the list of features that were finalized during the fitting process.
        final_features = self.feature_engineer.final_selected_features
        print(f"Initializing data transformers with {len(final_features)} features.")

        self.image_transformer = UNSWNB15ToImage(feature_names=final_features)

        self.is_fitted = True
        print("Data Preparer fitted successfully.")
        return self

    async def prepare_data(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        redis_client: Optional[redis.Redis] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Prepares the final data dictionary for model training or prediction.

        Args:
            df (pd.DataFrame): The raw dataframe to prepare.
            is_training (bool): Flag to indicate if this is for training (for future augmentations).

        Returns:
            A dictionary containing 'images', 'sequences', and 'labels'.
        """
        if not self.is_fitted or self.image_transformer is None:
            raise RuntimeError("DataPreparer must be fitted before preparing data.")

        is_inference = not is_training
        # Step 1: Feature Engineering
        x_eng, y_eng = self.feature_engineer.transform(df, is_inference=is_inference)

        # Step 2: Generate Sequences from the engineered features
        if redis_client and session_id:
            x_sequences = await self.sequence_generator.generate_online_sequence(
                x_eng, redis_client, session_id
            )
            y_sequences = None
            valid_indices = x_eng.index
        else:
            x_sequences, y_sequences, valid_indices = (
                self.sequence_generator.generate_offline(x_eng, y_eng)
            )

        # Handle cases where no sequences can be generated (e.g., very small input df)
        if len(x_sequences) == 0:
            return {
                "images": np.array([]),
                "sequences": np.array([]),
                "labels": None,
            }

        # Step 3: Generate Images for the corresponding valid sequence endpoints
        x_images_source = x_eng.iloc[valid_indices]
        x_images = self.image_transformer(x_images_source.values, training=is_training)

        # Step 4: One-hot encode labels if in multiclass mode
        labels = None
        if y_sequences is not None:
            if self.mode == "multiclass":
                num_classes = len(self.feature_engineer.label_encoder.classes_)
                labels = tf.keras.utils.to_categorical(
                    y_sequences, num_classes=num_classes
                )
            else:
                labels = np.array(y_sequences, dtype="int32")
        return {
            "images": np.array(x_images, dtype="float32"),
            "sequences": np.array(x_sequences, dtype="float32"),
            "labels": labels,
        }
