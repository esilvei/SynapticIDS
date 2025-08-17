from typing import Dict, Optional

import numpy as np
import pandas as pd
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
        self.sequence_generator = SequenceGenerator()  # Handles sequence creation
        self.mode = mode

        self.image_transformer: Optional[UNSWNB15ToImage] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the entire data preparation pipeline, including the feature engineer
        and the subsequent image transformer.
        """
        print("Fitting Data Preparer...")
        # First, fit the feature engineer to learn encodings, scaling, and to select features.
        self.feature_engineer.fit(df)

        # Get the list of features that were finalized during the fitting process.
        final_features = self.feature_engineer.final_selected_features
        print(f"Initializing data transformers with {len(final_features)} features.")

        self.image_transformer = UNSWNB15ToImage(feature_names=final_features)

        self.is_fitted = True
        print("Data Preparer fitted successfully.")
        return self

    def prepare_data(self, df: pd.DataFrame, is_training: bool = False) -> Dict:
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

        # Step 1: Feature Engineering
        x_eng, y_eng = self.feature_engineer.transform(df)

        # Step 2: Generate Sequences from the engineered features
        x_sequences, y_sequences, valid_indices = self.sequence_generator.generate(
            x_eng, y_eng
        )

        # Handle cases where no sequences can be generated (e.g., very small input df)
        if len(x_sequences) == 0:
            return {
                "images": np.array([]),
                "sequences": np.array([]),
                "labels": np.array([]),
            }

        # Step 3: Generate Images for the corresponding valid sequence endpoints
        x_images_source = x_eng.iloc[valid_indices]
        x_images = self.image_transformer(x_images_source.values, training=is_training)

        # Step 4: One-hot encode labels if in multiclass mode
        labels = y_sequences
        if self.mode == "multiclass":
            # Ensure the label encoder is fitted before using it
            if not hasattr(self.feature_engineer.label_encoder, "classes_"):
                raise RuntimeError("LabelEncoder in FeatureEngineer is not fitted.")
            num_classes = len(self.feature_engineer.label_encoder.classes_)
            labels = tf.keras.utils.to_categorical(y_sequences, num_classes=num_classes)

        return {
            "images": np.array(x_images, dtype="float32"),
            "sequences": np.array(x_sequences, dtype="float32"),
            "labels": np.array(labels, dtype="float32"),
        }
