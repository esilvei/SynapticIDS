from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from boruta import BorutaPy


class UNSWNB15FeatureEngineer:
    """
    Handles all feature engineering for the UNSW-NB15 dataset.
    This includes cleaning, encoding, creating new features, feature selection,
    and scaling. It operates in a fit/transform paradigm to prevent data leakage.
    It receives all its configuration from an external source.
    """

    def __init__(self, mode: str, target_col: str, selected_features: List[str]):
        """
        Initializes the Feature Engineer with parameters from a config object.

        Args:
            mode (str): The operational mode ('multiclass' or 'binary').
            target_col (str): The name of the target column for the specified mode.
            selected_features (List[str]): A list of pre-selected features.
                                           If empty, feature selection will be run.
        """
        if mode not in ["binary", "multiclass"]:
            raise ValueError("Mode must be either 'binary' or 'multiclass'")
        self.mode = mode
        self.target_col = target_col
        self.initial_selected_features = selected_features  # Store the provided list

        # Artifacts learned during fit()
        self.freq_encoders: Dict[str, pd.Series] = {}
        self.label_encoder = LabelEncoder()
        self.final_selected_features: List[str] = []  # To be populated by fit()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.log_features = [
            "sbytes",
            "dbytes",
            "spkts",
            "dpkts",
            "sload",
            "dload",
            "dur",
            "ct_src_dport_ltm",
        ]

    def fit(self, df: pd.DataFrame):
        """
        Learns all necessary transformations (encoders, scalers, selected features)
        from the training data.
        """
        print("Fitting Feature Engineer...")
        df_processed = df.copy()

        # Step 1: Handle rare classes and encode target
        df_processed = self._remove_rare_classes(df_processed)
        if self.mode == "multiclass":
            self.label_encoder.fit(df_processed[self.target_col])
            df_processed[self.target_col] = self.label_encoder.transform(
                df_processed[self.target_col]
            )

        # Step 2: Clean and create features
        df_processed = self._handle_duplicates(df_processed)
        df_processed = self._encode_categoricals(df_processed, is_training=True)
        df_processed = self._create_interaction_features(df_processed)

        # Prepare data for feature selection/scaling
        # Always drop both potential target columns to be safe
        x = df_processed.drop(columns=["label", "attack_cat"], errors="ignore")
        y = df_processed[self.target_col]

        # --- REFACTORED FEATURE SELECTION ---
        # Step 3: Feature Selection
        if not self.initial_selected_features:  # Check if the list from config is empty
            print(
                "No pre-selected features in config. Running Boruta for feature selection..."
            )
            self._select_features_boruta(x, y)
        else:
            print(
                f"Using {len(self.initial_selected_features)} pre-selected features from config."
            )
            self.final_selected_features = self.initial_selected_features

        x_selected = x[self.final_selected_features]

        # Step 4: Outlier detection
        x_selected, y = self._remove_outliers(x_selected, y)

        # Step 5: Scaling (learns scaler on the final selected features)
        self.scaler.fit(x_selected)

        self.is_fitted = True
        print("Feature Engineer fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies the learned transformations to new data.
        """
        if not self.is_fitted:
            raise RuntimeError("Transform called before fitting the engineer.")

        print(f"Transforming data ({df.shape[0]} samples)...")
        df_processed = df.copy()

        # Apply target encoding if multiclass
        if self.mode == "multiclass":
            valid_labels = set(self.label_encoder.classes_)
            df_processed = df_processed[
                df_processed[self.target_col].isin(valid_labels)
            ].copy()
            df_processed[self.target_col] = self.label_encoder.transform(
                df_processed[self.target_col]
            )

        # Apply feature engineering steps
        df_processed = self._handle_duplicates(df_processed)
        df_processed = self._encode_categoricals(df_processed, is_training=False)
        df_processed = self._create_interaction_features(df_processed)

        # Ensure all selected features are present
        x = df_processed.drop(columns=["label", "attack_cat"], errors="ignore")
        y = df_processed[self.target_col]

        for col in self.final_selected_features:
            if col not in x.columns:
                x[col] = 0  # Add missing columns with a neutral value
        x = x[self.final_selected_features]  # Ensure correct order and selection

        # Apply scaling
        x_scaled = pd.DataFrame(
            self.scaler.transform(x),
            columns=self.final_selected_features,
            index=x.index,
        )

        # Optimize dtypes
        x_scaled = x_scaled.astype("float32")
        y = y.astype("int32")

        return x_scaled, y

    def _remove_rare_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "multiclass":
            class_freq = df[self.target_col].value_counts(normalize=True)
            rare_classes = class_freq[class_freq < 0.05].index
            return df[~df[self.target_col].isin(rare_classes)]
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        features_to_check = [
            col
            for col in df.columns
            if col not in [self.target_col, "label", "attack_cat"]
        ]

        # Ensure we only check for duplicates on columns that exist
        features_to_check = [f for f in features_to_check if f in df.columns]

        is_duplicate = (df[features_to_check] == df[features_to_check].shift(1)).all(
            axis=1
        )
        group_key = (~is_duplicate).cumsum()
        df["duplicate_score"] = df.groupby(group_key).cumcount() + 1
        return df

    def _encode_categoricals(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        categorical_cols = ["proto", "service", "state"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "UNKNOWN")
                if is_training:
                    freq_map = df[col].value_counts(normalize=True)
                    self.freq_encoders[col] = freq_map
                else:
                    # Use the stored encoder, handling unseen categories
                    freq_map = self.freq_encoders.get(col, pd.Series(dtype="float64"))

                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
                df.drop(columns=[col], inplace=True)
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-8
        for col in self.log_features:
            if col in df.columns and df[col].min() >= 0:
                df[col] = np.log1p(df[col])

        if "sbytes" in df.columns and "dbytes" in df.columns:
            df["sbytes_dbytes_ratio"] = df["sbytes"] / (df["dbytes"] + eps)
        if "spkts" in df.columns and "dpkts" in df.columns:
            df["spkts_dpkts_ratio"] = df["spkts"] / (df["dpkts"] + eps)
        if all(x in df.columns for x in ["sloss", "dloss", "spkts", "dpkts"]):
            df["loss_ratio"] = (df["sloss"] + df["dloss"]) / (
                df["spkts"] + df["dpkts"] + eps
            )
        if all(x in df.columns for x in ["spkts", "dpkts"]):
            df["packet_error_ratio"] = np.abs(df["spkts"] - df["dpkts"]) / (
                df["spkts"] + df["dpkts"] + eps
            )
        if "dur" in df.columns and all(x in df.columns for x in ["sbytes", "dbytes"]):
            df["throughput"] = (df["sbytes"] + df["dbytes"]) / (df["dur"] + eps)
        if all(x in df.columns for x in ["sload", "dload"]):
            df["load_ratio"] = df["sload"] / (df["dload"] + eps)
        if all(x in df.columns for x in ["sjit", "djit"]):
            df["jitter_ratio"] = df["sjit"] / (df["djit"] + eps)
        if all(x in df.columns for x in ["stcpb", "dtcpb"]):
            df["tcp_window_ratio"] = df["stcpb"] / (df["dtcpb"] + eps)
        if all(x in df.columns for x in ["ct_src_dport_ltm", "ct_dst_sport_ltm"]):
            df["port_conn_ratio"] = df["ct_src_dport_ltm"] / (
                df["ct_dst_sport_ltm"] + eps
            )
        if all(x in df.columns for x in ["synack", "ackdat"]):
            df["ack_error_ratio"] = np.abs(df["synack"] - df["ackdat"]) / (
                df["synack"] + df["ackdat"] + eps
            )
        return df

    def _select_features_boruta(self, x: pd.DataFrame, y: pd.Series):
        lgb = LGBMClassifier(
            n_jobs=-1,
            class_weight="balanced",
            max_depth=20,
            n_estimators=200,
            random_state=42,
            verbose=0,
        )
        boruta = BorutaPy(
            estimator=lgb,
            n_estimators="auto",
            max_iter=200,
            verbose=0,
            random_state=42,
            alpha=0.01,
        )

        # BorutaPy expects numpy arrays
        boruta.fit(x.values, y.values)
        self.final_selected_features = x.columns[boruta.support_].tolist()
        print(f"✅ {len(self.final_selected_features)} features selected by Boruta.")

    def _remove_outliers(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        detector = IsolationForest(
            contamination=0.01, n_estimators=200, random_state=42
        )
        outliers = detector.fit_predict(x) == -1

        # Only remove outliers from the 'normal' class
        normal_label = (
            0 if self.mode == "binary" else self.label_encoder.transform(["Normal"])[0]
        )
        normal_outliers = (y == normal_label) & outliers

        return x.loc[~normal_outliers], y.loc[~normal_outliers]
