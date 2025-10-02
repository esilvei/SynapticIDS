import logging
from typing import List

import pandas as pd
import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.synaptic_ids.api import crud, schemas
from src.synaptic_ids.config import settings

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model):
        self.model = model
        self._initialize_label_mapping()

    def _initialize_label_mapping(self):
        if settings.api.model_mode == "binary":
            self.index_to_label = {0: "Normal", 1: "Attack"}
            self.label_to_index = {v: k for k, v in self.index_to_label.items()}
        elif settings.api.model_mode == "multiclass":
            label_mapping = {
                "Normal": 0,
                "Analysis": 1,
                "Backdoor": 2,
                "DoS": 3,
                "Exploits": 4,
                "Fuzzers": 5,
                "Generic": 6,
                "Reconnaissance": 7,
                "Shellcode": 8,
                "Worms": 9,
            }
            self.index_to_label = {v: k for k, v in label_mapping.items()}
            self.label_to_index = label_mapping
        else:
            raise ValueError(f"Invalid model_mode: {settings.api.model_mode}")

    def _prepare_input_data(self, records: List[schemas.TrafficRecord]) -> pd.DataFrame:
        input_df = pd.DataFrame([record.model_dump() for record in records])
        for col in input_df.select_dtypes(include=["float64"]).columns:
            input_df[col] = input_df[col].astype("float32")
        for col in input_df.select_dtypes(include=["int64"]).columns:
            input_df[col] = input_df[col].astype("int32")
        return input_df

    def _process_prediction(self, prediction_output):
        if settings.api.model_mode == "binary":
            prob_attack = float(prediction_output[0])
            prob_normal = 1.0 - prob_attack

            probabilities = {"Normal": prob_normal, "Attack": prob_attack}

            confidence = float(prediction_output[0])
            prediction_value = 1 if confidence > 0.5 else 0
            label = self.index_to_label[prediction_value]
        else:  # multiclass
            probabilities = {
                self.index_to_label.get(i, "Unknown"): float(prob)
                for i, prob in enumerate(prediction_output)
            }

            prediction_value = int(np.argmax(prediction_output))
            confidence = float(prediction_output[prediction_value])
            label = self.index_to_label.get(prediction_value, "Unknown")
        return schemas.PredictionResult(
            label=label,
            prediction=prediction_value,
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_and_store(
        self,
        db: Session,
        prediction_input: schemas.PredictionInput,
    ) -> List[schemas.PredictionResult]:
        if not self.model:
            logger.error("ML model is not available.")
            raise HTTPException(status_code=503, detail="ML model is not available.")

        input_df = self._prepare_input_data(prediction_input.records)

        try:
            predictions = self.model.predict(input_df)
        except Exception as e:
            logger.error("Error during model prediction: %s", e, exc_info=True)
            raise HTTPException(
                status_code=400, detail=f"Error during model prediction: {e}"
            ) from e
        results = []
        for i, record in enumerate(prediction_input.records):
            prediction_result = self._process_prediction(predictions[i])
            crud.create_prediction_with_record(
                db, record_in=record, result_in=prediction_result
            )
            results.append(prediction_result)
        logger.info("%d predictions processed and stored.", len(results))
        return results
