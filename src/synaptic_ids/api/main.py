import logging
from typing import List
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from mlflow import MlflowException
from sqlalchemy.orm import Session

from src.synaptic_ids.config import settings
from . import crud, models, schemas
from .database import SessionLocal, engine

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML model from MLflow URI: %s", settings.api.mlflow_model_uri)
    try:
        ml_models["ids_model"] = mlflow.pyfunc.load_model(
            model_uri=settings.api.mlflow_model_uri
        )
        logger.info("ML model loaded successfully and is available.")
    except MlflowException as e:
        logger.error("Failed to load ML model on startup: %s", e, exc_info=True)
        ml_models["ids_model"] = None

    yield
    ml_models.clear()
    logger.info("ML model cleared.")


# Ensure database tables are created based on SQLAlchemy models
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SynapticIDS API",
    description="API for the Synaptic Intrusion Detection System.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_db():
    """
    FastAPI dependency to manage database sessions per request.
    Ensures that the session is always closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", tags=["Status"])
def read_root():
    """Health check endpoint to confirm the API is running."""
    return {"message": "SynapticIDS API is active and running"}


@app.post("/predict/", response_model=schemas.PredictionResponse, tags=["Predictions"])
def predict(
    *, prediction_input: schemas.PredictionInput, db: Session = Depends(get_db)
):
    """
    Receives traffic records, performs a prediction, and stores the transaction.
    """
    model = ml_models.get("ids_model")
    if not model:
        logger.error("Prediction attempted while the ML model is unavailable.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="The ML model is not available. Please check server logs.",
        )
    input_df = pd.DataFrame(
        [record.model_dump() for record in prediction_input.records]
    )

    for col in input_df.select_dtypes(include=["float64"]).columns:
        input_df[col] = input_df[col].astype("float32")
    for col in input_df.select_dtypes(include=["int64"]).columns:
        input_df[col] = input_df[col].astype("int32")
    try:
        predictions = model.predict(input_df)
    except Exception as e:
        logger.error("An error occurred during model prediction: %s", e, exc_info=True)
        raise HTTPException(
            status_code=400,  # Bad Request
            detail=f"An error occurred during model prediction: {e}",
        ) from e

    if settings.api.model_mode == "binary":
        index_to_label = {0: "Normal", 1: "Attack"}
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
        index_to_label = {v: k for k, v in label_mapping.items()}
    else:
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail=f"Invalid model_mode configured in settings: {settings.api.model_mode}",
        )
    results = []
    for index, record in enumerate(prediction_input.records):
        prediction_output = predictions[index]
        if settings.api.model_mode == "binary":
            # For binary, output is a single probability for the positive class.
            confidence = float(prediction_output[0])
            prediction_value = 1 if confidence > 0.5 else 0
            prediction_label = index_to_label[prediction_value]
        else:  # This handles the 'multiclass' case
            # For multiclass, output is an array of probabilities.
            prediction_value = int(prediction_output.argmax())
            confidence = float(prediction_output[prediction_value])
            prediction_label = index_to_label.get(prediction_value, "Unknown")

        prediction_result = schemas.PredictionResult(
            label=prediction_label, prediction=prediction_value, confidence=confidence
        )

        crud.create_prediction_with_record(
            db=db, record_in=record, result_in=prediction_result
        )
        results.append(prediction_result)
    logger.info("%d predictions were successfully processed and stored.", len(results))
    return schemas.PredictionResponse(predictions=results)


@app.get(
    "/predictions/", response_model=List[schemas.PredictionResult], tags=["Predictions"]
)
def read_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves a paginated list of all stored predictions from the database.
    """
    predictions = crud.get_predictions(db, skip=skip, limit=limit)
    return predictions


@app.get(
    "/predictions/{prediction_id}",
    response_model=schemas.PredictionResult,
    tags=["Predictions"],
)
def read_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single prediction by its unique ID.
    """
    db_prediction = crud.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction
