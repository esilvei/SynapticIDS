from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.synaptic_ids.api import crud, schemas, services
from src.synaptic_ids.api.database import get_db
from src.synaptic_ids.api.app_state import ml_models

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
)


def get_prediction_service() -> services.PredictionService:
    model = ml_models.get("ids_model")
    if not model:
        raise HTTPException(status_code=503, detail="ML model is not available.")
    return services.PredictionService(model)


@router.post("/", response_model=schemas.PredictionResponse)
def predict(
    prediction_input: schemas.PredictionInput,
    db: Session = Depends(get_db),
    service: services.PredictionService = Depends(get_prediction_service),  # Corrigido
):
    """
    Receives traffic records, performs a prediction, and stores the transaction.
    """
    results = service.predict_and_store(db, prediction_input)
    return schemas.PredictionResponse(predictions=results)


@router.get("/", response_model=List[schemas.PredictionRecord])
def read_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves a paginated list of all stored predictions.
    """
    return crud.get_predictions(db, skip=skip, limit=limit)


@router.get("/{prediction_id}", response_model=schemas.PredictionRecord)
def read_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single prediction by its unique ID.
    """
    db_prediction = crud.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction
