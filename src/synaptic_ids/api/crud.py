from typing import List, Optional
from sqlalchemy.orm import Session
from . import models, schemas


def create_prediction_with_record(
    db: Session,
    *,
    record_in: schemas.TrafficRecord,
    result_in: schemas.PredictionResult,
) -> models.Prediction:
    """
    Creates a new Prediction and its associated TrafficRecord in the database.

    This function performs a single, atomic transaction to ensure that every
    prediction is stored with the exact input data that generated it.

    Args:
        db: The SQLAlchemy database session, injected by FastAPI.
        record_in: The Pydantic schema containing the input traffic features.
        result_in: The Pydantic schema containing the model's prediction output.

    Returns:
        The created SQLAlchemy Prediction object now contains a database ID
        and other generated values.
    """
    db_traffic_record = models.TrafficRecord(**record_in.model_dump())

    db_prediction = models.Prediction(
        label=result_in.label,
        prediction=result_in.prediction,
        confidence=result_in.confidence,
        traffic_record=db_traffic_record,
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return db_prediction


def get_prediction(db: Session, *, prediction_id: int) -> Optional[models.Prediction]:
    """
    Retrieves a single prediction from the database by its primary key.

    Args:
        db: The SQLAlchemy database session.
        prediction_id: The unique ID of the prediction to retrieve.

    Returns:
        The SQLAlchemy Prediction object if found, otherwise None.
    """
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.id == prediction_id)
        .first()
    )


def get_predictions(
    db: Session, *, skip: int = 0, limit: int = 100
) -> List[models.Prediction]:
    """
    Retrieves a list of predictions from the database with pagination.

    Args:
        db: The SQLAlchemy database session.
        skip: The number of records to skip (for pagination).
        limit: The maximum number of records to return.

    Returns:
        A list of SQLAlchemy Prediction objects.
    """
    return db.query(models.Prediction).offset(skip).limit(limit).all()  # type: ignore
