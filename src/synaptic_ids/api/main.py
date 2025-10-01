import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from mlflow import MlflowException

from src.synaptic_ids.config import settings
from src.synaptic_ids.training.model.transformer_fusion import TransformerFusion  # noqa: F401
from src.synaptic_ids.training.lr_scheduler import OneCycleLR  # noqa: F401
from . import models
from .database import engine
from .routers import predictions
from .app_state import ml_models

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

app.include_router(predictions.router)


@app.get("/", tags=["Status"])
def read_root():
    """Health check endpoint to confirm the API is running."""
    return {"message": "SynapticIDS API is active and running"}
