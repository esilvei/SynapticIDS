import logging
import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
import redis.asyncio as redis
from redis.exceptions import RedisError

from src.synaptic_ids.training.model.transformer_fusion import TransformerFusion  # noqa: F401
from src.synaptic_ids.training.lr_scheduler import OneCycleLR  # noqa: F401
from src.synaptic_ids.api import models
from src.synaptic_ids.api.database import engine
from src.synaptic_ids.api.routers import predictions
from src.synaptic_ids.api.app_state import ml_models, load_model

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Connecting to Redis at %s...", REDIS_URL)
    try:
        app.state.redis = await redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True, protocol=3
        )
        logger.info("Connected to Redis successfully.")
    except RedisError as e:
        logger.error("Failed to connect to Redis: %s", e, exc_info=True)
        app.state.redis = None

    yield
    ml_models.clear()
    logger.info("ML model cleared.")
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.aclose()
        logger.info("Redis connection closed.")


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
