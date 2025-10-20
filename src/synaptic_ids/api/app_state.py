import logging
import mlflow
from mlflow import MlflowException
from src.synaptic_ids.config import settings

ml_models = {}

logger = logging.getLogger(__name__)


def load_model():
    """
    Loads the ML model from the configured MLflow URI and stores it in the
    ml_models dictionary. This function is designed to be patched out during testing.
    """
    logger.info("Loading ML model from MLflow URI: %s", settings.api.mlflow_model_uri)
    try:
        ml_models["ids_model"] = mlflow.pyfunc.load_model(
            model_uri=settings.api.mlflow_model_uri
        )
        logger.info("ML model loaded successfully.")
    except MlflowException as e:
        logger.error("Failed to load ML model: %s", e, exc_info=True)
        ml_models["ids_model"] = None
