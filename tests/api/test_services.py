from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException

from src.synaptic_ids.api import schemas, services
from src.synaptic_ids.config import settings
from src.synaptic_ids.api.schemas import TrafficRecord

TRAFFIC_RECORD_EXAMPLE = TrafficRecord.model_config["json_schema_extra"]["example"]


@pytest.fixture
def mock_db_session():
    """Provides a mock of the database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_redis_client():
    """Provides a mock of the redis client."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_prediction_service_binary_mode(
    mock_db_session: MagicMock, mock_redis_client: AsyncMock
):
    """Test the prediction service in binary classification mode."""
    settings.api.model_mode = "binary"
    mock_model = MagicMock()
    mock_pipeline = MagicMock()
    mock_data_preparer = MagicMock()
    mock_data_preparer.prepare_data = AsyncMock(
        return_value={"images": [], "sequences": []}
    )
    mock_pipeline.data_preparer = mock_data_preparer
    mock_pipeline.model.predict.return_value = [[0.8], [0.4]]  # Attack, Normal
    mock_model.unwrap_python_model.return_value = mock_pipeline

    service = services.PredictionService(mock_model)
    prediction_input = schemas.PredictionInput(
        session_id="test_binary",
        records=[
            schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE),
            schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE),
        ],
    )
    with patch(
        "src.synaptic_ids.api.crud.create_prediction_with_record"
    ) as mock_create:
        results = await service.predict_and_store(
            mock_db_session, prediction_input, mock_redis_client
        )

    assert len(results) == 2
    assert results[0].label == "Attack"
    assert results[0].prediction == 1
    assert results[0].confidence == 0.8

    assert results[1].label == "Normal"
    assert results[1].prediction == 0
    assert results[1].confidence == 0.4

    assert mock_create.call_count == 2
    mock_pipeline.model.predict.assert_called_once()


@pytest.mark.asyncio
async def test_prediction_service_multiclass_mode(
    mock_db_session: MagicMock, mock_redis_client: AsyncMock
):
    """Test the prediction service in multiclass classification mode."""
    settings.api.model_mode = "multiclass"
    mock_model = MagicMock()
    mock_pipeline = MagicMock()
    mock_data_preparer = MagicMock()
    mock_data_preparer.prepare_data = AsyncMock(
        return_value={"images": [], "sequences": []}
    )
    mock_pipeline.data_preparer = mock_data_preparer
    # Mock output for two records: (DoS, Fuzzers)
    mock_pipeline.model.predict.return_value = [
        [0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 0.1, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0],
    ]
    mock_model.unwrap_python_model.return_value = mock_pipeline

    service = services.PredictionService(mock_model)
    prediction_input = schemas.PredictionInput(
        session_id="test_multiclass",
        records=[
            schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE),
            schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE),
        ],
    )

    with patch(
        "src.synaptic_ids.api.crud.create_prediction_with_record"
    ) as mock_create:
        results = await service.predict_and_store(
            mock_db_session, prediction_input, mock_redis_client
        )

    assert len(results) == 2
    assert results[0].label == "DoS"
    assert results[0].prediction == 3  # Index of DoS
    assert results[0].confidence == 0.8

    assert results[1].label == "Fuzzers"
    assert results[1].prediction == 5  # Index of Fuzzers
    assert results[1].confidence == 0.6

    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_predict_and_store_no_model():
    """Test that the service raises an exception if the model is not available."""
    service = services.PredictionService(model=None)
    prediction_input = schemas.PredictionInput(session_id="test_no_model", records=[])

    with pytest.raises(HTTPException) as excinfo:
        await service.predict_and_store(
            MagicMock(), prediction_input, mock_redis_client
        )

    assert excinfo.value.status_code == 503
    assert "ML model is not available" in excinfo.value.detail
