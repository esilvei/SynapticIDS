from typing import List
from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import pandas as pd
from src.synaptic_ids.api.app_state import ml_models
from src.synaptic_ids.api.schemas import TrafficRecord

TRAFFIC_RECORD_EXAMPLE = TrafficRecord.model_config["json_schema_extra"]["example"]


def test_read_root(client: TestClient):
    """
    Tests the API's health check endpoint.
    Verifies that the API is active and responding correctly.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SynapticIDS API is active and running"}


def create_mock_prediction(
    winner_index: int, num_classes: int = 10
) -> List[List[float]]:
    """Creates a multiclass mock prediction where the winner_index has the highest probability."""
    prediction = [0.0] * num_classes
    prediction[winner_index] = 0.9
    remaining_prob = 0.1 / (num_classes - 1)
    for i in range(num_classes):
        if i != winner_index:
            prediction[i] = remaining_prob
    return [prediction]


@pytest.mark.parametrize(
    "model_mode, mock_prediction, expected_label",
    [
        # --- Multiclass Scenarios ---
        ("multiclass", create_mock_prediction(winner_index=0), "Normal"),
        ("multiclass", create_mock_prediction(winner_index=1), "Analysis"),
        ("multiclass", create_mock_prediction(winner_index=3), "DoS"),
        ("multiclass", create_mock_prediction(winner_index=9), "Worms"),
        # --- Binary Scenarios ---
        ("binary", [[0.9]], "Attack"),
        ("binary", [[0.1]], "Normal"),
    ],
)
def test_predict_success(
    client: TestClient,
    db_session: Session,
    mock_ml_model: MagicMock,
    monkeypatch,
    model_mode,
    mock_prediction,
    expected_label,
):
    """Tests successful prediction and storage
    for binary and multiclass modes with various labels."""
    monkeypatch.setattr("src.synaptic_ids.config.settings.api.model_mode", model_mode)
    mock_data_preparer = MagicMock()
    mock_data_preparer.prepare_data = AsyncMock(
        return_value={
            "images": pd.DataFrame([TRAFFIC_RECORD_EXAMPLE]),
            "sequences": pd.DataFrame([TRAFFIC_RECORD_EXAMPLE]),
        }
    )
    mock_pipeline = MagicMock()
    mock_pipeline.data_preparer = mock_data_preparer
    mock_pipeline.model.predict.return_value = mock_prediction
    mock_ml_model.unwrap_python_model.return_value = mock_pipeline
    payload = {"session_id": "test_session", "records": [TRAFFIC_RECORD_EXAMPLE]}

    response = client.post("/predictions/", json=payload)
    assert response.status_code == 200, response.json()

    response_data = response.json()["predictions"]
    assert len(response_data) == 1
    assert response_data[0]["label"] == expected_label

    get_response = client.get("/predictions/")
    assert get_response.status_code == 200
    retrieved_data = get_response.json()
    assert len(retrieved_data) >= 1
    assert retrieved_data[-1]["label"] == expected_label


def test_predict_no_model(client: TestClient):
    """Tests the prediction endpoint when the ML model is unavailable."""
    if "ids_model" in ml_models:
        del ml_models["ids_model"]
    response = client.post(
        "/predictions/",
        json={"session_id": "test_no_model", "records": [TRAFFIC_RECORD_EXAMPLE]},
    )
    assert response.status_code == 503


def test_predict_invalid_payload(client: TestClient, mock_ml_model: MagicMock):
    """Tests prediction with a payload missing a required field."""
    invalid_payload = TRAFFIC_RECORD_EXAMPLE.copy()
    del invalid_payload["proto"]
    response = client.post(
        "/predictions/",
        json={"session_id": "test_session", "records": [invalid_payload]},
    )
    assert response.status_code == 422


def test_read_prediction_not_found(client: TestClient, db_session: Session):
    """Tests retrieving a prediction with an ID that does not exist."""
    response = client.get("/predictions/99999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found"}
