from sqlalchemy.orm import Session
from src.synaptic_ids.api import crud, schemas

TRAFFIC_RECORD_EXAMPLE = {
    "proto": "tcp",
    "state": "FIN",
    "dur": 0.000001,
    "sbytes": 100,
    "dbytes": 200,
    "sttl": 254,
    "dttl": 252,
    "sloss": 0,
    "dloss": 0,
    "service": "http",
    "sload": 80000000.0,
    "dload": 160000000.0,
    "spkts": 1,
    "dpkts": 1,
    "smean": 100,
    "dmean": 200,
    "sjit": 0.0,
    "djit": 0.0,
    "stime": 1421927414,
    "ltime": 1421927414,
    "sinpkt": 0.0,
    "dinpkt": 0.0,
    "is_sm_ips_ports": 0,
    "ct_srv_src": 1,
    "ct_srv_dst": 1,
    "ct_dst_ltm": 1,
    "ct_src_ltm": 1,
    "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1,
    "ct_dst_src_ltm": 1,
    "rate": 1000000.0,
    "response_body_len": 0,
    "tcprtt": 0.0,
    "synack": 0.0,
    "ackdat": 0.0,
}


def test_create_prediction_with_record(db_session: Session):
    """Test creating a prediction and its associated traffic record."""
    record_in = schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE)
    result_in = schemas.PredictionResult(
        label="Attack",
        prediction=1,
        confidence=0.9,
        probabilities={"Attack": 0.9, "Normal": 0.1},
    )

    db_prediction = crud.create_prediction_with_record(
        db=db_session, record_in=record_in, result_in=result_in
    )

    assert db_prediction.id is not None
    assert db_prediction.label == "Attack"
    assert db_prediction.confidence == 0.9
    assert db_prediction.traffic_record is not None
    assert db_prediction.traffic_record.proto == "tcp"
    assert db_prediction.traffic_record.sbytes == 100


def test_get_prediction(db_session: Session):
    """Test retrieving a single prediction by ID."""
    record_in = schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE)
    result_in = schemas.PredictionResult(label="Normal", prediction=0, confidence=0.8)

    created_prediction = crud.create_prediction_with_record(
        db=db_session, record_in=record_in, result_in=result_in
    )

    retrieved_prediction = crud.get_prediction(
        db=db_session, prediction_id=created_prediction.id
    )

    assert retrieved_prediction is not None
    assert retrieved_prediction.id == created_prediction.id
    assert retrieved_prediction.label == "Normal"


def test_get_prediction_not_found(db_session: Session):
    """Test retrieving a non-existent prediction."""
    retrieved_prediction = crud.get_prediction(db=db_session, prediction_id=999)
    assert retrieved_prediction is None


def test_get_predictions(db_session: Session):
    """Test retrieving a list of predictions with pagination."""
    record_in = schemas.TrafficRecord(**TRAFFIC_RECORD_EXAMPLE)
    result_in_1 = schemas.PredictionResult(label="Attack", prediction=1)
    result_in_2 = schemas.PredictionResult(label="Normal", prediction=0)

    crud.create_prediction_with_record(
        db=db_session, record_in=record_in, result_in=result_in_1
    )
    crud.create_prediction_with_record(
        db=db_session, record_in=record_in, result_in=result_in_2
    )

    # Test retrieving all
    predictions = crud.get_predictions(db=db_session, skip=0, limit=10)
    assert len(predictions) == 2

    # Test pagination (skip 1)
    predictions_skipped = crud.get_predictions(db=db_session, skip=1, limit=10)
    assert len(predictions_skipped) == 1
    assert predictions_skipped[0].label == "Normal"

    # Test pagination (limit 1)
    predictions_limited = crud.get_predictions(db=db_session, skip=0, limit=1)
    assert len(predictions_limited) == 1
    assert predictions_limited[0].label == "Attack"
