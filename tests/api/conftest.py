from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.synaptic_ids.api.database import Base, get_db
from src.synaptic_ids.api.main import app
from src.synaptic_ids.api.app_state import ml_models
from src.synaptic_ids.api.routers.predictions import get_redis_client

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """
    Create a new database session for each test function.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def mock_redis_client():
    """
    mock_redis_client
    """
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    return mock_redis


@pytest.fixture(scope="function")
def client(db_session, mock_redis_client: MagicMock):
    """
    Creates a new FastAPI TestClient that uses the `db_session` fixture to override
    the `get_db` dependency that is injected into routes.
    """

    def _get_db_override():
        yield db_session

    def _get_redis_override():
        return mock_redis_client

    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[get_redis_client] = _get_redis_override

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ml_model():
    """
    Provides a mock of the ML model for testing purposes.
    """
    mock_model = MagicMock()
    ml_models["ids_model"] = mock_model
    yield mock_model
    ml_models.clear()
