from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.synaptic_ids.config import settings

DATABASE_URL = settings.api.database_url
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


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
