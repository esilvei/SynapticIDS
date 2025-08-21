from typing import List


def get_api_dependencies() -> List[str]:
    """
    Returns a list of the main application and API dependencies.

    These are the essential libraries for data preprocessing, model training,
    inference, and serving the API.

    Returns:
        List[str]: A list of strings, where each string is a dependency.
    """
    return [
        # Core for ML and Data Handling
        "pandas>=2.3.1",
        "numpy>=2.1.3",
        "scikit-learn>=1.7.1",
        "tensorflow>=2.19.0",
        "keras>=3.11.1",
        "h5py>=3.14.0",
        "scipy>=1.16.1",
        "seaborn>=0.13.2",
        "matplotlib>=3.10.5",
        "boruta>=0.4.3",
        "lightgbm>=4.6.0",
        # API Framework and Server
        "fastapi>=0.116.1",
        "uvicorn>=0.35.0",
        "python-multipart>=0.0.20",
        "gunicorn>=23.0.0",
        "httptools>=0.6.4",
        "uvloop>=0.17.0",
        "watchfiles>=1.1.0",
        "websockets>=15.0.1",
        # ML Operations (MLOps)
        "mlflow>=3.2.0",
        "mlflow-skinny>=3.2.0",
        # File and Data Handling
        "fastparquet>=2024.11.0",
        "pyarrow>=21.0.0",
        "kaggle>=1.7.4.5",
        "kagglehub>=0.3.12",
        # Utilities and Others
        "pydantic>=2.11.7",
        "python-dotenv>=1.1.1",
        "tqdm>=4.67.1",
        "joblib>=1.5.1",
        "pyyaml>=6.0.2",
        "requests>=2.32.4",
        "sqlalchemy>=1.4.54",
    ]


def get_development_dependencies() -> List[str]:
    """
    Returns a list of development dependencies.

    These are libraries for testing, linting, code formatting, and
    pre-commit hook automation.

    Returns:
        List[str]: A list of strings for development dependencies.
    """
    return [
        "pytest>=8.4.1",
        "pylint>=3.3.8",
        "black>=23.0",
        "isort>=6.0.1",
        "pre-commit>=4.3.0",
        "mypy>=1.17.1",
        "ruff>=0.12.8",
        "apache-airflow>=3.0.4",
    ]
