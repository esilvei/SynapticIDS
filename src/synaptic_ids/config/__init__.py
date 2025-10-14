import os
from pathlib import Path
from types import SimpleNamespace
import yaml


def _deep_namespace(data: dict) -> SimpleNamespace:
    """
    Recursively converts a nested dictionary into a nested SimpleNamespace,
    allowing for dot notation access to keys (e.g., settings.training.epochs).
    """
    if not isinstance(data, dict):
        return data

    # Create a namespace for the current level
    namespace = SimpleNamespace()

    # Recursively convert nested dictionaries
    for key, value in data.items():
        setattr(namespace, key, _deep_namespace(value))

    return namespace


def load_settings() -> SimpleNamespace:
    """
    Loads the defaults.yaml configuration file, validates its existence,
    and returns it as a nested SimpleNamespace object.

    This function is the core of the configuration loader.
    """
    config_path = Path(__file__).parent / "defaults.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    database_url_from_env = os.getenv("DATABASE_URL")
    if database_url_from_env:
        config_data["api"]["database_url"] = database_url_from_env
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if "paths" not in config_data:
        config_data["paths"] = {}
    config_data["paths"]["root"] = str(project_root)
    settings_namespace = _deep_namespace(config_data)
    settings_namespace.paths.raw_data = str(
        project_root / settings_namespace.paths.raw_data
    )
    settings_namespace.paths.processed_data = str(
        project_root / settings_namespace.paths.processed_data
    )
    settings_namespace.paths.model_save = str(
        project_root / settings_namespace.paths.model_save
    )
    settings_namespace.paths.requirements = str(
        project_root / settings_namespace.paths.requirements
    )
    return settings_namespace


# --- THE SINGLETON INSTANCE ---
settings = load_settings()
