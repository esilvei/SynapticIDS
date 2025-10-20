from pathlib import Path


def get_project_root() -> Path:
    """
    Finds the project root by searching for the 'pyproject.toml' file.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Project root not found. Make sure 'pyproject.toml' exists."
    )
