import pytest
from _pytest.monkeypatch import MonkeyPatch
from src.synaptic_ids.api import app_state


@pytest.fixture(scope="session", autouse=True)
def disable_model_loading_on_startup(monkeypatch_session):
    monkeypatch_session.setattr(app_state, "load_model", lambda: None)


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = MonkeyPatch()
    yield mp
    mp.undo()
