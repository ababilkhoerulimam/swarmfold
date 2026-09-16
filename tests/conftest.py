import pytest


@pytest.fixture(autouse=True)
def isolate_planner_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARMFOLD_PLANNER", "mock")
    monkeypatch.delenv("SWARMFOLD_MODEL", raising=False)
