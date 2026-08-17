"""Shared pytest fixtures. This project's src/ modules are plain scripts
(bare `from foo import bar` between siblings, each inserting its own
sibling dirs into sys.path when imported directly -- see e.g.
src/app/state.py's own sys.path.insert calls) rather than an installable
package, so tests need the same sys.path setup to import them directly.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ["models", "explainability", "fairness", "drift", "features", "app"]:
    p = str(ROOT / "src" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def api_client():
    """Builds the FastAPI app's AppState exactly once for the whole test
    session (it loads the persisted model, the full feature table, and a
    SHAP explainer -- expensive to redo per test)."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def app_state(api_client):
    """The real AppState instance backing the running test API -- lets
    tests exercise production code paths (e.g. state.compute_weighted_shap)
    directly, without rebuilding the ~15s startup a second time."""
    return api_client.app.state.app_state
