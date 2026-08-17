"""FastAPI entrypoint. Constructs AppState once at startup (lifespan),
not per-request or as a module global -- see state.py's docstring."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI  # noqa: E402

from routers import drift, explain, fairness, scoring, segments, simulate  # noqa: E402
from state import AppState  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = AppState()
    yield


app = FastAPI(title="Credit Risk FYP API", lifespan=lifespan)

app.include_router(scoring.router)
app.include_router(explain.router)
app.include_router(fairness.router)
app.include_router(segments.router)
app.include_router(drift.router)
app.include_router(simulate.router)


@app.get("/health")
def health():
    return {"status": "ok"}
