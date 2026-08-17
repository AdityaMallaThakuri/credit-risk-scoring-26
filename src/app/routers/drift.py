"""Serves Phase 5's precomputed PSI/KS/JS drift metrics as-is (see
state.py's docstring on why these aren't recomputed per request)."""

from fastapi import APIRouter, Depends, Query

from schemas import DriftRow
from state import AppState, get_state

router = APIRouter(prefix="/drift", tags=["drift"])


@router.get("", response_model=list[DriftRow])
def get_drift(
    state: AppState = Depends(get_state),
    period: str | None = Query(default=None),
    feature: str | None = Query(default=None),
):
    df = state.drift
    if period is not None:
        df = df[df["period"] == period]
    if feature is not None:
        df = df[df["feature"] == feature]
    return df[["period", "feature", "psi", "js_divergence", "ks_statistic", "ks_pvalue"]].to_dict(orient="records")
