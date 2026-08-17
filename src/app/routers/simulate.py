"""Policy-simulator endpoints. `/simulate/threshold` recomputes profit for
an arbitrary threshold on demand (cheap -- one pass over the precomputed
per-applicant PD/TARGET/EAD population, via state.simulate_profit, which
reuses expected_loss.py's profit_at_threshold directly). `/simulate/profit-curve`
just serves the already-computed Phase 3 curve (same as-is convention as
the fairness/drift routers) for the dashboard to plot without re-deriving it.
"""

import pandas as pd
from fastapi import APIRouter, Depends

from schemas import ProfitSimulation
from state import DATA_DIR, SIMULATE_MODEL_NAME, AppState, get_state

router = APIRouter(prefix="/simulate", tags=["simulate"])

PROFIT_CURVE_PATH = DATA_DIR / "phase3_profit_curves.csv"


@router.get("/threshold", response_model=ProfitSimulation)
def simulate_threshold(threshold: float, state: AppState = Depends(get_state)):
    return state.simulate_profit(threshold)


@router.get("/profit-curve", summary="Precomputed Phase 3 profit-vs-threshold curve for the deployed model")
def profit_curve():
    df = pd.read_csv(PROFIT_CURVE_PATH)
    df = df[df["model"] == SIMULATE_MODEL_NAME]
    return df[["threshold", "profit", "approval_rate", "approved_default_rate"]].to_dict(orient="records")
