"""Serves Phase 4/4-bias precomputed fairness results as-is (see state.py's
docstring on why these aren't recomputed per request)."""

from fastapi import APIRouter, Depends, HTTPException, Query

from schemas import FairnessRealRow, FairnessSyntheticRow, IsolatedBiasEffectRow, LiveFairness, TradeoffRow
from state import AppState, get_state

router = APIRouter(prefix="/fairness", tags=["fairness"])

LIVE_ATTRIBUTES = ["CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"]


@router.get("/by-period", response_model=list[FairnessSyntheticRow])
def fairness_by_period(state: AppState = Depends(get_state), period: str | None = Query(default=None)):
    """CAUTION -- this is the naive cross-period comparison (P0 vs P1-P4),
    which src/fairness/synthetic_bias_detection.py's own docstring says is
    confounded by each period's independent covariate-drift resampling and
    looks noisy/non-monotonic even though the planted bias is real and
    detectable. Context only -- use /fairness/isolated-effect for the
    actual, confound-free finding."""
    df = state.fairness_synthetic
    if period is not None:
        df = df[df["period"] == period]
    return df.to_dict(orient="records")


@router.get("/isolated-effect", response_model=list[IsolatedBiasEffectRow])
def fairness_isolated_effect(state: AppState = Depends(get_state), period: str | None = Query(default=None)):
    """The confound-free Phase 4 result: same population, same model
    decisions, per biased period (P1-P4; P0 has no planted bias so isn't
    included) -- fairness metrics computed against TARGET_original
    (pre-flip) vs TARGET (post-flip). F_FPR_inflation is the clean signal:
    consistently positive across all 4 periods."""
    df = state.fairness_isolated_effect
    if period is not None:
        df = df[df["period"] == period]
    return df.to_dict(orient="records")


@router.get("/by-attribute", response_model=list[FairnessRealRow])
def fairness_by_attribute(state: AppState = Depends(get_state), attribute: str | None = Query(default=None)):
    df = state.fairness_real
    if attribute is not None:
        df = df[df["attribute"] == attribute]
    return df.to_dict(orient="records")


@router.get("/live", response_model=LiveFairness, summary="Recompute DPD/EOD/EqualizedOddsDiff at an arbitrary threshold -- for the policy simulator's slider")
def fairness_live(threshold: float, attribute: str = "age_band", state: AppState = Depends(get_state)):
    if attribute not in LIVE_ATTRIBUTES:
        raise HTTPException(status_code=400, detail=f"attribute must be one of {LIVE_ATTRIBUTES}")
    summary, excluded_groups = state.compute_live_fairness(threshold, attribute)
    return LiveFairness(
        attribute=attribute,
        threshold=threshold,
        DPD=summary["DPD"],
        EOD=summary["EOD"],
        EqualizedOddsDiff=summary["EqualizedOddsDiff"],
        excluded_groups=excluded_groups,
    )


@router.get("/tradeoff", response_model=list[TradeoffRow])
def fairness_tradeoff(attribute: str, state: AppState = Depends(get_state)):
    """Phase 4's fairness-accuracy trade-off table (src/fairness/fairness_accuracy_tradeoff.py):
    baseline vs. two targeted mitigation strategies vs. one naive blunt
    strategy, on the attribute with the largest real disparity (age_band;
    CODE_GENDER kept too even though its real disparity was too small to
    be illustrative, per that module's own docstring)."""
    if attribute not in state.fairness_tradeoff:
        raise HTTPException(status_code=400, detail=f"attribute must be one of {list(state.fairness_tradeoff)}")
    return state.fairness_tradeoff[attribute].to_dict(orient="records")
