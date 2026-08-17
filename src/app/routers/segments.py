"""Portfolio segments (roadmap 7.2 #2): breaks the portfolio down by
predicted risk tier and by demographic group. Reuses `state.segment_population`
(Phase 3's OOF calibrated PD for LightGBM + fairness_metrics.py's own
demographic/age-band joins) -- pure aggregation on top of already-computed
outputs, no new modeling.
"""

from fastapi import APIRouter, Depends, HTTPException

from schemas import DemographicSegmentRow, RiskTierRow
from state import SMALL_GROUP_WARNING_N, AppState, get_state

router = APIRouter(prefix="/segments", tags=["segments"])

DEMOGRAPHIC_ATTRIBUTES = ["CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"]


@router.get("/risk-tier", response_model=list[RiskTierRow])
def risk_tier_segments(state: AppState = Depends(get_state)):
    """Note: approval_rate is deliberately NOT included here -- the tier
    boundaries are defined by the deployed decision threshold itself (see
    state.py's RISK_TIER_EDGES), so approval rate per tier is a tautological
    0%/100% split, not an informative stat. default_rate (observed TARGET
    rate) is the real signal, and it should rise monotonically with tier by
    construction of a well-calibrated model -- worth checking, not assuming."""
    df = state.segment_population
    total = len(df)
    grouped = df.groupby("risk_tier", observed=True).agg(
        count=("PD", "size"),
        default_rate=("TARGET", "mean"),
        avg_pd=("PD", "mean"),
        avg_ead=("EAD", "mean"),
    )
    return [
        RiskTierRow(
            tier=tier,
            count=int(row["count"]),
            share=float(row["count"]) / total,
            default_rate=float(row["default_rate"]),
            avg_pd=float(row["avg_pd"]),
            avg_ead=float(row["avg_ead"]),
        )
        for tier, row in grouped.iterrows()
    ]


@router.get("/demographic", response_model=list[DemographicSegmentRow])
def demographic_segments(attribute: str, state: AppState = Depends(get_state)):
    """`reliable` flags groups with n < SMALL_GROUP_WARNING_N (the same
    threshold fairness_metrics.py uses to exclude a group from the
    headline DPD/EOD spread) -- a 100%/0% rate from 2 or 4 applicants is
    not a confident number, and this stops the dashboard from displaying
    it as if it were one."""
    if attribute not in DEMOGRAPHIC_ATTRIBUTES:
        raise HTTPException(status_code=400, detail=f"attribute must be one of {DEMOGRAPHIC_ATTRIBUTES}")

    df = state.segment_population
    grouped = df.groupby(attribute, observed=True).agg(
        count=("PD", "size"),
        approval_rate=("approved", "mean"),
        default_rate=("TARGET", "mean"),
    )
    return [
        DemographicSegmentRow(
            attribute=attribute,
            group=str(group),
            count=int(row["count"]),
            approval_rate=float(row["approval_rate"]),
            default_rate=float(row["default_rate"]),
            reliable=bool(row["count"] >= SMALL_GROUP_WARNING_N),
        )
        for group, row in grouped.iterrows()
    ]
