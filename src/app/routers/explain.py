"""Local SHAP explanation for one applicant, on the persisted deployed
model. Same convention as src/explainability/static_shap.py: SHAP
explains the model's raw margin output, not the isotonic-calibrated
probability (calibration has no feature-by-feature decomposition) --
additive consistency (base_value + sum(shap) == raw margin) is asserted,
not just assumed, exactly as static_shap.py's hard-rule check does.
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from routers.scoring import _row_to_model_input
from schemas import ApplicantExplanation, FeatureContribution, WeightedFeatureContribution, WeightedShapExplanation
from state import AppState, get_state

router = APIRouter(prefix="/applicants", tags=["explain"])

PERIODS = ["P0", "P1", "P2", "P3", "P4"]


def _static_shap(state: AppState, sk_id_curr: int):
    """Shared by /explanation and /weighted-explanation: the tree_path_dependent
    static SHAP values for one applicant, additive consistency asserted."""
    row = _row_to_model_input(state, sk_id_curr)
    X_t = state.preprocessor.transform(row)

    shap_values = state.explainer(X_t)
    base_value = float(shap_values.base_values[0])
    values = shap_values.values[0]

    raw_margin = float(state.model.predict(X_t, raw_score=True)[0])
    reconstructed = base_value + values.sum()
    assert np.isclose(raw_margin, reconstructed, atol=1e-4), (
        f"SHAP additive consistency violated for SK_ID_CURR={sk_id_curr}: "
        f"raw_margin={raw_margin}, base_value+sum(shap)={reconstructed}"
    )

    pd_raw = float(state.model.predict_proba(X_t)[:, 1][0])
    pd_calibrated = float(state.calibrator.predict([pd_raw])[0])
    return row, base_value, values, raw_margin, pd_calibrated


@router.get("/{sk_id_curr}/explanation", response_model=ApplicantExplanation)
def explain_applicant(sk_id_curr: int, state: AppState = Depends(get_state), top_k: int = 15):
    row, base_value, values, raw_margin, pd_calibrated = _static_shap(state, sk_id_curr)

    row_values = row.iloc[0]
    order = np.argsort(-np.abs(values))[:top_k]
    contributions = [
        FeatureContribution(
            feature=state.feature_names[i],
            value=_lookup_raw_value(row_values, state.feature_names[i]),
            shap_value=float(values[i]),
        )
        for i in order
    ]

    return ApplicantExplanation(
        sk_id_curr=sk_id_curr,
        base_value=base_value,
        raw_margin=raw_margin,
        pd_calibrated=pd_calibrated,
        contributions=contributions,
    )


@router.get("/{sk_id_curr}/weighted-explanation", response_model=WeightedShapExplanation)
def weighted_explain_applicant(
    sk_id_curr: int,
    period: str = "P3",
    alpha: float = 0.5,
    top_k: int = 15,
    state: AppState = Depends(get_state),
):
    """Phase 6's Weighted Temporal SHAP (Weighted_SHAP = SHAP x [alpha*w_drift +
    (1-alpha)*w_cost], rescaled to exact additive consistency), applied live to
    an arbitrary applicant -- reuses Phase 6's own precomputed per-period
    w_drift/w_cost (period-level, not applicant-specific) rather than
    recomputing PSI/counterfactual-cost simulation per request."""
    if period not in PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {PERIODS}")
    if not 0.0 <= alpha <= 1.0:
        raise HTTPException(status_code=400, detail="alpha must be in [0, 1]")

    row, base_value, static_values, raw_margin, pd_calibrated = _static_shap(state, sk_id_curr)
    weights = state.blend_weights_for(period, alpha)
    weighted_values = state.compute_weighted_shap(static_values, base_value, raw_margin, period, alpha)

    reconstructed = base_value + weighted_values.sum()
    assert np.isclose(raw_margin, reconstructed, atol=1e-4), (
        f"Weighted SHAP additive consistency violated for SK_ID_CURR={sk_id_curr}: "
        f"raw_margin={raw_margin}, base_value+sum(weighted_shap)={reconstructed}"
    )

    row_values = row.iloc[0]
    order = np.argsort(-np.abs(weighted_values))[:top_k]
    contributions = [
        WeightedFeatureContribution(
            feature=state.feature_names[i],
            value=_lookup_raw_value(row_values, state.feature_names[i]),
            static_shap_value=float(static_values[i]),
            weight=float(weights[i]),
            weighted_shap_value=float(weighted_values[i]),
        )
        for i in order
    ]

    return WeightedShapExplanation(
        sk_id_curr=sk_id_curr,
        period=period,
        alpha=alpha,
        base_value=base_value,
        raw_margin=raw_margin,
        pd_calibrated=pd_calibrated,
        contributions=contributions,
    )


def _lookup_raw_value(row_values, transformed_feature_name):
    """Transformed feature names are ColumnTransformer-prefixed
    (`num__dti_ratio`, `cat__prev_app_last_status_...`) -- strip the
    `num__` prefix to look up the original numeric value for display;
    one-hot categorical columns have no single scalar original value, so
    those are left as None."""
    if transformed_feature_name.startswith("num__"):
        original = transformed_feature_name[len("num__"):]
        if original in row_values.index:
            return float(row_values[original])
    return None
