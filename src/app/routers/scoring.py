"""Applicant scoring. Scores an EXISTING applicant by SK_ID_CURR, looked
up from `modeling_feature_set` -- not an arbitrary raw-JSON feature
payload. This is a deliberate scope decision, not a shortcut: this
project's features are built by a multi-table SQL aggregation pipeline
(bureau/previous_application/installments/credit_card/POS history --
sql/01-07_*.sql) that takes ~25 minutes over the full dataset and cannot
run per-request. A dashboard demo realistically works the same way a
credit-risk analyst would: look up/browse existing applicants, not type
in a live loan application by hand. Scoring genuinely-new applications
would need a separate, much smaller "single-applicant feature builder"
that re-implements the SQL pipeline's per-applicant logic in Python --
out of scope for this skeleton.
"""

from fastapi import APIRouter, Depends, HTTPException

from schemas import ApplicantScore
from state import AppState, get_state

router = APIRouter(prefix="/applicants", tags=["scoring"])


def _row_to_model_input(state: AppState, sk_id_curr: int):
    if sk_id_curr not in state.applicant_features.index:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} not found")
    row = state.applicant_features.loc[[sk_id_curr]].drop(columns=["TARGET"])
    return row


@router.get("", summary="List a page of known applicant IDs (for picking one to score/explain)")
def list_applicants(state: AppState = Depends(get_state), limit: int = 50, offset: int = 0):
    page = state.applicant_features.iloc[offset: offset + limit]
    return {
        "total": len(state.applicant_features),
        "limit": limit,
        "offset": offset,
        "sk_id_curr": page.index.tolist(),
    }


@router.get("/{sk_id_curr}/score", response_model=ApplicantScore)
def score_applicant(sk_id_curr: int, state: AppState = Depends(get_state)):
    row = _row_to_model_input(state, sk_id_curr)
    X_t = state.preprocessor.transform(row)

    pd_raw = float(state.model.predict_proba(X_t)[:, 1][0])
    pd_calibrated = float(state.calibrator.predict([pd_raw])[0])
    approved = pd_calibrated <= state.threshold

    if sk_id_curr in state.applicant_demo.index:
        ead = float(state.applicant_demo.loc[sk_id_curr, "EAD"])
    else:
        ead = float("nan")
    expected_loss = pd_calibrated * state.lgd * ead

    return ApplicantScore(
        sk_id_curr=sk_id_curr,
        pd_raw=pd_raw,
        pd_calibrated=pd_calibrated,
        threshold=state.threshold,
        approved=approved,
        expected_loss=expected_loss,
    )
