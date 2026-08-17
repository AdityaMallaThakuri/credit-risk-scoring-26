"""Phase 8.2: basic functional tests for every backend endpoint -- does it
return the right shape of data, and does it handle a missing/edge-case
applicant or an invalid filter gracefully (not a 500)."""

import numpy as np
import pytest

KNOWN_SK_ID_CURR = 100002
UNKNOWN_SK_ID_CURR = 999_999_999


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- scoring -----------------------------------------------------------


def test_list_applicants(api_client):
    r = api_client.get("/applicants", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 300_000
    assert len(body["sk_id_curr"]) == 5


def test_score_known_applicant(api_client):
    r = api_client.get(f"/applicants/{KNOWN_SK_ID_CURR}/score")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["pd_calibrated"] <= 1.0
    assert body["threshold"] == pytest.approx(0.220, abs=1e-6)
    assert body["approved"] == (body["pd_calibrated"] <= body["threshold"])
    assert body["expected_loss"] >= 0


def test_score_unknown_applicant_is_404(api_client):
    r = api_client.get(f"/applicants/{UNKNOWN_SK_ID_CURR}/score")
    assert r.status_code == 404


# --- explanation ---------------------------------------------------------


def test_explanation_known_applicant(api_client):
    r = api_client.get(f"/applicants/{KNOWN_SK_ID_CURR}/explanation", params={"top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) == 5


def test_explanation_full_vector_is_additive_consistent(api_client):
    """Requesting effectively all features and summing them should
    reconstruct the raw margin -- the same hard rule checked server-side,
    verified again here from the API's own response."""
    r = api_client.get(f"/applicants/{KNOWN_SK_ID_CURR}/explanation", params={"top_k": 200})
    body = r.json()
    total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert total == pytest.approx(body["raw_margin"], abs=1e-4)


def test_explanation_unknown_applicant_is_404(api_client):
    r = api_client.get(f"/applicants/{UNKNOWN_SK_ID_CURR}/explanation")
    assert r.status_code == 404


def test_weighted_explanation(api_client):
    r = api_client.get(f"/applicants/{KNOWN_SK_ID_CURR}/weighted-explanation", params={"period": "P3", "alpha": 0.5, "top_k": 200})
    assert r.status_code == 200
    body = r.json()
    total = body["base_value"] + sum(c["weighted_shap_value"] for c in body["contributions"])
    assert total == pytest.approx(body["raw_margin"], abs=1e-4)


@pytest.mark.parametrize("bad_params", [{"period": "NOT_A_PERIOD"}, {"alpha": 1.5}, {"alpha": -0.1}])
def test_weighted_explanation_rejects_invalid_params(api_client, bad_params):
    params = {"period": "P3", "alpha": 0.5, **bad_params}
    r = api_client.get(f"/applicants/{KNOWN_SK_ID_CURR}/weighted-explanation", params=params)
    assert r.status_code == 400


# --- segments --------------------------------------------------------------


def test_segments_risk_tier(api_client):
    r = api_client.get("/segments/risk-tier")
    assert r.status_code == 200
    rows = {row["tier"]: row for row in r.json()}
    assert set(rows) == {"Low", "Medium", "High"}
    assert np.isclose(sum(row["share"] for row in rows.values()), 1.0, atol=1e-6)
    # a well-calibrated model's default rate should rise with risk tier
    assert rows["Low"]["default_rate"] < rows["Medium"]["default_rate"] < rows["High"]["default_rate"]


def test_segments_demographic(api_client):
    r = api_client.get("/segments/demographic", params={"attribute": "CODE_GENDER"})
    assert r.status_code == 200
    groups = {row["group"] for row in r.json()}
    assert "F" in groups and "M" in groups


def test_segments_demographic_rejects_invalid_attribute(api_client):
    r = api_client.get("/segments/demographic", params={"attribute": "NOT_A_COLUMN"})
    assert r.status_code == 400


def test_segments_demographic_flags_small_groups_as_unreliable(api_client):
    """Regression test for a real gap found while stress-testing the
    dashboard: CODE_GENDER's 4-row 'XNA' group showed a clean 100%
    approval / 0% default rate on the Segments screen with nothing to
    mark it as unreliable -- exactly the "silently confident, actually
    just noise" failure mode. reliable=False must now be present."""
    r = api_client.get("/segments/demographic", params={"attribute": "CODE_GENDER"})
    rows = {row["group"]: row for row in r.json()}
    assert rows["XNA"]["count"] < 500
    assert rows["XNA"]["reliable"] is False
    assert rows["F"]["reliable"] is True
    assert rows["M"]["reliable"] is True


# --- fairness ----------------------------------------------------------


def test_fairness_by_period(api_client):
    r = api_client.get("/fairness/by-period")
    assert r.status_code == 200
    periods = {row["period"] for row in r.json()}
    assert periods == {"P0", "P1", "P2", "P3", "P4"}


def test_fairness_by_period_filter(api_client):
    r = api_client.get("/fairness/by-period", params={"period": "P1"})
    assert r.status_code == 200
    assert [row["period"] for row in r.json()] == ["P1"]


def test_fairness_isolated_effect(api_client):
    r = api_client.get("/fairness/isolated-effect")
    assert r.status_code == 200
    rows = r.json()
    assert {row["period"] for row in rows} == {"P1", "P2", "P3", "P4"}  # P0 has no planted bias
    assert all(row["F_FPR_inflation"] > 0 for row in rows)  # the documented finding


def test_fairness_by_attribute(api_client):
    r = api_client.get("/fairness/by-attribute")
    assert r.status_code == 200
    attributes = {row["attribute"] for row in r.json()}
    assert attributes == {"CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"}


def test_fairness_live(api_client):
    r = api_client.get("/fairness/live", params={"threshold": 0.22, "attribute": "age_band"})
    assert r.status_code == 200
    body = r.json()
    assert body["DPD"] >= 0


def test_fairness_live_rejects_invalid_attribute(api_client):
    r = api_client.get("/fairness/live", params={"threshold": 0.22, "attribute": "NOT_AN_ATTRIBUTE"})
    assert r.status_code == 400


def test_fairness_live_reports_excluded_small_groups(api_client):
    """Same regression as test_segments_demographic_flags_small_groups_as_unreliable,
    for the live-threshold fairness endpoint the Policy Simulator uses:
    CODE_GENDER's 'XNA' (n=4) must be surfaced as excluded from the
    DPD/EOD spread, not silently dropped with no trace."""
    r = api_client.get("/fairness/live", params={"threshold": 0.22, "attribute": "CODE_GENDER"})
    body = r.json()
    assert any("XNA" in g for g in body["excluded_groups"])


def test_fairness_live_no_excluded_groups_when_none_are_small(api_client):
    r = api_client.get("/fairness/live", params={"threshold": 0.22, "attribute": "age_band"})
    body = r.json()
    assert body["excluded_groups"] == []


def test_fairness_tradeoff(api_client):
    r = api_client.get("/fairness/tradeoff", params={"attribute": "age_band"})
    assert r.status_code == 200
    strategies = {row["strategy"] for row in r.json()}
    assert strategies == {"baseline", "equalize_dp", "equalize_eo", "blunt_stricter"}


def test_fairness_tradeoff_rejects_invalid_attribute(api_client):
    r = api_client.get("/fairness/tradeoff", params={"attribute": "NOT_AN_ATTRIBUTE"})
    assert r.status_code == 400


# --- drift ---------------------------------------------------------------


def test_drift_all(api_client):
    r = api_client.get("/drift")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 16  # 4 features x 4 periods (P0 excluded, it's the reference)


def test_drift_filtered(api_client):
    r = api_client.get("/drift", params={"period": "P4", "feature": "EXT_SOURCE_1"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["psi"] > 0.5  # P4 is the largest-drift period by design


# --- simulate --------------------------------------------------------------


def test_simulate_threshold_extremes(api_client):
    r0 = api_client.get("/simulate/threshold", params={"threshold": 0.0}).json()
    r1 = api_client.get("/simulate/threshold", params={"threshold": 1.0}).json()
    assert r0["approval_rate"] < r1["approval_rate"]
    assert r1["approval_rate"] == 1.0


def test_simulate_threshold_has_expected_loss(api_client):
    r = api_client.get("/simulate/threshold", params={"threshold": 0.22}).json()
    assert r["total_expected_loss"] > 0


# --- extreme / missing feature-value applicants ---------------------------
# Real applicants picked because they sit at a genuine extreme in
# modeling_feature_set, found by scanning the table directly rather than
# guessing -- the scoring endpoint only accepts an existing SK_ID_CURR
# (see scoring.py's own docstring on why), so "extreme input" here means
# "a real row with an extreme/missing value", not an arbitrary payload.

MOSTLY_MISSING_APPLICANT = 100024  # 35 of 46 modeling_feature_set columns are NaN
EXTREME_BUREAU_CREDIT_APPLICANT = 442645  # bureau_total_amt_credit_sum ~= 1.02 billion
EXTREME_NEGATIVE_INTERACTION_APPLICANT = 244750  # dti_utilization_interaction ~= -17.5
EXTREME_DTI_APPLICANT = 124157  # dti_ratio ~= 1.88 (debt far exceeds income)


@pytest.mark.parametrize(
    "sk_id_curr",
    [
        MOSTLY_MISSING_APPLICANT,
        EXTREME_BUREAU_CREDIT_APPLICANT,
        EXTREME_NEGATIVE_INTERACTION_APPLICANT,
        EXTREME_DTI_APPLICANT,
    ],
)
def test_score_extreme_or_missing_value_applicant(api_client, sk_id_curr):
    r = api_client.get(f"/applicants/{sk_id_curr}/score")
    assert r.status_code == 200
    body = r.json()
    assert np.isfinite(body["pd_calibrated"])
    assert 0.0 <= body["pd_calibrated"] <= 1.0
    assert np.isfinite(body["expected_loss"])
    assert body["expected_loss"] >= 0


@pytest.mark.parametrize(
    "sk_id_curr",
    [
        MOSTLY_MISSING_APPLICANT,
        EXTREME_BUREAU_CREDIT_APPLICANT,
        EXTREME_NEGATIVE_INTERACTION_APPLICANT,
        EXTREME_DTI_APPLICANT,
    ],
)
def test_explanation_extreme_or_missing_value_applicant_stays_additive_consistent(api_client, sk_id_curr):
    """The hard rule (additive consistency) must hold even for the
    heaviest-missing-data / most-extreme-value real applicants, not just
    "typical" ones."""
    r = api_client.get(f"/applicants/{sk_id_curr}/explanation", params={"top_k": 200})
    assert r.status_code == 200
    body = r.json()
    assert all(np.isfinite(c["shap_value"]) for c in body["contributions"])
    total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert total == pytest.approx(body["raw_margin"], abs=1e-4)


def test_simulate_profit_curve(api_client):
    r = api_client.get("/simulate/profit-curve")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 101  # N_THRESHOLDS from expected_loss.py
