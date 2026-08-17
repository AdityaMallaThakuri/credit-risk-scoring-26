"""Phase 8.1 + the project's standing hard rule: any modified SHAP values
(Weighted Temporal SHAP included) must be renormalized so they still sum
to prediction - base_value. Checked two ways:
  - a pure-function unit test with synthetic data (fast, no model needed)
  - an integration test against the real deployed model + a real
    applicant, through the exact same state.py code path the API uses
"""

import numpy as np
import pytest

from weighted_temporal_shap import blend_weights, weighted_temporal_shap

RNG = np.random.RandomState(42)


def test_weighted_shap_sums_to_raw_margin_minus_base_value_synthetic():
    n_applicants, n_features = 20, 10
    static_values = RNG.normal(size=(n_applicants, n_features))
    base_value = RNG.normal(size=n_applicants)
    raw_margin = base_value + static_values.sum(axis=1) + RNG.normal(scale=0.1, size=n_applicants)  # simulate the usual small SHAP-mode discrepancy
    w_drift = RNG.uniform(0.2, 1.0, size=n_features)
    w_cost = RNG.uniform(0.0, 1.0, size=n_features)

    for alpha in [1.0, 0.75, 0.5, 0.25, 0.0]:
        weights = blend_weights(w_drift, w_cost, alpha)
        weighted = weighted_temporal_shap(static_values, base_value, raw_margin, weights)
        reconstructed = base_value + weighted.sum(axis=1)
        np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-8)


def test_weighted_shap_degenerate_zero_sum_row_does_not_crash():
    """Known edge case, not papered over: if a row's WEIGHTED contributions
    sum to exactly zero, no finite rescale can hit a nonzero target (0 x
    anything is still 0) -- rescale_to_additive_consistency's documented
    fallback (scale=1.0) means additive consistency is NOT guaranteed for
    this specific degenerate input. This test only asserts the function
    stays finite and doesn't raise, not that consistency holds here."""
    static_values = np.zeros((1, 4))
    base_value = np.array([1.0])
    raw_margin = np.array([2.0])  # nonzero target, unreachable from an all-zero row
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = weighted_temporal_shap(static_values, base_value, raw_margin, weights)
    assert np.all(np.isfinite(weighted))


@pytest.mark.parametrize("period", ["P0", "P1", "P2", "P3", "P4"])
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_weighted_shap_exact_reconstruction_on_real_applicant(app_state, period, alpha):
    """Integration-level: the exact code path the /weighted-explanation
    endpoint uses, on a real applicant and the real deployed model."""
    sk_id_curr = app_state.applicant_features.index[0]
    row = app_state.applicant_features.loc[[sk_id_curr]].drop(columns=["TARGET"])
    X_t = app_state.preprocessor.transform(row)

    shap_values = app_state.explainer(X_t)
    base_value = float(shap_values.base_values[0])
    static_values = shap_values.values[0]
    raw_margin = float(app_state.model.predict(X_t, raw_score=True)[0])

    weighted = app_state.compute_weighted_shap(static_values, base_value, raw_margin, period, alpha)

    reconstructed = base_value + weighted.sum()
    assert np.isclose(reconstructed, raw_margin, atol=1e-6)
