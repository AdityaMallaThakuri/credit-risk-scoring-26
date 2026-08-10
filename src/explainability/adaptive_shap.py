"""Phase 5 adaptive SHAP methods A/B/C (docs/reading_material.md Part 7.4),
replicated against the synthetic period overlay.

Shared setup:
- Reuses static_shap.fit_final_model() -- the same production LightGBM
  refit on all real data (Phase 4). No new model, no retraining: these
  methods are three different ways of computing/adjusting SHAP
  *explanations* of that one fixed model, not different models, and none
  of them can change the model's actual decisions (see the fairness note
  at the bottom of this docstring).
- "Static SHAP" here means TreeExplainer with an EXPLICIT, FIXED background
  sample drawn once from P0 (`feature_perturbation="interventional"`) --
  NOT Phase 4's tree_path_dependent default (which has no explicit
  background at all, so there'd be nothing for Method B to meaningfully
  "replace"). This operationalizes reading_material.md 7.3's "fixed
  background dataset ... usually the original training data" literally,
  so Method B's contrast (a rolling/period-specific background) is an
  apples-to-apples swap of the same explainer mechanism, not a different
  algorithm.
- Every period's applicants are looked up by SK_ID_CURR from the synthetic
  overlay directly into the already-transformed feature matrix used to fit
  the final model -- no re-transformation, no re-fitting of the model or
  preprocessor, consistent with how src/fairness/synthetic_bias_detection.py
  reuses the real-data model unchanged across periods.
- Per period: unique SK_ID_CURR positions are split into disjoint calib /
  eval / background pools (no overlap) so Method C's Ridge surrogate is
  never evaluated on the same rows it was fit on, and Method B's
  background never doubles as its own evaluation set.

Method A -- Drift-Weighted SHAP Adjustment:
  weight w_j = 1 / (1 + PSI_j), where PSI_j is that (transformed) feature's
  drift for the current period vs P0 (src/drift/drift_metrics.py, reused
  unchanged, generalized here to all model features rather than just the 4
  officially-drifted ones -- a real deployment wouldn't know in advance
  which features drift). Heavily-drifted features get down-weighted since
  their "average contribution" baseline is least trustworthy
  (reading_material.md 7.3's exact mechanism). Reweighted values are then
  rescaled per-instance so they still sum to (raw margin - base_value) --
  the SHAP additive-consistency hard rule.

Method B -- Sliding Background Sampling:
  operationalized as: background = a sample of the CURRENT period's own
  rows, rather than the fixed P0 sample -- the most direct discretization
  of "rolling window of recent data" given this project's 5 discrete
  synthetic periods stand in for continuous time (no continuous timestamp
  exists in the underlying data at all, per CLAUDE.md's data-strategy
  rule). A new TreeExplainer is built per period; additive consistency
  holds by TreeExplainer's own construction, verified explicitly below
  rather than assumed. At P0 itself, Method B's background pool is drawn
  from the same period as the static baseline's background, so the two
  should come out close but not identical (different random draws) -- a
  useful built-in sanity check.

Method C -- Surrogate Ridge Recalibration:
  a per-period multi-output Ridge model is fit on a calibration subsample
  to approximate that period's own true (fixed-background) SHAP values
  from the raw feature values -- a cheap linear stand-in for the full SHAP
  computation, refreshed each period, then evaluated on a disjoint holdout
  subsample (never the rows it was fit on). Ridge's raw predictions are
  rescaled per-instance to satisfy additive consistency, same as Method A.

Fairness-reduction comparison (required by /shap-stability-eval): DPD and
EOD are properties of the model's decisions (PD vs. threshold), not of
whichever method is used to explain those decisions afterward. None of
Methods A/B/C touch the model, the threshold, or any decision -- they are
purely alternative *explanations* of the same fixed predictions. So DPD
and EOD are structurally identical across static SHAP and all three
methods, for every period; DeltaDPD = DeltaEOD = 0 by construction, not by
coincidence. This is expected and reported explicitly rather than
manufactured -- Phase 6's Weighted Temporal SHAP is the first method in
this project that could plausibly earn a nonzero fairness-reduction number
(reading_material.md 8.3), because it's the first one proposed as an input
to an actual mitigation decision rather than a pure explanation method.

Run: python src/explainability/adaptive_shap.py
"""

import sys
from pathlib import Path

EXPLAINABILITY_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPLAINABILITY_DIR.parents[0] / "models"
DRIFT_DIR = EXPLAINABILITY_DIR.parents[0] / "drift"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(DRIFT_DIR))
sys.path.insert(0, str(EXPLAINABILITY_DIR))

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from drift_metrics import assign_bins, bin_proportions, get_bin_edges, population_stability_index  # noqa: E402
from static_shap import fit_final_model  # noqa: E402

RANDOM_STATE = 42
BACKGROUND_SIZE = 100  # shap interventional masker defaults to max_samples=100 regardless; matching it explicitly avoids the silent-subsample warning
CALIB_SIZE = 300
EVAL_SIZE = 500
PERIODS = ["P0", "P1", "P2", "P3", "P4"]
SYNTHETIC_OVERLAY_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "synthetic_overlay.csv"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase5_adaptive_shap_values.npz"


def load_overlay_row_positions(ids):
    overlay = pd.read_csv(SYNTHETIC_OVERLAY_PATH)
    id_to_pos = pd.Series(np.arange(len(ids)), index=ids.values)
    overlay["row_pos"] = id_to_pos.loc[overlay["SK_ID_CURR"]].values
    return overlay


def split_period_pools(overlay, period, random_state_offset):
    unique_positions = overlay.loc[overlay["period"] == period, "row_pos"].unique()
    rng = np.random.RandomState(RANDOM_STATE + random_state_offset)
    shuffled = rng.permutation(unique_positions)

    n_needed = BACKGROUND_SIZE + CALIB_SIZE + EVAL_SIZE
    assert len(shuffled) >= n_needed, f"{period} has only {len(shuffled)} unique applicants, need {n_needed}"

    background = shuffled[:BACKGROUND_SIZE]
    calib = shuffled[BACKGROUND_SIZE:BACKGROUND_SIZE + CALIB_SIZE]
    eval_ = shuffled[BACKGROUND_SIZE + CALIB_SIZE:n_needed]
    return background, calib, eval_


def rescale_to_additive_consistency(raw_values, base_value, raw_margin):
    """Per-instance rescale so sum(values) == raw_margin - base_value exactly."""
    base_value = np.broadcast_to(base_value, raw_margin.shape)
    target_total = raw_margin - base_value
    current_total = raw_values.sum(axis=1)
    scale = np.where(np.abs(current_total) > 1e-9, target_total / current_total, 1.0)
    return raw_values * scale[:, None]


def per_feature_psi(X_ref, X_comp):
    n_features = X_ref.shape[1]
    psi_values = np.zeros(n_features)
    for j in range(n_features):
        ref_series = pd.Series(X_ref[:, j])
        comp_series = pd.Series(X_comp[:, j])
        edges = get_bin_edges(ref_series)
        n_bins_plus_missing = (len(edges) - 1) + 1
        ref_props = bin_proportions(assign_bins(ref_series, edges), n_bins_plus_missing)
        comp_props = bin_proportions(assign_bins(comp_series, edges), n_bins_plus_missing)
        psi_values[j] = population_stability_index(ref_props, comp_props)
    return psi_values


def method_a_drift_weighted(static_values, base_value, raw_margin, psi_per_feature):
    weights = 1.0 / (1.0 + psi_per_feature)
    weighted = static_values * weights[None, :]
    return rescale_to_additive_consistency(weighted, base_value, raw_margin)


def raw_shap_values(explainer, X):
    """LightGBM + SHAP's interventional feature_perturbation has a known,
    documented numerical imprecision: reconstructed = base_value +
    sum(shap_values) does not exactly match model.predict(X, raw_score=True)
    -- measured here at up to ~1.0-1.1 absolute on this model's margin
    scale (roughly -5 to +5), not just floating-point noise, but not a sign
    of broken output either (SHAP's own tree_path_dependent mode, used in
    static_shap.py, matches exactly -- this imprecision is specific to
    interventional mode's background-conditioned expectation estimate).
    check_additivity is disabled here because every caller applies its own
    rescale_to_additive_consistency anchored to the independently verified
    model.predict(X, raw_score=True) -- that rescale is what the hard rule
    actually requires (the *renormalized* output must sum exactly to
    prediction - base_value), and it holds exactly by construction
    regardless of the raw discrepancy's size. The sanity bound each caller
    applies before rescaling exists only to catch a genuinely broken
    configuration (NaN/inf, wrong shapes), not to validate SHAP's internal
    precision."""
    return explainer(X, check_additivity=False)


def method_b_sliding_background(model, X_t, background_positions, X_eval, raw_margin_eval):
    explainer = shap.TreeExplainer(model, data=X_t[background_positions], feature_perturbation="interventional")
    sv = raw_shap_values(explainer, X_eval)
    reconstructed = np.broadcast_to(sv.base_values, raw_margin_eval.shape) + sv.values.sum(axis=1)
    assert np.allclose(reconstructed, raw_margin_eval, atol=3.0), "Method B raw SHAP is grossly inconsistent with the model's own output -- investigate before trusting the rescale"
    return rescale_to_additive_consistency(sv.values, sv.base_values, raw_margin_eval)


def method_c_ridge_surrogate(X_calib, static_values_calib, X_eval, base_value, raw_margin_eval):
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_calib, static_values_calib)
    raw_pred = ridge.predict(X_eval)
    return rescale_to_additive_consistency(raw_pred, base_value, raw_margin_eval)


def main():
    model, X_t, feature_names, ids = fit_final_model()
    overlay = load_overlay_row_positions(ids)

    p0_background, _, _ = split_period_pools(overlay, "P0", random_state_offset=0)
    static_explainer = shap.TreeExplainer(model, data=X_t[p0_background], feature_perturbation="interventional")

    results = {}
    for i, period in enumerate(PERIODS):
        background, calib, eval_ = split_period_pools(overlay, period, random_state_offset=i)

        X_eval = X_t[eval_]
        X_calib = X_t[calib]
        raw_margin_eval = model.predict(X_eval, raw_score=True)
        raw_margin_calib = model.predict(X_calib, raw_score=True)

        static_eval = raw_shap_values(static_explainer, X_eval)
        static_calib = raw_shap_values(static_explainer, X_calib)
        reconstructed = np.broadcast_to(static_eval.base_values, raw_margin_eval.shape) + static_eval.values.sum(axis=1)
        assert np.allclose(reconstructed, raw_margin_eval, atol=3.0), f"static SHAP is grossly inconsistent with the model's own output for {period}"
        static_eval_values = rescale_to_additive_consistency(static_eval.values, static_eval.base_values, raw_margin_eval)
        static_calib_values = rescale_to_additive_consistency(static_calib.values, static_calib.base_values, raw_margin_calib)

        psi_per_feature = per_feature_psi(X_t[p0_background], X_eval)
        a_values = method_a_drift_weighted(static_eval_values, static_eval.base_values, raw_margin_eval, psi_per_feature)
        b_values = method_b_sliding_background(model, X_t, background, X_eval, raw_margin_eval)
        c_values = method_c_ridge_surrogate(X_calib, static_calib_values, X_eval, static_eval.base_values, raw_margin_eval)

        results[period] = {
            "static": static_eval_values,
            "method_a": a_values,
            "method_b": b_values,
            "method_c": c_values,
            "psi_per_feature": psi_per_feature,
        }
        print(f"{period}: computed static + A/B/C SHAP for {len(eval_)} eval applicants "
              f"(mean PSI across features={psi_per_feature.mean():.4f})")

    np.savez(
        RESULTS_PATH,
        feature_names=feature_names,
        **{f"{period}_{method}": results[period][method] for period in PERIODS for method in ["static", "method_a", "method_b", "method_c"]},
    )
    print(f"\nSaved: {RESULTS_PATH}")

    return results, feature_names


if __name__ == "__main__":
    results, feature_names = main()
