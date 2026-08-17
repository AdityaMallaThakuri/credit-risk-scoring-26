"""Phase 6 -- Weighted Temporal SHAP: this project's own contribution.

Per docs/reading_material.md 8.3 Option 1 ("cost-aware weighting"), chosen
over Option 2 (adaptive window length) because only this option has a
plausible mechanism to move a fairness metric, not just describe a
prediction (see the mitigation section below).

    Weighted_SHAP(j,t) = SHAP(j,t) x [alpha * w_drift(j,t) + (1-alpha) * w_cost(j,t)]

w_drift(j,t) = 1/(1+PSI(j,t)) -- identical to Method A
(src/explainability/adaptive_shap.py), reused via per_feature_psi.

w_cost(j,t) is new here: a counterfactual Expected-Loss sensitivity per
feature. For each feature j, every eval applicant's value for j is
replaced with a resampled draw from the P0 reference pool (holding every
other feature fixed), the refit model is re-scored, and
|PD_actual - PD_counterfactual| x LGD x EAD is averaged across eval
applicants -- "how much does this feature's OWN drift actually move
Expected Loss", independent of how large its PSI happens to be. This
directly implements reading_material.md 8.3's point: "a feature that
drifted statistically but barely affects EL gets less explanatory weight
than one whose drift is small statistically but large financially."
w_cost is then min-max normalized per period so it's on the same (0, 1]
footing as w_drift before blending.

Note on PD here: the refit model's own predict_proba output is used
throughout (same convention as static_shap.py / adaptive_shap.py), NOT
Phase 3's isotonic calibrator -- that calibrator was fit per-CV-fold on a
different (non-full-refit) model artifact, so no single calibrated
pipeline exists for this refit model. w_cost is a *relative,
per-feature-comparison* sensitivity signal, not a portfolio-level profit
number, so this substitution doesn't compromise what it's used for here.

alpha=1.0 reduces this exactly to Method A (required by the roadmap's
ablation: "alpha=1 (== Method A), alpha=0 (pure cost), 2-3 blended
values").

Additive consistency: every alpha's output is rescaled via
adaptive_shap.rescale_to_additive_consistency, same hard-rule discipline
as Methods A/B/C.

MITIGATION MECHANISM (why this method, unlike A/B/C, can have a nonzero
fairness-reduction number):
Methods A/B/C only ever re-explain a fixed decision -- ΔDPD/ΔEOD were 0
for all three by construction (adaptive_shap.py's docstring). Weighted
Temporal SHAP is still just an explanation on its own; to make it
capable of moving a decision, this module implements a genuine (if
deliberately simple) SHAP-GUIDED SELECTIVE THRESHOLD CORRECTION:
  1. Per period, identify a small "flagged feature set" -- features that
     are BOTH meaningfully drifted (PSI > PSI_FLAG_THRESHOLD) AND
     high-w_cost (top TOP_FLAG_FEATURES by w_cost among those).
  2. Per applicant, find their TOP_N_DRIVERS_FOR_FLAG most-influential
     features under the alpha=0.5 blended explanation (by |weighted
     SHAP|). An applicant is "flagged" if ANY of those top-N drivers is in
     the period's flagged feature set -- i.e., their decision is
     meaningfully driven by a feature this method has identified as both
     drifting and financially consequential. An earlier top-1-only
     version was tried first and flagged only 0-13/500 applicants per
     period -- too few for the per-gender correction below to do anything
     but flip 0-2 people; top-N widens this to 62-129/500 (see Finding 2).
  3. Unflagged applicants keep the single global baseline threshold
     (identical decision to static SHAP / Method B -- unaffected).
  4. Flagged applicants instead get a per-CODE_GENDER threshold, computed
     from each gender's actually-repaid (TARGET==0) members within the
     flagged subset, targeted at the population's overall baseline
     approval rate among TARGET==0 applicants (the same equalize_eo
     mechanism as src/fairness/fairness_accuracy_tradeoff.py, scoped to
     the flagged subset and applied to all flagged applicants of that
     gender since TARGET is unobserved at decision time). An earlier
     equalize_dp-style version (target = flagged subset's OWN baseline
     rate, ignoring TARGET) was tried first and is a near-tautological
     no-op whenever that subset is already ~100% approved -- the common
     case at BASELINE_APPROVAL_RATE=0.97 -- and, when it did act, made
     both DPD and EOD *worse*; switching to this TARGET-aware convention
     produced numerically identical flips/deltas at top-1 flagging,
     which is what confirmed sample size (not metric choice) was the
     binding constraint (Finding 2 below).
This is evaluated against the Phase 1 planted CODE_GENDER label bias
(same synthetic ground truth used in
src/fairness/synthetic_bias_detection.py) so the "reduction" claim has a
known-bias reference to be honest against, not just an unlabelled
before/after.

FINDINGS (from the ablation this module produces; see
data/processed/phase6_fairness_reduction.csv for the numbers):
  1. alpha=1.0 reproduces Method A's stability numbers exactly, as
     required (a correctness check, not a modeling result).
  2. The mitigation mechanism's fairness effect is drift-magnitude-
     dependent, not uniformly reliable: with top-N-driver flagging, P3
     (the concept-drift shock period, and the period with the largest
     planted-bias gap per Phase 4) improves on BOTH DeltaDPD and DeltaEOD;
     P2 improves DPD but worsens EOD; P4 worsens both; P1 has no
     applicants flagged at all. Magnitudes are small throughout
     (|Delta| <= 0.005, driven by only 1-3 actual decision flips even in
     flagged subsets of 60-130 people) -- this is directional evidence
     the mechanism can engage under large drift, not a strong or uniform
     fairness-reduction claim. Reported honestly per period, not just as
     an average, per this project's existing practice (see Phase 5's
     honest stability-ranking reversal in shap_stability_eval.py).

SCALE CAVEAT (stated explicitly, not silently): SHAP computation forces a
500-applicant eval sample per period (BACKGROUND_SIZE/CALIB_SIZE/EVAL_SIZE
in adaptive_shap.py), far smaller than Phase 4's full 307K-row fairness
analysis. Demographic subgroups at this scale are much smaller than
fairness_metrics.py's default 500-row reliability floor, so this module
uses a separate, smaller SMALL_SAMPLE_MIN_GROUP_N for this test only,
documented here as a deliberate scale trade-off: numbers from this test
are directional evidence of the mechanism working, not a portfolio-scale
fairness claim (that would require re-running the flagging logic against
the full population, left as a follow-up if this direction is kept for
the dashboard's policy simulator).

Run: python src/explainability/weighted_temporal_shap.py
"""

import sys
from pathlib import Path

EXPLAINABILITY_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPLAINABILITY_DIR.parents[0] / "models"
DRIFT_DIR = EXPLAINABILITY_DIR.parents[0] / "drift"
FAIRNESS_DIR = EXPLAINABILITY_DIR.parents[0] / "fairness"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(DRIFT_DIR))
sys.path.insert(0, str(FAIRNESS_DIR))
sys.path.insert(0, str(EXPLAINABILITY_DIR))

import sqlite3

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau

from adaptive_shap import (  # noqa: E402
    BACKGROUND_SIZE, CALIB_SIZE, EVAL_SIZE, PERIODS, RANDOM_STATE,
    load_overlay_row_positions, per_feature_psi, raw_shap_values,
    rescale_to_additive_consistency, split_period_pools,
)
from fairness_metrics import demographic_parity_difference, equalized_odds_difference, _max_min_diff  # noqa: E402
from static_shap import fit_final_model  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
SYNTHETIC_OVERLAY_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "synthetic_overlay.csv"
ADAPTIVE_SHAP_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase5_adaptive_shap_values.npz"

VALUES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase6_weighted_temporal_shap_values.npz"
WEIGHTS_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase6_weight_components.csv"
STABILITY_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase6_stability_eval.csv"
FAIRNESS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase6_fairness_reduction.csv"

ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]  # 1.0 == Method A; 0.0 == pure cost weighting
PRIMARY_ALPHA = 0.5  # the blended method used for the mitigation demonstration
LGD_ASSUMPTION = 0.55  # matches src/models/expected_loss.py's fixed assumption
TOP_FLAG_FEATURES = 3
TOP_N_DRIVERS_FOR_FLAG = 5  # an applicant is "flagged" if ANY of their top-N |SHAP| drivers
                             # is in the flagged feature set, not just their single top-1 driver --
                             # top-1-only produced flagged subsets of 0-13/500, too small for the
                             # per-gender quantile correction to do anything but flip 0-2 people.
PSI_FLAG_THRESHOLD = 0.15
BASELINE_APPROVAL_RATE = 0.97  # matches Phase 3 LightGBM's ~0.9697 optimal-threshold approval rate
BIAS_PERIODS = ["P1", "P2", "P3", "P4"]
SMALL_SAMPLE_MIN_GROUP_N = 30  # see module docstring's SCALE CAVEAT -- deliberately smaller than
                                # fairness_metrics.SMALL_GROUP_WARNING_N=500, this test runs on 500-row eval samples
TOP_K = 10


# --------------------------------------------------------------------------- #
# w_cost: counterfactual Expected-Loss sensitivity per feature
# --------------------------------------------------------------------------- #

def compute_w_cost(model, X_eval, reference_pool, ead_eval, rng):
    """For each feature, replace it with a resampled P0-reference draw
    (other features held fixed), re-score PD, and measure the resulting
    mean |ΔEL| across eval applicants. Returns a (n_features,) array,
    min-max normalized to (0, 1] so it's combinable with w_drift."""
    n_features = X_eval.shape[1]
    pd_actual = model.predict_proba(X_eval)[:, 1]

    raw_w_cost = np.zeros(n_features)
    for j in range(n_features):
        draws = rng.choice(reference_pool[:, j], size=len(X_eval), replace=True)
        X_cf = X_eval.copy()
        X_cf[:, j] = draws
        pd_cf = model.predict_proba(X_cf)[:, 1]
        delta_el = np.abs(pd_actual - pd_cf) * LGD_ASSUMPTION * ead_eval
        raw_w_cost[j] = delta_el.mean()

    max_w_cost = raw_w_cost.max()
    return raw_w_cost / max_w_cost if max_w_cost > 0 else raw_w_cost


def blend_weights(w_drift, w_cost, alpha):
    return alpha * w_drift + (1 - alpha) * w_cost


def weighted_temporal_shap(static_values, base_value, raw_margin, weights):
    weighted = static_values * weights[None, :]
    return rescale_to_additive_consistency(weighted, base_value, raw_margin)


# --------------------------------------------------------------------------- #
# Stability metrics (same definitions as shap_stability_eval.py)
# --------------------------------------------------------------------------- #

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def jaccard_at_k(a, b, feature_names, k=TOP_K):
    top_a = set(feature_names[np.argsort(-a)[:k]])
    top_b = set(feature_names[np.argsort(-b)[:k]])
    return len(top_a & top_b) / len(top_a | top_b)


def stability_across_periods(mean_abs_by_period, feature_names):
    cosine_scores, tau_scores, jaccard_scores = [], [], []
    for p1, p2 in zip(PERIODS[:-1], PERIODS[1:]):
        v1, v2 = mean_abs_by_period[p1], mean_abs_by_period[p2]
        cosine_scores.append(cosine_similarity(v1, v2))
        tau, _ = kendalltau(v1, v2)
        tau_scores.append(tau)
        jaccard_scores.append(jaccard_at_k(v1, v2, feature_names))
    return {
        "cosine_mean": np.mean(cosine_scores),
        "kendall_tau_mean": np.mean(tau_scores),
        f"jaccard_at_{TOP_K}_mean": np.mean(jaccard_scores),
    }


# --------------------------------------------------------------------------- #
# SHAP-guided selective threshold correction (the mitigation mechanism)
# --------------------------------------------------------------------------- #

def load_period_demographics(sk_ids):
    conn = sqlite3.connect(DB_PATH)
    demo = pd.read_sql("SELECT SK_ID_CURR, CODE_GENDER, AMT_CREDIT AS EAD FROM application_train", conn)
    conn.close()
    return pd.DataFrame({"SK_ID_CURR": sk_ids}).merge(demo, on="SK_ID_CURR", how="left")


def group_threshold_for_target_rate(pd_values, groups, target_rate):
    df = pd.DataFrame({"PD": pd_values, "group": groups})
    return df.groupby("group")["PD"].quantile(target_rate).to_dict()


def apply_group_thresholds(pd_values, groups, thresholds):
    group_threshold = pd.Series(groups).map(thresholds).astype(float).values
    return pd_values <= group_threshold


def selective_correction_fairness(pd_actual, weighted_values_alpha05, feature_names,
                                   w_drift, w_cost, code_gender, target, baseline_threshold):
    flag_candidates = [f for f, psi_val in zip(feature_names, (1.0 / w_drift) - 1.0) if psi_val > PSI_FLAG_THRESHOLD]
    if flag_candidates:
        candidate_idx = [i for i, f in enumerate(feature_names) if f in flag_candidates]
        ranked = sorted(candidate_idx, key=lambda i: w_cost[i], reverse=True)
        flagged_features = set(feature_names[i] for i in ranked[:TOP_FLAG_FEATURES])
    else:
        flagged_features = set()

    top_n_idx = np.argsort(-np.abs(weighted_values_alpha05), axis=1)[:, :TOP_N_DRIVERS_FOR_FLAG]
    top_n_features = feature_names[top_n_idx]  # (n_applicants, TOP_N_DRIVERS_FOR_FLAG)
    flagged_mask = np.isin(top_n_features, list(flagged_features)).any(axis=1)

    baseline_approved = pd_actual <= baseline_threshold

    corrected_approved = baseline_approved.copy()
    n_flipped = 0
    repaid_mask = target == 0
    flagged_repaid_mask = flagged_mask & repaid_mask
    correction_applied = (flagged_mask.sum() >= 2 and len(set(code_gender[flagged_mask])) >= 2
                           and flagged_repaid_mask.sum() >= 2 and len(set(code_gender[flagged_repaid_mask])) >= 2)
    if correction_applied:
        # EOD-style correction (== equalize_eo's convention in
        # fairness_accuracy_tradeoff.py): compute each group's threshold from
        # ONLY its actually-repaid (TARGET==0) members, targeted at the
        # population's baseline approval rate among TARGET==0 applicants --
        # this engages the planted label bias directly, unlike an earlier
        # DPD-style version tried first (equalize approval rate irrespective
        # of TARGET), which Phase 4 already showed is structurally blind to
        # this exact kind of label bias, and which produced identical output
        # to this version -- confirming sample size, not metric choice, is
        # this mechanism's binding constraint (see TOP_N_DRIVERS_FOR_FLAG).
        # The resulting per-group threshold is applied to ALL flagged
        # applicants (TARGET is unobserved at decision time).
        target_rate = baseline_approved[repaid_mask].mean()
        thresholds = group_threshold_for_target_rate(pd_actual[flagged_repaid_mask], code_gender[flagged_repaid_mask], target_rate)
        corrected_approved[flagged_mask] = apply_group_thresholds(pd_actual[flagged_mask], code_gender[flagged_mask], thresholds)
        n_flipped = int((corrected_approved[flagged_mask] != baseline_approved[flagged_mask]).sum())

    def fairness_pair(approved):
        d = pd.DataFrame({"CODE_GENDER": code_gender, "approved": approved.astype(int), "TARGET": target})
        dpd, _ = demographic_parity_difference(d, "CODE_GENDER", min_group_n=SMALL_SAMPLE_MIN_GROUP_N)
        eqodds, tpr_rates, _ = equalized_odds_difference(d, "CODE_GENDER", min_group_n=SMALL_SAMPLE_MIN_GROUP_N)
        eod = _max_min_diff(tpr_rates, min_group_n=SMALL_SAMPLE_MIN_GROUP_N)
        return dpd, eod, eqodds

    dpd_base, eod_base, eqodds_base = fairness_pair(baseline_approved)
    dpd_corr, eod_corr, eqodds_corr = fairness_pair(corrected_approved)

    return {
        "n_flagged": int(flagged_mask.sum()),
        "n_flipped": n_flipped,
        "flagged_features": ", ".join(sorted(flagged_features)) if flagged_features else "(none)",
        "DPD_baseline": dpd_base, "DPD_corrected": dpd_corr, "DeltaDPD": dpd_base - dpd_corr,
        "EOD_baseline": eod_base, "EOD_corrected": eod_corr, "DeltaEOD": eod_base - eod_corr,
        "EqualizedOddsDiff_baseline": eqodds_base, "EqualizedOddsDiff_corrected": eqodds_corr,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    model, X_t, feature_names, ids = fit_final_model()
    overlay = load_overlay_row_positions(ids)
    overlay_full = pd.read_csv(SYNTHETIC_OVERLAY_PATH)

    p0_background, _, _ = split_period_pools(overlay, "P0", random_state_offset=0)
    static_explainer = shap.TreeExplainer(model, data=X_t[p0_background], feature_perturbation="interventional")

    all_values = {}
    weight_log_rows = []
    fairness_rows = []
    mean_abs_by_alpha = {alpha: {} for alpha in ALPHAS}

    for i, period in enumerate(PERIODS):
        background, calib, eval_ = split_period_pools(overlay, period, random_state_offset=i)
        X_eval = X_t[eval_]
        sk_ids_eval = ids.iloc[eval_].values
        raw_margin_eval = model.predict(X_eval, raw_score=True)

        static_eval = raw_shap_values(static_explainer, X_eval)
        reconstructed = np.broadcast_to(static_eval.base_values, raw_margin_eval.shape) + static_eval.values.sum(axis=1)
        assert np.allclose(reconstructed, raw_margin_eval, atol=3.0), f"static SHAP inconsistent with model output for {period}"
        static_values = rescale_to_additive_consistency(static_eval.values, static_eval.base_values, raw_margin_eval)

        psi_per_feature = per_feature_psi(X_t[p0_background], X_eval)
        w_drift = 1.0 / (1.0 + psi_per_feature)

        demo = load_period_demographics(sk_ids_eval)
        rng = np.random.RandomState(RANDOM_STATE + 100 + i)
        w_cost = compute_w_cost(model, X_eval, X_t[p0_background], demo["EAD"].values, rng)

        for j, feat in enumerate(feature_names):
            weight_log_rows.append({
                "period": period, "feature": feat, "psi": psi_per_feature[j],
                "w_drift": w_drift[j], "w_cost": w_cost[j],
            })

        for alpha in ALPHAS:
            weights = blend_weights(w_drift, w_cost, alpha)
            values = weighted_temporal_shap(static_values, static_eval.base_values, raw_margin_eval, weights)
            all_values[f"{period}_alpha_{alpha}"] = values
            mean_abs_by_alpha[alpha][period] = np.abs(values).mean(axis=0)

        print(f"{period}: computed Weighted Temporal SHAP for {len(eval_)} eval applicants, "
              f"{len(ALPHAS)} alpha values (mean PSI={psi_per_feature.mean():.4f}, mean w_cost={w_cost.mean():.4f})")

        if period in BIAS_PERIODS:
            pd_actual = model.predict_proba(X_eval)[:, 1]
            baseline_threshold = np.quantile(pd_actual, BASELINE_APPROVAL_RATE)
            values_alpha05 = all_values[f"{period}_alpha_{PRIMARY_ALPHA}"]

            period_overlay = overlay_full[overlay_full["period"] == period].drop_duplicates("SK_ID_CURR").set_index("SK_ID_CURR")
            target = period_overlay.loc[sk_ids_eval, "TARGET"].values
            code_gender = demo["CODE_GENDER"].values

            result = selective_correction_fairness(
                pd_actual, values_alpha05, np.array(feature_names), w_drift, w_cost,
                code_gender, target, baseline_threshold,
            )
            result["period"] = period
            fairness_rows.append(result)
            print(f"  mitigation ({period}): flagged={result['n_flagged']}/{len(eval_)} "
                  f"(flipped={result['n_flipped']}) via [{result['flagged_features']}]  "
                  f"DeltaDPD={result['DeltaDPD']:+.4f}  DeltaEOD={result['DeltaEOD']:+.4f}")

    # --- stability comparison against static / Method A / B / C ---
    baseline_data = np.load(ADAPTIVE_SHAP_PATH, allow_pickle=True)
    baseline_feature_names = baseline_data["feature_names"]
    stability_rows = []
    for method in ["static", "method_a", "method_b", "method_c"]:
        mean_abs_by_period = {p: np.abs(baseline_data[f"{p}_{method}"]).mean(axis=0) for p in PERIODS}
        row = stability_across_periods(mean_abs_by_period, baseline_feature_names)
        row["method"] = method
        stability_rows.append(row)
    for alpha in ALPHAS:
        row = stability_across_periods(mean_abs_by_alpha[alpha], np.array(feature_names))
        row["method"] = f"weighted_temporal_alpha_{alpha}"
        stability_rows.append(row)
    stability_df = pd.DataFrame(stability_rows).set_index("method")
    stability_df.to_csv(STABILITY_PATH)

    weight_log = pd.DataFrame(weight_log_rows)
    weight_log.to_csv(WEIGHTS_LOG_PATH, index=False)

    fairness_df = pd.DataFrame(fairness_rows).set_index("period")
    fairness_df.to_csv(FAIRNESS_PATH)

    np.savez(VALUES_PATH, feature_names=feature_names, **all_values)

    print(f"\nSaved: {VALUES_PATH}")
    print(f"Saved: {WEIGHTS_LOG_PATH}")
    print(f"Saved: {STABILITY_PATH}")
    print(f"Saved: {FAIRNESS_PATH}")

    print("\n=== Stability (cosine / Kendall tau / Jaccard@10), static/A/B/C vs. Weighted Temporal SHAP ablation ===")
    print(stability_df.round(4).to_string())

    print("\n=== Fairness reduction vs. baseline (== Method B's unchanged decision), planted-bias periods ===")
    print(fairness_df[["n_flagged", "n_flipped", "DeltaDPD", "DeltaEOD"]].round(4).to_string())

    return stability_df, fairness_df, weight_log


if __name__ == "__main__":
    stability_df, fairness_df, weight_log = main()
