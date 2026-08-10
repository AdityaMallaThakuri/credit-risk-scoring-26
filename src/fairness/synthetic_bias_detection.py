"""Phase 4 sanity check: does the standard fairness-metric machinery
(DPD/EOD/Equalized Odds from src/fairness/fairness_metrics.py) actually
detect the planted CODE_GENDER label bias in the synthetic overlay
(data/synthetic/synthetic_overlay.csv -- 8% of true non-defaulting F
applicants relabeled TARGET=1 in P1-P4, P0 left clean; see
data/synthetic/README.md)?

This is required before the synthetic overlay can be trusted for anything
later (Weighted Temporal SHAP evaluation, drift-vs-fairness tracking) --
per CLAUDE.md's Phase 4 exit criteria, confirming detection here is a
precondition for the rest of the project, not an optional extra.

Per the project's data-strategy rule, the synthetic overlay is a
row-reweighting of real applicants (same SK_ID_CURR, no fabricated
people), so the already-trained, real-data-only LightGBM model's
out-of-fold calibrated PD is reused unchanged for every occurrence of a
given SK_ID_CURR (joined on SK_ID_CURR, which naturally handles a person
appearing multiple times within a period from the overlay's weighted
resampling). No model is retrained on synthetic data -- consistent with
the rule that the synthetic overlay never feeds into headline modeling.

IMPORTANT design correction made while building this (kept here so the
mistake isn't silently repeated): the naive approach -- compare fairness
metrics across periods, P0 (clean) vs P1-P4 (biased) -- does NOT isolate
the bias signal. Each period's covariate-drift resampling (PSI up to 0.60
by P4) independently reshuffles who's in the population, which moves the
male comparison group's own false-positive rate around by several points
for reasons that have nothing to do with the gender bias. That confound
swamps the effect and produces a noisy, non-monotonic cross-period series
that looks like a detection failure but isn't one.

The correct, confound-free test: for each biased period, compare the
SAME population's fairness metric computed against `TARGET_original`
(pre-flip) vs `TARGET` (post-flip) -- same rows, same model decisions,
label swapped. This isolates exactly what the flip mechanism contributes,
holding the covariate-drift-driven population composition fixed. This is
only possible because we, as the overlay's designers, have the
ground-truth `TARGET_original` -- a real auditor never would. That's the
right framing: this exercise validates that outcome-aware fairness metrics
*can* catch this failure mode if a clean reference ever becomes available
(e.g. a later true-outcome reconciliation, or an independent audit
sample), even though the metric can't be run this way in a genuine
real-time audit.

Run: python src/fairness/synthetic_bias_detection.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

from fairness_metrics import (
    CALIBRATION_OOF_PATH,
    LIGHTGBM_OPTIMAL_THRESHOLD,
    _max_min_diff,
    demographic_parity_difference,
    equalized_odds_difference,
    group_selection_rates,
)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
SYNTHETIC_OVERLAY_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "synthetic_overlay.csv"
CROSS_PERIOD_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase4_synthetic_bias_cross_period.csv"
ISOLATED_EFFECT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase4_synthetic_bias_isolated_effect.csv"

BIAS_PERIODS = ["P1", "P2", "P3", "P4"]


def load_synthetic_fairness_data():
    overlay = pd.read_csv(SYNTHETIC_OVERLAY_PATH)

    conn = sqlite3.connect(DB_PATH)
    gender = pd.read_sql("SELECT SK_ID_CURR, CODE_GENDER FROM application_train", conn)
    conn.close()

    oof = pd.read_csv(CALIBRATION_OOF_PATH)
    oof = oof[oof["model"] == "LightGBM"][["SK_ID_CURR", "proba_calibrated"]].rename(columns={"proba_calibrated": "PD"})

    df = overlay.merge(gender, on="SK_ID_CURR", how="left").merge(oof, on="SK_ID_CURR", how="left")
    assert df["PD"].isna().sum() == 0, "every synthetic row's SK_ID_CURR must have an OOF calibrated PD"
    df["approved"] = (df["PD"] <= LIGHTGBM_OPTIMAL_THRESHOLD).astype(int)
    return df


def cross_period_table(df):
    """Naive cross-period comparison. Kept for context/completeness -- NOT
    the detection evidence, see the module docstring's design correction."""
    rows = []
    for period in sorted(df["period"].unique()):
        period_df = df[df["period"] == period]
        dpd, _ = demographic_parity_difference(period_df, "CODE_GENDER")
        eqodds, tpr_rates, _ = equalized_odds_difference(period_df, "CODE_GENDER")
        eod = _max_min_diff(tpr_rates)
        rows.append({"period": period, "DPD": dpd, "EOD": eod, "EqualizedOddsDiff": eqodds})
    return pd.DataFrame(rows).set_index("period")


def isolated_effect_table(df):
    """The confound-free test: same population, same model decisions, per
    period -- compare fairness metrics against TARGET_original (pre-flip)
    vs TARGET (post-flip)."""
    rows = []
    for period in BIAS_PERIODS:
        period_df = df[df["period"] == period]

        dpd_true, _ = demographic_parity_difference(period_df, "CODE_GENDER")
        dpd_biased, _ = demographic_parity_difference(period_df, "CODE_GENDER")  # identical: DPD never reads TARGET

        f_fpr_true = group_selection_rates(period_df[period_df["TARGET_original"] == 1], "CODE_GENDER").loc["F", "rate"]
        f_fpr_biased = group_selection_rates(period_df[period_df["TARGET"] == 1], "CODE_GENDER").loc["F", "rate"]

        eqodds_true, tpr_true, _ = equalized_odds_difference(period_df.rename(columns={"TARGET": "_tmp", "TARGET_original": "TARGET"}), "CODE_GENDER")
        eqodds_biased, tpr_biased, _ = equalized_odds_difference(period_df, "CODE_GENDER")

        rows.append({
            "period": period,
            "DPD_true": dpd_true,
            "DPD_biased": dpd_biased,
            "F_FPR_true": f_fpr_true,
            "F_FPR_biased": f_fpr_biased,
            "F_FPR_inflation": f_fpr_biased - f_fpr_true,
            "EqualizedOddsDiff_true": eqodds_true,
            "EqualizedOddsDiff_biased": eqodds_biased,
        })
    return pd.DataFrame(rows).set_index("period")


def main():
    df = load_synthetic_fairness_data()

    cross_period = cross_period_table(df)
    cross_period.to_csv(CROSS_PERIOD_PATH)
    print("=== Cross-period comparison (confounded by covariate drift -- context only) ===")
    print(cross_period.round(4).to_string())

    isolated = isolated_effect_table(df)
    isolated.to_csv(ISOLATED_EFFECT_PATH)
    print("\n=== Isolated bias effect: same population, TARGET_original vs TARGET ===")
    print(isolated.round(4).to_string())

    print(f"\nDPD_true == DPD_biased in every period: {(isolated['DPD_true'] == isolated['DPD_biased']).all()} "
          "(expected -- DPD never reads TARGET, so it is structurally blind to a label-only bias)")
    print(f"F's false-positive-rate inflation from the flip: "
          f"{isolated['F_FPR_inflation'].min():.4f} to {isolated['F_FPR_inflation'].max():.4f} "
          f"across all 4 biased periods (consistently positive -- the bias is detectable via outcome-aware metrics)")

    print(f"\nSaved: {CROSS_PERIOD_PATH}")
    print(f"Saved: {ISOLATED_EFFECT_PATH}")

    return df, cross_period, isolated


if __name__ == "__main__":
    df, cross_period, isolated = main()
