"""Phase 4 fairness metrics: Demographic Parity Difference (DPD), Equal
Opportunity Difference (EOD), and Equalized Odds Difference, computed on
the Phase 3 winning model (LightGBM) using its out-of-fold calibrated PD
(data/processed/phase3_calibration_oof.csv) -- i.e. genuine held-out
predictions, not in-sample ones -- joined to real demographic attributes
from `application_train` (CODE_GENDER, NAME_FAMILY_STATUS, an age band
derived from DAYS_BIRTH). None of these attributes are in
`modeling_feature_set` / were used to train the model; they're pulled in
here only for post-hoc fairness auditing, which is the standard, correct
place for them.

Decision rule: approved = 1 if PD <= 0.220 (LightGBM's Phase 3
profit-optimal threshold from src/models/expected_loss.py). "Approved" is
treated as the favorable outcome, and TARGET==0 (actually repaid) as the
favorable ground-truth label, matching standard fairness-in-lending
convention:

- DPD  (Demographic Parity Difference): spread (max - min) of the approval
  rate P(approved=1) across the attribute's groups. Does the model approve
  people at different rates by group, full stop -- independent of whether
  those decisions are "correct."
- EOD  (Equal Opportunity Difference): spread of the *true positive rate*
  P(approved=1 | TARGET=0) across groups -- among applicants who actually
  would have repaid, are they approved at the same rate regardless of
  group?
- Equalized Odds Difference: max(EOD, spread of the *false positive rate*
  P(approved=1 | TARGET=1) across groups) -- EOD's requirement plus: among
  applicants who actually would have defaulted, is the (mistaken) approval
  rate also equal across groups?

For attributes with more than 2 categories (family status, age band), all
three metrics generalize as max-min across every category present with a
non-trivial sample size (n < 500 groups are reported but flagged, per the
project's stress-test concern about fairness metrics getting unreliable on
small subgroups).

Run: python src/fairness/fairness_metrics.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
CALIBRATION_OOF_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_calibration_oof.csv"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase4_real_fairness_metrics.csv"

LIGHTGBM_OPTIMAL_THRESHOLD = 0.220
SMALL_GROUP_WARNING_N = 500

AGE_BAND_EDGES = [18, 25, 35, 45, 55, 65, 100]
AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def load_real_fairness_data():
    conn = sqlite3.connect(DB_PATH)
    demo = pd.read_sql(
        "SELECT SK_ID_CURR, CODE_GENDER, NAME_FAMILY_STATUS, DAYS_BIRTH, AMT_CREDIT AS EAD FROM application_train",
        conn,
    )
    conn.close()

    oof = pd.read_csv(CALIBRATION_OOF_PATH)
    oof = oof[oof["model"] == "LightGBM"].rename(columns={"proba_calibrated": "PD", "y_true": "TARGET"})

    df = oof.merge(demo, on="SK_ID_CURR", how="left")
    df["age_years"] = -df["DAYS_BIRTH"] / 365.25
    df["age_band"] = pd.cut(df["age_years"], bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=False)
    df["approved"] = (df["PD"] <= LIGHTGBM_OPTIMAL_THRESHOLD).astype(int)
    return df


def group_selection_rates(df, group_col, decision_col="approved"):
    return df.groupby(group_col, observed=True)[decision_col].agg(["mean", "count"]).rename(columns={"mean": "rate", "count": "n"})


def _max_min_diff(rates, min_group_n=SMALL_GROUP_WARNING_N):
    """max - min of `rate`, excluding groups with n < min_group_n so a
    handful of rows (e.g. CODE_GENDER's 4-row XNA group) can't dominate the
    headline gap by chance. The full (unfiltered) table is still returned
    for display."""
    reliable = rates[rates["n"] >= min_group_n]
    if len(reliable) < 2:
        return float("nan")
    return float(reliable["rate"].max() - reliable["rate"].min())


def demographic_parity_difference(df, group_col, decision_col="approved", min_group_n=SMALL_GROUP_WARNING_N):
    rates = group_selection_rates(df, group_col, decision_col)
    return _max_min_diff(rates, min_group_n), rates


def equal_opportunity_difference(df, group_col, decision_col="approved", target_col="TARGET", min_group_n=SMALL_GROUP_WARNING_N):
    actual_good = df[df[target_col] == 0]
    rates = group_selection_rates(actual_good, group_col, decision_col)
    return _max_min_diff(rates, min_group_n), rates


def false_positive_rate_difference(df, group_col, decision_col="approved", target_col="TARGET", min_group_n=SMALL_GROUP_WARNING_N):
    actual_bad = df[df[target_col] == 1]
    rates = group_selection_rates(actual_bad, group_col, decision_col)
    return _max_min_diff(rates, min_group_n), rates


def equalized_odds_difference(df, group_col, decision_col="approved", target_col="TARGET", min_group_n=SMALL_GROUP_WARNING_N):
    eod, tpr_rates = equal_opportunity_difference(df, group_col, decision_col, target_col, min_group_n)
    fpr_diff, fpr_rates = false_positive_rate_difference(df, group_col, decision_col, target_col, min_group_n)
    combined = np.nanmax([eod, fpr_diff]) if not (np.isnan(eod) and np.isnan(fpr_diff)) else float("nan")
    return float(combined), tpr_rates, fpr_rates


def summarize_attribute(df, group_col, decision_col="approved", target_col="TARGET", min_group_n=SMALL_GROUP_WARNING_N):
    dpd, selection_rates = demographic_parity_difference(df, group_col, decision_col, min_group_n)
    eqodds, tpr_rates, fpr_rates = equalized_odds_difference(df, group_col, decision_col, target_col, min_group_n)
    eod = _max_min_diff(tpr_rates, min_group_n)

    small_groups = selection_rates[selection_rates["n"] < SMALL_GROUP_WARNING_N]
    if len(small_groups) > 0:
        print(f"  WARNING: small subgroup(s) for {group_col} (n < {SMALL_GROUP_WARNING_N}), metric may be unreliable: "
              f"{dict(small_groups['n'])}")

    return {
        "attribute": group_col,
        "DPD": dpd,
        "EOD": eod,
        "EqualizedOddsDiff": eqodds,
    }, selection_rates, tpr_rates, fpr_rates


def main():
    df = load_real_fairness_data()

    rows = []
    for group_col in ["CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"]:
        summary, selection_rates, tpr_rates, fpr_rates = summarize_attribute(df, group_col)
        rows.append(summary)
        print(f"\n=== {group_col} ===")
        print("Approval rate by group:")
        print(selection_rates.round(4).to_string())
        print(f"DPD={summary['DPD']:.4f}  EOD={summary['EOD']:.4f}  EqualizedOddsDiff={summary['EqualizedOddsDiff']:.4f}")

    result = pd.DataFrame(rows).set_index("attribute")
    result.to_csv(RESULTS_PATH)
    print(f"\nSaved: {RESULTS_PATH}")
    return df, result


if __name__ == "__main__":
    df, result = main()
