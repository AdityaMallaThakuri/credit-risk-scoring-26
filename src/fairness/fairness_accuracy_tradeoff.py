"""Phase 4 fairness-accuracy trade-off report (CLAUDE.md's Objective 5
deliverable): for a handful of threshold/mitigation choices on a given real
attribute (LightGBM's OOF calibrated PD), report the accuracy/EL cost
against the fairness gain, all in one table.

Primary attribute: age_band. CODE_GENDER was tried first but its real-data
disparity is already close to parity (DPD=0.0010, from
src/fairness/fairness_metrics.py) -- there's no real gap for a mitigation
strategy to close, so that table's "gains" were noise-sized, not a
meaningful trade-off (see data/processed/phase4_fairness_accuracy_tradeoff_CODE_GENDER.csv,
kept for the record as a legitimate null result). age_band has a
substantial natural disparity (DPD=0.0565, EqualizedOddsDiff=0.0660),
giving the mitigation strategies something real to trade off against.

Mitigation strategies compared (all on the same real, unbiased data --
this is a separate exercise from src/fairness/synthetic_bias_detection.py,
which validates detection on the *synthetic* planted bias):

- baseline: Phase 3's single profit-optimal threshold (0.220), applied
  uniformly to every applicant regardless of group. No fairness
  intervention.
- equalize_dp: per-group thresholds (one cutoff per age band) chosen so
  every group has the SAME approval rate as the baseline's overall
  approval rate -- drives DPD to ~0 by construction. Standard post-hoc
  threshold-optimization mitigation (Hardt et al. 2016 style), not a new
  model.
- equalize_eo: per-group thresholds chosen so every group has the SAME
  true positive rate (approval rate among applicants who actually repaid)
  as the baseline's overall TPR -- drives EOD to ~0 by construction.
  Deliberately included alongside equalize_dp to demonstrate the standard
  fairness-literature impossibility result empirically: equalizing one
  criterion does not generally equalize the other (Kleinberg et al. /
  Chouldechova) -- expect equalize_dp to still show a nonzero EOD gap, and
  vice versa.
- blunt_stricter: a single uniform threshold lowered to 0.15 (naive
  "reject more people to be safe" mitigation, no group-awareness at all)
  -- included as a baseline for how much fairness a group-blind
  intervention buys per unit of profit given up, for comparison against
  the two targeted strategies above.

AUC is reported but is identical across all rows by construction -- it's a
ranking metric over a fixed PD column, unaffected by where the decision
threshold is drawn. It's included because it was asked for and because
"AUC doesn't change with the threshold" is itself informative: the real
accuracy cost of a threshold-based mitigation shows up in approval rate /
EL / profit, not in AUC. This is worth calling out explicitly, not a
subtle omission.

Groups with n < fairness_metrics.SMALL_GROUP_WARNING_N are excluded from
the reported DPD/EOD/EqualizedOddsDiff max-min spread (handled inside
fairness_metrics' difference functions), consistent with the small-sample
warning elsewhere in Phase 4.

Run: python src/fairness/fairness_accuracy_tradeoff.py [attribute]
  attribute defaults to age_band; pass CODE_GENDER or NAME_FAMILY_STATUS
  to reproduce the trade-off table for either of those instead.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from fairness_metrics import (
    LIGHTGBM_OPTIMAL_THRESHOLD,
    _max_min_diff,
    demographic_parity_difference,
    equalized_odds_difference,
    load_real_fairness_data,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

LGD_ASSUMPTION = 0.55
MARGIN_RATE = 0.15
BLUNT_STRICTER_THRESHOLD = 0.150
DEFAULT_ATTRIBUTE = "age_band"


def profit_and_el(df, approved):
    approved_df = df[approved]
    profit = ((1 - approved_df["TARGET"]) * MARGIN_RATE * approved_df["EAD"] - approved_df["TARGET"] * LGD_ASSUMPTION * approved_df["EAD"]).sum()
    el_approved = (approved_df["PD"] * LGD_ASSUMPTION * approved_df["EAD"]).sum()
    return profit, el_approved


def group_thresholds_for_target_rate(df, group_col, target_rate, subset_mask=None):
    """For each group, find the PD quantile such that target_rate of that
    group's PD (restricted to subset_mask if given) falls at or below it."""
    base = df if subset_mask is None else df[subset_mask]
    return base.groupby(group_col, observed=True)["PD"].quantile(target_rate).to_dict()


def apply_group_thresholds(df, group_col, thresholds):
    group_threshold = df[group_col].astype(object).map(thresholds).astype(float)
    return (df["PD"] <= group_threshold).values


def run_tradeoff(group_col=DEFAULT_ATTRIBUTE):
    df = load_real_fairness_data()
    if group_col == "CODE_GENDER":
        df = df[df["CODE_GENDER"].isin(["M", "F"])].copy()

    auc = roc_auc_score(df["TARGET"], -df["PD"])  # lower PD = more "positive"/favorable

    baseline_approved = (df["PD"] <= LIGHTGBM_OPTIMAL_THRESHOLD).values
    baseline_approval_rate = baseline_approved.mean()
    baseline_tpr = df.loc[df["TARGET"] == 0, "PD"].le(LIGHTGBM_OPTIMAL_THRESHOLD).mean()

    dp_thresholds = group_thresholds_for_target_rate(df, group_col, baseline_approval_rate)
    eo_thresholds = group_thresholds_for_target_rate(df, group_col, baseline_tpr, subset_mask=(df["TARGET"] == 0))

    strategies = {
        "baseline": baseline_approved,
        "equalize_dp": apply_group_thresholds(df, group_col, dp_thresholds),
        "equalize_eo": apply_group_thresholds(df, group_col, eo_thresholds),
        "blunt_stricter": (df["PD"] <= BLUNT_STRICTER_THRESHOLD).values,
    }

    rows = []
    for name, approved in strategies.items():
        d = df.copy()
        d["approved"] = approved.astype(int)

        dpd, _ = demographic_parity_difference(d, group_col)
        eqodds, tpr_rates, _ = equalized_odds_difference(d, group_col)
        eod = _max_min_diff(tpr_rates)
        profit, el_approved = profit_and_el(d, approved)

        rows.append({
            "strategy": name,
            "auc": auc,
            "approval_rate": approved.mean(),
            "profit": profit,
            "el_approved": el_approved,
            "DPD": dpd,
            "EOD": eod,
            "EqualizedOddsDiff": eqodds,
        })
        print(f"{name:<16} approval_rate={approved.mean():.4f}  profit={profit:,.0f}  "
              f"EL_approved={el_approved:,.0f}  DPD={dpd:.4f}  EOD={eod:.4f}  EqOdds={eqodds:.4f}")

    result = pd.DataFrame(rows).set_index("strategy")
    results_path = RESULTS_DIR / f"phase4_fairness_accuracy_tradeoff_{group_col}.csv"
    result.to_csv(results_path)
    print(f"\nSaved: {results_path}")

    baseline_row = result.loc["baseline"]
    for name in ["equalize_dp", "equalize_eo", "blunt_stricter"]:
        row = result.loc[name]
        print(f"\n{name} vs baseline: "
              f"profit cost = {baseline_row['profit'] - row['profit']:,.0f} "
              f"({(baseline_row['profit'] - row['profit']) / baseline_row['profit'] * 100:.2f}%), "
              f"DPD gain = {baseline_row['DPD'] - row['DPD']:.4f}, "
              f"EOD gain = {baseline_row['EOD'] - row['EOD']:.4f}")

    return df, result


if __name__ == "__main__":
    attribute = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ATTRIBUTE
    df, result = run_tradeoff(attribute)
