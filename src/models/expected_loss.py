"""Phase 3 Expected Loss + cost-sensitive threshold selection, for all 4
baseline models (per the Phase 3 exit criteria requiring EL at optimal
threshold across the full model comparison table).

Expected Loss (EL) = PD x LGD x EAD, computed per applicant per model from
the out-of-fold calibrated probabilities produced by
src/models/calibration.py (data/processed/phase3_calibration_oof.csv). EL
itself is a forward-looking, model-implied risk estimate; it is reported
per applicant but the threshold search below is a separate, standard
exercise: a retrospective backtest against the *realized* historical
outcome (TARGET), since that's what's actually available to evaluate "what
would our profit have been under policy X" on this dataset.

Assumptions (Home Credit provides neither LGD nor EAD directly -- both are
simulated at plausible fixed values, documented here rather than tuned):

- EAD (Exposure at Default) = AMT_CREDIT, the originated credit amount from
  `application_train`. Simplifying assumption: these are unsecured
  installment/cash loans, and we assume default can occur before
  meaningful principal is repaid, so the full disbursed amount is treated
  as the exposure. In reality EAD would decay over the loan's term as
  installments are paid down; this is the conservative (upper-bound) case.
- LGD (Loss Given Default) = 0.55, fixed across all loans and all models.
  This sits in the middle of commonly-cited unsecured-retail-lending LGD
  benchmarks (roughly 0.45-0.75 depending on collections effectiveness and
  collateral -- these are unsecured consumer loans, so no recovery from
  collateral, but partial recovery via collections/write-off proceeds is
  assumed). A single fixed value is a simplification; a real deployment
  would model LGD per segment or as its own predictive model.
- MARGIN_RATE = 0.15: assumed net interest/fee margin earned on a fully
  repaid loan, as a fraction of AMT_CREDIT (net of funding cost and
  operating expense, i.e. this is intended as *profit* margin, not gross
  interest rate). Needed to define "expected profit" for the threshold
  search -- the EL formula alone only defines the cost side. Also
  simulated/assumed, not fit from data (Home Credit doesn't provide
  realized loan-level profit).

Cost-sensitive threshold convention: `threshold` is the maximum acceptable
predicted default probability (PD) for approval -- a loan is approved iff
PD <= threshold. So threshold=0 rejects (almost) everyone and threshold=1
approves everyone; profit is checked at both ends as a sanity check before
trusting each model's interior optimum.

For each threshold, retrospective profit over approved loans is:
    sum( (1 - TARGET) * MARGIN_RATE * EAD - TARGET * LGD * EAD )
i.e. approved loans that didn't actually default earn the margin; approved
loans that did default cost LGD * EAD. Rejected loans contribute 0 (no
loan issued, no opportunity-cost revenue assumed).

Run: python src/models/expected_loss.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline_models import MODEL_FACTORIES

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_calibration_oof.csv"
PROFIT_CURVE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_profit_curves.csv"
EL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_expected_loss.csv"
OPTIMA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_optimal_thresholds.csv"

LGD_ASSUMPTION = 0.55
MARGIN_RATE = 0.15
N_THRESHOLDS = 101


def load_pd_and_ead():
    calib = pd.read_csv(CALIBRATION_PATH)
    conn = sqlite3.connect(DB_PATH)
    ead = pd.read_sql("SELECT SK_ID_CURR, AMT_CREDIT AS EAD FROM application_train", conn)
    conn.close()

    df = calib.merge(ead, on="SK_ID_CURR", how="left")
    assert df["EAD"].isna().sum() == 0, "every SK_ID_CURR in the calibration set must have an AMT_CREDIT"
    df = df.rename(columns={"proba_calibrated": "PD", "y_true": "TARGET"})
    return df[["model", "SK_ID_CURR", "PD", "TARGET", "EAD"]]


def compute_expected_loss(df):
    df = df.copy()
    df["LGD"] = LGD_ASSUMPTION
    df["expected_loss"] = df["PD"] * df["LGD"] * df["EAD"]
    return df


def profit_at_threshold(df, threshold):
    approved = df[df["PD"] <= threshold]
    profit = ((1 - approved["TARGET"]) * MARGIN_RATE * approved["EAD"] - approved["TARGET"] * LGD_ASSUMPTION * approved["EAD"]).sum()
    approval_rate = len(approved) / len(df)
    approved_default_rate = approved["TARGET"].mean() if len(approved) > 0 else np.nan
    return profit, approval_rate, approved_default_rate


def build_profit_curve(df, n_thresholds=N_THRESHOLDS):
    thresholds = np.linspace(0, 1, n_thresholds)
    rows = []
    for t in thresholds:
        profit, approval_rate, approved_default_rate = profit_at_threshold(df, t)
        rows.append({
            "threshold": t,
            "profit": profit,
            "approval_rate": approval_rate,
            "approved_default_rate": approved_default_rate,
        })
    return pd.DataFrame(rows)


def main():
    all_df = load_pd_and_ead()
    all_df = compute_expected_loss(all_df)
    all_df.to_csv(EL_PATH, index=False)
    print(f"Expected Loss per applicant per model saved: {EL_PATH} ({len(all_df):,} rows)")

    curves = []
    optima = []
    for model_name in MODEL_FACTORIES.keys():
        df = all_df[all_df["model"] == model_name]
        print(f"\n=== {model_name} (total portfolio EL = {df['expected_loss'].sum():,.0f}) ===")

        curve = build_profit_curve(df)
        curve.insert(0, "model", model_name)
        curves.append(curve)

        at_zero = curve.iloc[0]
        at_one = curve.iloc[-1]
        print(f"threshold=0.00: approval_rate={at_zero['approval_rate']:.4f}  profit={at_zero['profit']:,.0f}")
        print(f"threshold=1.00: approval_rate={at_one['approval_rate']:.4f}  profit={at_one['profit']:,.0f}")

        best = curve.loc[curve["profit"].idxmax()]
        print(f"Optimal threshold = {best['threshold']:.3f}  profit={best['profit']:,.0f}  "
              f"approval_rate={best['approval_rate']:.4f}  approved_default_rate={best['approved_default_rate']:.4f}")
        optima.append(best)

    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(PROFIT_CURVE_PATH, index=False)
    print(f"\nSaved profit curves: {PROFIT_CURVE_PATH}")

    optima_df = pd.DataFrame(optima).set_index("model").reindex(MODEL_FACTORIES.keys())
    optima_df.to_csv(OPTIMA_PATH)
    print(f"Saved optimal thresholds: {OPTIMA_PATH}")
    print(optima_df.round(4).to_string())

    return all_df, all_curves, optima_df


if __name__ == "__main__":
    all_df, all_curves, optima_df = main()
