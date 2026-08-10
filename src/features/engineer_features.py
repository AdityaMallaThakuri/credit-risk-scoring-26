"""Engineer DTI, repayment velocity, delinquency recency, spending
volatility, and a ratio-stacked interaction feature, on top of
sql/final_feature_table (built by run_sql_pipeline.py).

Per CLAUDE.md ("one shared feature-engineering pipeline -- do not
duplicate preprocessing logic per model"): repayment velocity, delinquency
recency (per-source), and instalment-amount volatility are already
computed once in the SQL layer (sql/01_bureau_features.sql,
03_installments_features.sql) -- this script REUSES those columns rather
than recomputing them, and only adds what doesn't already exist:
DTI, a unified cross-source delinquency-recency figure, credit-card-
drawings spending volatility, and the DTI x utilization interaction.

Run: python src/features/engineer_features.py
(requires data/processed/credit_risk.db to already exist -- run
run_sql_pipeline.py first)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"


def load_base(conn):
    """final_feature_table (SQL-engineered) + the raw application-level
    amounts needed for DTI. AMT_ANNUITY/AMT_INCOME_TOTAL/AMT_CREDIT are the
    applicant's own stated income and the loan terms being requested --
    part of the current application itself, known before the decision."""
    return pd.read_sql(
        """
        SELECT f.*, a.AMT_ANNUITY, a.AMT_INCOME_TOTAL, a.AMT_CREDIT
        FROM final_feature_table f
        JOIN application_train a ON a.SK_ID_CURR = f.SK_ID_CURR
        """,
        conn,
    )


def compute_dti(df):
    """DTI = AMT_ANNUITY / AMT_INCOME_TOTAL. Both are annual in Home
    Credit's schema, so no unit conversion is needed -- the ratio is scale-
    invariant regardless (a /12 on both sides would cancel out anyway)."""
    return np.where(df["AMT_INCOME_TOTAL"] > 0, df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"], np.nan)


def compute_delinquency_recency(df):
    """Unified 'how recently did something go wrong' signal, blending the
    two source-specific recency columns already computed in SQL:
    bureau_months_since_last_delinquency (bureau_balance, in months) and
    days_since_last_late_payment (installments_payments, in days).
    Converts months->days (~30/month) and takes whichever source reports
    the MORE recent event (the smaller distance); NaN if neither exists."""
    bureau_days = df["bureau_months_since_last_delinquency"] * 30
    installments_days = df["days_since_last_late_payment"]
    return pd.concat([bureau_days, installments_days], axis=1).min(axis=1, skipna=True)


def compute_spending_volatility(conn):
    """Standard deviation of monthly credit-card spending (AMT_DRAWINGS_CURRENT),
    per card, averaged across a client's cards. Distinct from the SQL layer's
    installments_amt_volatility (which is repayment-amount volatility, not
    spending) -- this is the reading_material.md 2.4 definition: 'erratic
    spenders may be riskier even with the same average spend.'
    MONTHS_BALANCE in credit_card_balance is verified <= -1 for all rows
    (checked before the SQL pipeline was built) -- entirely historical.
    Re-checked here rather than just trusted, since this query is new and
    doesn't itself filter on MONTHS_BALANCE."""
    max_months_balance = conn.execute("SELECT MAX(MONTHS_BALANCE) FROM credit_card_balance").fetchone()[0]
    assert max_months_balance is not None and max_months_balance <= 0, (
        f"credit_card_balance has MONTHS_BALANCE > 0 (max={max_months_balance}) -- "
        "would mean future-relative rows exist; spending_volatility would leak."
    )
    cc = pd.read_sql(
        "SELECT SK_ID_CURR, SK_ID_PREV, AMT_DRAWINGS_CURRENT FROM credit_card_balance",
        conn,
    )
    per_card_std = cc.groupby(["SK_ID_CURR", "SK_ID_PREV"])["AMT_DRAWINGS_CURRENT"].std()
    return per_card_std.groupby("SK_ID_CURR").mean().rename("spending_volatility")


def compute_ratio_stack(df):
    """DTI x bureau credit utilization. bureau_credit_utilization is used
    over cc_avg_utilization for the utilization side of the interaction --
    it covers 85.7% of applicants vs. credit-card utilization's 28.3% (not
    everyone has a card), so the interaction stays defined for far more
    applicants than a card-utilization-based version would."""
    return df["dti_ratio"] * df["bureau_credit_utilization"]


def main():
    conn = sqlite3.connect(DB_PATH)

    df = load_base(conn)
    df["dti_ratio"] = compute_dti(df)

    # already computed once in SQL -- reused, not recomputed
    df["repayment_velocity"] = df["installments_avg_repayment_velocity"]

    df["delinquency_recency_days"] = compute_delinquency_recency(df)

    spending_vol = compute_spending_volatility(conn)
    df = df.merge(spending_vol, on="SK_ID_CURR", how="left")

    df["dti_utilization_interaction"] = compute_ratio_stack(df)

    new_cols = [
        "SK_ID_CURR", "TARGET", "dti_ratio", "repayment_velocity",
        "delinquency_recency_days", "spending_volatility", "dti_utilization_interaction",
    ]
    out = df[new_cols]
    out.to_sql("engineered_features", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_engineered_features_sk_id_curr ON engineered_features (SK_ID_CURR)")
    conn.commit()

    print(f"engineered_features: {len(out):,} rows x {len(new_cols)} columns")
    print()
    print("=== Coverage / null rates ===")
    for col in new_cols[2:]:
        n_null = out[col].isna().sum()
        print(f"{col:32s} null={n_null:,}/{len(out):,} ({100*n_null/len(out):.1f}%)")
    print()
    print("=== Value ranges ===")
    print(out[new_cols[2:]].describe().T[["min", "50%", "max"]])

    conn.close()
    return out


if __name__ == "__main__":
    main()
