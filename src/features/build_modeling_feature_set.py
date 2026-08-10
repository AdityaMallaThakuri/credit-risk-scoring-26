"""Build the curated modeling feature set: final_feature_table +
engineered_features, minus the 3 columns confirmed redundant by the VIF
check (see src/features/vif_check.py output, reviewed and signed off).

Nothing is deleted from the underlying tables -- final_feature_table and
engineered_features stay intact as the full audit trail of what the SQL/
Python pipeline computed. This script only curates a narrower table for
actual modeling.

Drops (each verified empirically before dropping, not assumed from names):
- installments_avg_repayment_velocity: exact duplicate of repayment_velocity
  (VIF=inf on both). Kept `repayment_velocity` as the canonical name.
- bureau_total_credit_count: = bureau_active_credit_count +
  bureau_closed_credit_count for 98.0% of rows exactly (VIF 1522/731/235).
  Kept the active/closed split -- same information, plus the split for free.
- days_since_last_late_payment: r=0.934 with delinquency_recency_days,
  which is min(days_since_last_late_payment, bureau_months_since_last_
  delinquency*30) by construction -- a strict superset of its information.
  Kept the unified delinquency_recency_days.

NOT dropped here, left for later phases:
- has_bureau_history / has_previous_application / has_pos_cash_history /
  has_credit_card_history: each is an exact deterministic function of
  "is the paired count/months column non-null" -- but they become the only
  way to distinguish "genuinely zero" from "missing" once NaNs are
  imputed in Phase 3. Dropping now, before an imputation strategy exists,
  would destroy that distinction. Revisit at Phase 3.
- The remaining VIF-flagged columns (installments_count, pos_months_of_
  history, cc_months_of_history, pos_completion_trend, prev_app_approved_
  rate, HOUR_APPR_PROCESS_START, bureau_avg_days_credit) reflect a shared
  "history depth" latent factor across several columns at once, not
  pairwise duplication -- dropping them would lose real signal. Handle
  via L2/Ridge regularization on the Logistic Regression model instead
  (Phase 3); tree models are unaffected by this kind of collinearity.

Run: python src/features/build_modeling_feature_set.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"

DROP_COLUMNS = [
    "installments_avg_repayment_velocity",
    "bureau_total_credit_count",
    "days_since_last_late_payment",
]


def main():
    conn = sqlite3.connect(DB_PATH)

    final = pd.read_sql("SELECT * FROM final_feature_table", conn)
    engineered = pd.read_sql(
        "SELECT SK_ID_CURR, dti_ratio, repayment_velocity, delinquency_recency_days, "
        "spending_volatility, dti_utilization_interaction FROM engineered_features",
        conn,
    )
    df = final.merge(engineered, on="SK_ID_CURR", how="left")

    before_cols = set(df.columns)
    df = df.drop(columns=DROP_COLUMNS)
    dropped = before_cols - set(df.columns)
    assert dropped == set(DROP_COLUMNS), f"expected to drop exactly {DROP_COLUMNS}, actually dropped {dropped}"

    df.to_sql("modeling_feature_set", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_modeling_feature_set_sk_id_curr ON modeling_feature_set (SK_ID_CURR)")
    conn.commit()

    print(f"modeling_feature_set: {len(df):,} rows x {len(df.columns)} columns (dropped {len(DROP_COLUMNS)}: {DROP_COLUMNS})")
    conn.close()
    return df


if __name__ == "__main__":
    main()
