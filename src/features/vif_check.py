"""VIF (multicollinearity) check on the final numeric feature set --
final_feature_table (SQL) joined with engineered_features (Python).

Report only -- doesn't drop or transform anything. Per docs/roadmap.md
Phase 2 exit criteria: "run a multicollinearity check (VIF) on the final
feature set."

VIF needs a complete-case (no-NaN) matrix. Median imputation here is used
ONLY to make the VIF computation possible -- it is not the project's real
imputation strategy (that belongs in Phase 3's train-only fit/transform
step, per CLAUDE.md's "fit scalers/imputers on training data only" rule).
Categorical/flag/text columns are excluded; identifiers and TARGET are
excluded from the design matrix.

Run: python src/features/vif_check.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"

EXCLUDE_COLS = {
    "SK_ID_CURR", "TARGET",
    "WEEKDAY_APPR_PROCESS_START", "prev_app_last_status",  # categorical/text
}


def load_full_feature_set(conn):
    final = pd.read_sql("SELECT * FROM final_feature_table", conn)
    engineered = pd.read_sql(
        "SELECT SK_ID_CURR, dti_ratio, repayment_velocity, delinquency_recency_days, "
        "spending_volatility, dti_utilization_interaction FROM engineered_features",
        conn,
    )
    return final.merge(engineered, on="SK_ID_CURR", how="left")


def compute_vif(df, numeric_cols):
    X = df[numeric_cols].copy()
    # median imputation for VIF computation only -- see module docstring
    X = X.fillna(X.median())
    # drop any column that's still constant/all-NaN after imputation -- VIF is undefined for it
    X = X.loc[:, X.std() > 0]

    vif_data = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": col, "VIF": vif})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)


def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_full_feature_set(conn)
    conn.close()

    numeric_cols = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Computing VIF over {len(numeric_cols)} numeric features, n={len(df):,} rows (median-imputed for this check only)\n")

    vif_df = compute_vif(df, numeric_cols)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(vif_df.to_string(index=False))

    flagged = vif_df[vif_df["VIF"] > 10]
    print(f"\n=== Flagged (VIF > 10): {len(flagged)} features ===")
    print(flagged.to_string(index=False))

    return vif_df


if __name__ == "__main__":
    main()
