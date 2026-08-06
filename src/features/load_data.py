"""Load the 7 raw Home Credit CSVs and report basic shape + the DAYS_EMPLOYED anomaly.

Read-only against data/raw/ (never write there). No fixes applied here —
reporting only, per project convention (see docs/roadmap.md Phase 1).
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

RAW_FILES = {
    "application_train": "HC_application_train.csv",
    "bureau": "HC_bureau.csv",
    "bureau_balance": "HC_bureau_balance.csv",
    "credit_card_balance": "HC_credit_card_balance.csv",
    "installments_payments": "HC_installments_payments.csv",
    "pos_cash_balance": "HC_POS_CASH_balance.csv",
    "previous_application": "HC_previous_application.csv",
}

DAYS_EMPLOYED_SENTINEL = 365243


def load_raw_tables():
    tables = {}
    for name, filename in RAW_FILES.items():
        tables[name] = pd.read_csv(RAW_DIR / filename)
    return tables


def report_shapes(tables):
    print("=== Row / column counts ===")
    for name, df in tables.items():
        print(f"{name:24s} rows={df.shape[0]:>9,}  cols={df.shape[1]:>3}")


def report_days_employed_anomaly(tables):
    print("\n=== DAYS_EMPLOYED sentinel-value check (365243) ===")
    for name, df in tables.items():
        if "DAYS_EMPLOYED" not in df.columns:
            continue
        total = len(df)
        anomaly_count = int((df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum())
        pct = 100 * anomaly_count / total if total else 0.0
        print(f"{name}: {anomaly_count:,} / {total:,} rows ({pct:.2f}%) have DAYS_EMPLOYED == {DAYS_EMPLOYED_SENTINEL}")


def handle_days_employed_anomaly(df):
    """Flag DAYS_EMPLOYED sentinel rows (365243, i.e. pensioners/unemployed) and
    null out the placeholder so it's not treated as a real tenure value."""
    df = df.copy()
    df["is_pensioner_flag"] = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df.loc[df["is_pensioner_flag"], "DAYS_EMPLOYED"] = pd.NA
    return df


def main():
    tables = load_raw_tables()
    report_shapes(tables)
    report_days_employed_anomaly(tables)

    tables["application_train"] = handle_days_employed_anomaly(tables["application_train"])

    flagged = tables["application_train"]["is_pensioner_flag"].sum()
    remaining_nan = tables["application_train"]["DAYS_EMPLOYED"].isna().sum()
    print("\n=== After handling anomaly ===")
    print(f"is_pensioner_flag set on {flagged:,} rows")
    print(f"DAYS_EMPLOYED now NaN on {remaining_nan:,} rows")

    return tables


if __name__ == "__main__":
    main()
