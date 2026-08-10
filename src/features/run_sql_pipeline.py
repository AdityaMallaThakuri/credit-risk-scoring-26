"""Load raw Home Credit CSVs into a local SQLite db and run the sql/
feature-extraction pipeline (sql/01_*.sql .. sql/07_*.sql, in order).

Never writes to data/raw/. The SQLite db is a derived artifact under
data/processed/.

Run: python src/features/run_sql_pipeline.py [--sample N]
    --sample N   load only the first N SK_ID_CURR (by row order in
                 application_train) into every table, for a fast dev-scale
                 run. Omit for a full-scale run.
"""

import argparse
import math
import sqlite3
from pathlib import Path

import pandas as pd

from load_data import RAW_DIR

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"

RAW_TABLES = {
    "application_train": "HC_application_train.csv",
    "bureau": "HC_bureau.csv",
    "bureau_balance": "HC_bureau_balance.csv",
    "credit_card_balance": "HC_credit_card_balance.csv",
    "installments_payments": "HC_installments_payments.csv",
    "pos_cash_balance": "HC_POS_CASH_balance.csv",
    "previous_application": "HC_previous_application.csv",
}

# Composite indexes matching each table's actual join/partition keys in the
# sql/ scripts -- a plain SK_ID_CURR-only index isn't enough to keep the
# window-function sorts and the bureau_balance aggregation off full table
# scans at full scale (27.3M rows for bureau_balance, 10M for POS_CASH).
TABLE_INDEXES = {
    "application_train": ["CREATE UNIQUE INDEX IF NOT EXISTS idx_app_sk_id_curr ON application_train (SK_ID_CURR)"],
    "bureau": [
        "CREATE INDEX IF NOT EXISTS idx_bureau_sk_id_curr ON bureau (SK_ID_CURR)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_bureau_sk_id_bureau ON bureau (SK_ID_BUREAU)",
    ],
    "bureau_balance": ["CREATE INDEX IF NOT EXISTS idx_bb_sk_id_bureau_months ON bureau_balance (SK_ID_BUREAU, MONTHS_BALANCE)"],
    "credit_card_balance": [
        "CREATE INDEX IF NOT EXISTS idx_cc_curr_prev_months ON credit_card_balance (SK_ID_CURR, SK_ID_PREV, MONTHS_BALANCE)"
    ],
    "pos_cash_balance": [
        "CREATE INDEX IF NOT EXISTS idx_pos_curr_prev_months ON pos_cash_balance (SK_ID_CURR, SK_ID_PREV, MONTHS_BALANCE)"
    ],
    "installments_payments": ["CREATE INDEX IF NOT EXISTS idx_inst_sk_id_curr ON installments_payments (SK_ID_CURR)"],
    "previous_application": ["CREATE INDEX IF NOT EXISTS idx_prev_sk_id_curr ON previous_application (SK_ID_CURR)"],
}

PIPELINE_SCRIPTS = [
    "01_bureau_features.sql",
    "02_previous_application_features.sql",
    "03_installments_features.sql",
    "04_credit_card_utilization_trend.sql",
    "05_pos_cash_trend.sql",
    "06_application_time_signals.sql",
    "07_final_feature_table.sql",
]


def load_raw_tables_into_sqlite(conn, sample_n=None):
    sk_id_filter = None
    sk_bureau_filter = None
    if sample_n is not None:
        app_ids = pd.read_csv(RAW_DIR / RAW_TABLES["application_train"], usecols=["SK_ID_CURR"])
        sk_id_filter = set(app_ids["SK_ID_CURR"].head(sample_n))
        bureau_ids = pd.read_csv(RAW_DIR / RAW_TABLES["bureau"], usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
        sk_bureau_filter = set(bureau_ids.loc[bureau_ids["SK_ID_CURR"].isin(sk_id_filter), "SK_ID_BUREAU"])

    for table_name, filename in RAW_TABLES.items():
        path = RAW_DIR / filename
        chunks = pd.read_csv(path, chunksize=200_000)
        first = True
        for chunk in chunks:
            if table_name == "bureau_balance" and sk_bureau_filter is not None:
                chunk = chunk[chunk["SK_ID_BUREAU"].isin(sk_bureau_filter)]
            elif sk_id_filter is not None and "SK_ID_CURR" in chunk.columns:
                chunk = chunk[chunk["SK_ID_CURR"].isin(sk_id_filter)]
            chunk.to_sql(table_name, conn, if_exists="replace" if first else "append", index=False)
            first = False
        for idx_sql in TABLE_INDEXES.get(table_name, []):
            conn.execute(idx_sql)
        print(f"Loaded {table_name} ({filename})")


def run_pipeline_scripts(conn):
    for script_name in PIPELINE_SCRIPTS:
        sql_text = (SQL_DIR / script_name).read_text()
        conn.executescript(sql_text)
        print(f"Ran {script_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Load only the first N SK_ID_CURR (dev-scale run)")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.create_function("SQRT", 1, math.sqrt)

    load_raw_tables_into_sqlite(conn, sample_n=args.sample)
    run_pipeline_scripts(conn)
    conn.commit()

    n_rows, n_cols = conn.execute("SELECT COUNT(*) FROM final_feature_table").fetchone()[0], len(
        [c[1] for c in conn.execute("PRAGMA table_info(final_feature_table)").fetchall()]
    )
    print(f"\nfinal_feature_table: {n_rows:,} rows x {n_cols} columns -> {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
