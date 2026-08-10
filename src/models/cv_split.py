"""Stratified 5-fold split for the real-data modeling baseline (Phase 3).

Per the data-strategy rule, headline model-comparison numbers come from
`modeling_feature_set` (real Home Credit data) only -- no synthetic overlay
involved here.

This module owns the fold assignment and nothing else: it does not scale,
impute, encode, or resample. Each model script (LogReg/RF/XGBoost/LightGBM)
reads the same `cv_folds` table so all 4 models are compared on identical
splits. Fold assignment is by SK_ID_CURR, computed once and persisted --
re-running this script reproduces the same folds (fixed random_state) but a
persisted table guarantees it, since sklearn/pandas versions can drift.

Split-before-resample: `get_fold` returns the raw, unresampled train/test
frames for a given fold. Any resampling (SMOTE etc., Phase 3 task 3) must be
applied by the caller to the returned X_train/y_train only, after this split
-- never before it, and never to X_test/y_test. This module contains no
resampling code, so there is nothing here that could leak test-fold rows
into training.

Run: python src/models/cv_split.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
N_SPLITS = 5
RANDOM_STATE = 42


def build_cv_folds(db_path=DB_PATH, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Assign every row in modeling_feature_set to one of n_splits stratified
    test folds and persist the assignment as a `cv_folds` table
    (SK_ID_CURR, TARGET, cv_fold)."""
    conn = sqlite3.connect(db_path)
    ids_targets = pd.read_sql("SELECT SK_ID_CURR, TARGET FROM modeling_feature_set", conn)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_assignment = pd.Series(index=ids_targets.index, dtype="int64")
    for fold_number, (_, test_idx) in enumerate(skf.split(ids_targets, ids_targets["TARGET"])):
        fold_assignment.iloc[test_idx] = fold_number
    ids_targets["cv_fold"] = fold_assignment

    ids_targets.to_sql("cv_folds", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cv_folds_sk_id_curr ON cv_folds (SK_ID_CURR)")
    conn.commit()
    conn.close()
    return ids_targets


def load_modeling_data(db_path=DB_PATH):
    """Load features (X), target (y), and SK_ID_CURR for every row of
    modeling_feature_set, joined to its cv_fold assignment."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT m.*, f.cv_fold FROM modeling_feature_set m "
        "JOIN cv_folds f ON f.SK_ID_CURR = m.SK_ID_CURR",
        conn,
    )
    conn.close()

    ids = df["SK_ID_CURR"]
    y = df["TARGET"]
    fold = df["cv_fold"]
    X = df.drop(columns=["SK_ID_CURR", "TARGET", "cv_fold"])
    return X, y, ids, fold


def get_fold(fold_number, X, y, fold):
    """Return the raw, unresampled (X_train, X_test, y_train, y_test) for one
    fold. Apply resampling (e.g. SMOTE) to X_train/y_train only, after this
    call -- never to X_test/y_test, never before this split."""
    test_mask = fold == fold_number
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


def iterate_folds(X, y, fold, n_splits=N_SPLITS):
    """Yield (fold_number, X_train, X_test, y_train, y_test) for every fold."""
    for fold_number in range(n_splits):
        X_train, X_test, y_train, y_test = get_fold(fold_number, X, y, fold)
        yield fold_number, X_train, X_test, y_train, y_test


def main():
    fold_table = build_cv_folds()
    print(f"cv_folds: {len(fold_table):,} rows assigned across {N_SPLITS} folds (random_state={RANDOM_STATE})")

    X, y, ids, fold = load_modeling_data()
    for fold_number, X_train, X_test, y_train, y_test in iterate_folds(X, y, fold):
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        print(
            f"fold {fold_number}: train={len(X_train):,} (default rate={train_rate:.4f})  "
            f"test={len(X_test):,} (default rate={test_rate:.4f})"
        )


if __name__ == "__main__":
    main()
