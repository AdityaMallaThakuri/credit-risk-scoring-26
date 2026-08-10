"""Phase 3 baseline model comparison: Logistic Regression, Random Forest,
XGBoost, LightGBM on `modeling_feature_set` (real data only), using the
stratified 5-fold harness from src/models/cv_split.py.

Split-before-resample (hard rule): for every fold, preprocessing
(imputation/encoding) is fit on that fold's training rows only, then SMOTE
is applied to the transformed training rows only. The test fold is never
touched by SMOTE and only ever sees `.transform()`, never `.fit()`.

Preprocessing choice for this baseline pass (documented here since Phase 3's
imputation strategy was left open in CLAUDE.md): median imputation for
numeric columns, a "missing" category for the 2 text columns
(WEEKDAY_APPR_PROCESS_START has no nulls; prev_app_last_status's nulls mean
"no previous application" and are kept as their own category rather than
imputed away), then one-hot encoding. This is applied uniformly across all
4 models so the comparison isn't confounded by different models seeing
different data -- even though XGBoost/LightGBM could natively handle NaNs,
and this may be revisited once the has_*_history-flag / imputation decision
is finalized. Logistic Regression additionally gets a StandardScaler (fit
on the post-SMOTE training fold) since it's scale-sensitive; the tree
models do not need it.

Metrics reported per fold and averaged: ROC-AUC, F1 (at the default 0.5
probability threshold -- cost-sensitive thresholding is a later Phase 3
task), and the KS statistic (max(TPR - FPR) over the ROC curve), the
standard credit-scoring separation measure.

Run: python src/models/train_baseline_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from cv_split import iterate_folds, load_modeling_data

RANDOM_STATE = 42
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_baseline_model_comparison.csv"

MODEL_FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"),
    "LightGBM": lambda: LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
}


def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = SimpleImputer(strategy="median")
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def ks_statistic(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def main():
    X, y, ids, fold = load_modeling_data()

    rows = []
    for fold_number, X_train, X_test, y_train, y_test in iterate_folds(X, y, fold):
        preprocessor = build_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)

        scaler = StandardScaler().fit(X_train_res)
        X_train_res_scaled = scaler.transform(X_train_res)
        X_test_scaled = scaler.transform(X_test_t)

        for model_name, factory in MODEL_FACTORIES.items():
            model = factory()
            if model_name == "LogisticRegression":
                model.fit(X_train_res_scaled, y_train_res)
                proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train_res, y_train_res)
                proba = model.predict_proba(X_test_t)[:, 1]

            preds = (proba >= 0.5).astype(int)
            rows.append({
                "model": model_name,
                "fold": fold_number,
                "auc": roc_auc_score(y_test, proba),
                "f1": f1_score(y_test, preds),
                "ks": ks_statistic(y_test, proba),
            })
            print(f"fold {fold_number} | {model_name:<18} AUC={rows[-1]['auc']:.4f}  F1={rows[-1]['f1']:.4f}  KS={rows[-1]['ks']:.4f}")

    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("model")[["auc", "f1", "ks"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reindex(MODEL_FACTORIES.keys())

    print("\n=== Phase 3 baseline comparison (mean +/- std across 5 folds) ===")
    print(summary.round(4).to_string())

    summary.round(4).to_csv(RESULTS_PATH)
    print(f"\nSaved: {RESULTS_PATH}")
    return summary, per_fold


if __name__ == "__main__":
    summary, per_fold = main()
