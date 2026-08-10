"""Phase 3 calibration: out-of-fold reliability data for all 4 baseline
models (LogReg, RandomForest, XGBoost, LightGBM), per the Phase 3 exit
criteria requiring calibration quality for the full model comparison table,
not just the AUC/KS winner.

Produces raw and isotonic-calibrated probabilities for every row of
`modeling_feature_set`, for each model, each one predicted by a model that
never saw that row during training or calibration fitting -- so the
reliability diagram/Brier score built from this output reflects genuine
held-out behavior, not train-set optimism.

Nested split per outer CV fold (reuses src/models/cv_split.py's 5 folds):
  outer train  ->  inner_train (fit the model, after SMOTE) / calib (fit the
  isotonic calibrator, NOT resampled -- calibration must be learned against
  the real class balance, not SMOTE's synthetic one)
  outer test   ->  scored by both the raw model and the calibrator; never
  used to fit anything.

This keeps every hard rule intact: split before resample (inner_train/calib
split happens before SMOTE, and SMOTE only ever touches inner_train), and
the calibrator only ever sees `.fit()` on calib and `.predict()` on test.
Logistic Regression additionally gets a StandardScaler fit on the
post-SMOTE inner_train (same convention as train_baseline_models.py).

Run: python src/models/calibration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cv_split import iterate_folds, load_modeling_data
from train_baseline_models import MODEL_FACTORIES, build_preprocessor

RANDOM_STATE = 42
CALIB_FRACTION = 0.2
OOF_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_calibration_oof.csv"
QUALITY_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_calibration_quality.csv"


def get_oof_calibration_data(model_name, model_factory, calib_fraction=CALIB_FRACTION):
    """Return a DataFrame with columns [SK_ID_CURR, fold, y_true, proba_raw,
    proba_calibrated] for one model, one row per SK_ID_CURR in
    modeling_feature_set, all out-of-fold."""
    X, y, ids, fold = load_modeling_data()

    records = []
    for fold_number, X_train, X_test, y_train, y_test in iterate_folds(X, y, fold):
        X_inner_train, X_calib, y_inner_train, y_calib = train_test_split(
            X_train, y_train, test_size=calib_fraction, stratify=y_train, random_state=RANDOM_STATE
        )

        preprocessor = build_preprocessor(X_inner_train)
        X_inner_train_t = preprocessor.fit_transform(X_inner_train)
        X_calib_t = preprocessor.transform(X_calib)
        X_test_t = preprocessor.transform(X_test)

        X_inner_train_res, y_inner_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
            X_inner_train_t, y_inner_train
        )

        if model_name == "LogisticRegression":
            scaler = StandardScaler().fit(X_inner_train_res)
            X_inner_train_res = scaler.transform(X_inner_train_res)
            X_calib_t = scaler.transform(X_calib_t)
            X_test_t = scaler.transform(X_test_t)

        model = model_factory()
        model.fit(X_inner_train_res, y_inner_train_res)

        proba_calib_raw = model.predict_proba(X_calib_t)[:, 1]
        proba_test_raw = model.predict_proba(X_test_t)[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(proba_calib_raw, y_calib)
        proba_test_calibrated = calibrator.predict(proba_test_raw)

        records.append(pd.DataFrame({
            "model": model_name,
            "SK_ID_CURR": ids.loc[y_test.index].values,
            "fold": fold_number,
            "y_true": y_test.values,
            "proba_raw": proba_test_raw,
            "proba_calibrated": proba_test_calibrated,
        }))
        print(f"{model_name} | fold {fold_number}: calibrated {len(y_test):,} held-out rows")

    return pd.concat(records, ignore_index=True)


def main():
    all_results = []
    quality_rows = []
    for model_name, model_factory in MODEL_FACTORIES.items():
        result = get_oof_calibration_data(model_name, model_factory)
        all_results.append(result)

        brier_raw = brier_score_loss(result["y_true"], result["proba_raw"])
        brier_calibrated = brier_score_loss(result["y_true"], result["proba_calibrated"])
        quality_rows.append({"model": model_name, "brier_raw": brier_raw, "brier_calibrated": brier_calibrated})
        print(f"{model_name}: Brier raw={brier_raw:.4f}  calibrated={brier_calibrated:.4f}")

    oof = pd.concat(all_results, ignore_index=True)
    oof.to_csv(OOF_PATH, index=False)
    print(f"\nSaved: {OOF_PATH} ({len(oof):,} rows)")

    quality = pd.DataFrame(quality_rows).set_index("model").reindex(MODEL_FACTORIES.keys())
    quality.to_csv(QUALITY_PATH)
    print(f"Saved: {QUALITY_PATH}")
    print(quality.round(4).to_string())

    return oof, quality


if __name__ == "__main__":
    oof, quality = main()
