"""Phase 7 prerequisite: fit and persist ONE deployable scoring pipeline
(preprocessor + LightGBM + isotonic calibrator), so the FastAPI app
(src/app/) can load a single artifact once at startup instead of
refitting per request or per process restart.

This didn't exist before Phase 7: every prior phase either refits
LightGBM in-memory per script run (static_shap.py's fit_final_model(),
reused by adaptive_shap.py/weighted_temporal_shap.py) or produces
calibration purely for offline evaluation, per-CV-fold, never as a single
deployable artifact (src/models/calibration.py's nested outer-fold
inner_train/calib split, one calibrator per fold, never persisted).

Recipe (same split-before-resample and calibration discipline as
calibration.py, applied once instead of per outer fold, since this is
the final deployed artifact rather than a new evaluation -- Phase 3's
k-fold already validated the recipe generalizes, same reasoning
static_shap.py's fit_final_model() docstring gives for refitting the
model on all data):
  1. Split all of `modeling_feature_set` into train (80%) / calib (20%),
     stratified on TARGET.
  2. Fit the preprocessor (median-impute numeric, "missing"-category +
     one-hot categorical -- same as train_baseline_models.build_preprocessor)
     on train only.
  3. SMOTE the transformed train split only. Calib is NEVER resampled --
     calibration must be learned against the real class balance.
  4. Fit LightGBM (Phase 3's winning model, same hyperparameters) on the
     resampled train split.
  5. Fit an IsotonicRegression calibrator on the model's raw (uncalibrated)
     probability for the untouched calib split vs. its real labels.
Split-before-resample and fit-on-train-only are preserved exactly as
elsewhere in this project: calib never sees SMOTE, and the calibrator
never sees anything the model was trained on.

Persists (joblib): preprocessor, model, calibrator, feature_names,
LightGBM's Phase 3 profit-optimal threshold/LGD/margin-rate assumptions
(src/models/expected_loss.py), and calib-set Brier scores (raw vs.
calibrated) as a sanity check that calibration actually helped, printed
on every run so a regression is visible immediately.

Run: python src/models/train_final_model.py
"""

from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from cv_split import load_modeling_data
from train_baseline_models import build_preprocessor

RANDOM_STATE = 42
CALIB_FRACTION = 0.2
LIGHTGBM_OPTIMAL_THRESHOLD = 0.220  # Phase 3's profit-optimal threshold (src/models/expected_loss.py)
LGD_ASSUMPTION = 0.55  # matches src/models/expected_loss.py
MARGIN_RATE = 0.15  # matches src/models/expected_loss.py

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "final_model_pipeline.joblib"


def train_final_pipeline():
    X, y, ids, _ = load_modeling_data()
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=CALIB_FRACTION, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_calib_t = preprocessor.transform(X_calib)
    feature_names = list(preprocessor.get_feature_names_out())

    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)
    model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_train_res, y_train_res)

    calib_raw_proba = model.predict_proba(X_calib_t)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calib_raw_proba, y_calib)
    calib_calibrated_proba = calibrator.predict(calib_raw_proba)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,
        "feature_names": feature_names,
        "threshold": LIGHTGBM_OPTIMAL_THRESHOLD,
        "lgd": LGD_ASSUMPTION,
        "margin_rate": MARGIN_RATE,
        "trained_on_n": int(len(X_train)),
        "calib_n": int(len(X_calib)),
        "calib_brier_raw": float(brier_score_loss(y_calib, calib_raw_proba)),
        "calib_brier_calibrated": float(brier_score_loss(y_calib, calib_calibrated_proba)),
    }


def main():
    artifact = train_final_pipeline()
    joblib.dump(artifact, ARTIFACT_PATH)

    print(f"Trained on {artifact['trained_on_n']:,} rows, calibrated on {artifact['calib_n']:,} held-out rows")
    print(f"Calib Brier: raw={artifact['calib_brier_raw']:.4f}  calibrated={artifact['calib_brier_calibrated']:.4f}")
    print(f"Saved: {ARTIFACT_PATH}")
    return artifact


if __name__ == "__main__":
    artifact = main()
