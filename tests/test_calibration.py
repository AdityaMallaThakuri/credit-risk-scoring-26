"""Phase 8.1: confirm calibration holds on held-out data, not just on the
training population it was fit on.

Two independent held-out sources are checked:
  - phase3_calibration_oof.csv: genuine out-of-fold predictions from the
    nested inner_train/calib CV split in src/models/calibration.py --
    every row here is a prediction on data the calibrator never touched.
  - final_model_pipeline.joblib's own stored calib_brier_raw/calibrated:
    the deployed pipeline's 80/20 held-out calibration split
    (src/models/train_final_model.py).
"""

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from conftest import DATA_DIR

DEPLOYED_MODEL_NAME = "LightGBM"


def _lightgbm_oof():
    df = pd.read_csv(DATA_DIR / "phase3_calibration_oof.csv")
    return df[df["model"] == DEPLOYED_MODEL_NAME]


def test_oof_calibration_improves_or_matches_brier():
    df = _lightgbm_oof()
    brier_raw = brier_score_loss(df["y_true"], df["proba_raw"])
    brier_calibrated = brier_score_loss(df["y_true"], df["proba_calibrated"])
    assert brier_calibrated < brier_raw


def test_oof_calibration_curve_is_reliable():
    """Mean calibration error across 10 quantile bins of *held-out*
    predictions should be small -- i.e. "predicted PD of X%" really does
    mean "actual default rate of about X%" on data the calibrator never
    saw during fitting."""
    df = _lightgbm_oof()
    observed_rate, mean_predicted = calibration_curve(
        df["y_true"], df["proba_calibrated"], n_bins=10, strategy="quantile"
    )
    mean_abs_calibration_error = abs(observed_rate - mean_predicted).mean()
    assert mean_abs_calibration_error < 0.01


def test_oof_calibration_curve_is_monotonic():
    """Higher predicted-PD bins should correspond to higher actual default
    rates -- a basic sanity check that calibration didn't scramble rank
    ordering while fixing scale."""
    df = _lightgbm_oof()
    observed_rate, _ = calibration_curve(df["y_true"], df["proba_calibrated"], n_bins=10, strategy="quantile")
    assert list(observed_rate) == sorted(observed_rate)


def test_deployed_artifact_calibration_improves_on_held_out_split():
    """The persisted pipeline's OWN 80/20 held-out calibration split
    (distinct from the OOF CV split above) should show the same effect."""
    artifact = joblib.load(DATA_DIR / "final_model_pipeline.joblib")
    assert artifact["calib_brier_calibrated"] < artifact["calib_brier_raw"]
    assert artifact["calib_n"] > 0
    assert artifact["calib_brier_calibrated"] < 0.15  # sane upper bound, not near-random (0.25 for base-rate-only)
