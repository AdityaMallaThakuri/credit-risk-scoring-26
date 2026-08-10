"""Phase 4 static SHAP: global + local explanations for the Phase 3 winning
model (LightGBM), on real data only.

Final production model: refit LightGBM using the exact Phase 3 recipe
(preprocessor fit on all real applicants; SMOTE applied only to the
training data used to fit the model) but now on the FULL
`modeling_feature_set` rather than a single CV fold. Phase 3's k-fold
already validated that this recipe generalizes -- refitting on all
available real data for the deployed artifact is standard practice at this
point, not a new evaluation, so it doesn't need its own held-out split.

SHAP explains the model's raw margin (log-odds) output, NOT the isotonic-
calibrated probability from Phase 3's calibration step. This is a real
subtlety, not an oversight: isotonic calibration is a single monotonic
transform applied to the model's output as a whole -- it has no
feature-by-feature decomposition, so Shapley attribution is undefined
"after" it. `shap.TreeExplainer` guarantees sum(shap_values) + base_value
== the underlying LightGBM model's raw margin output, which is verified
explicitly below (the SHAP additive-consistency hard rule). The Phase 3
calibrated probability is reported alongside the local explanation for
context, but is not itself decomposed by SHAP.

Applicant selection for the local explanation: uses the genuine
out-of-fold calibrated PD from src/models/calibration.py
(data/processed/phase3_calibration_oof.csv, LightGBM rows) to find a real
applicant whose predicted PD sits right at Phase 3's optimal decision
threshold (~0.220) -- an actual borderline approve/reject case -- rather
than picking one from the final model's own in-sample predictions, which
would bias the selection toward whichever applicants this particular
refit is most confident about.

Run: python src/explainability/static_shap.py
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODELS_DIR))

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from cv_split import load_modeling_data  # noqa: E402
from train_baseline_models import build_preprocessor  # noqa: E402

RANDOM_STATE = 42
GLOBAL_SAMPLE_SIZE = 5000
OPTIMAL_THRESHOLD = 0.220  # LightGBM's Phase 3 optimal threshold (src/models/expected_loss.py)
CALIBRATION_OOF_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_calibration_oof.csv"


def fit_final_model():
    X, y, ids, fold = load_modeling_data()
    preprocessor = build_preprocessor(X)
    X_t = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_t, y)
    model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_res, y_res)

    return model, X_t, feature_names, ids


def pick_near_threshold_applicant():
    """Find a real applicant whose out-of-fold calibrated LightGBM PD sits
    right at Phase 3's optimal decision threshold."""
    oof = pd.read_csv(CALIBRATION_OOF_PATH)
    lgbm_oof = oof[oof["model"] == "LightGBM"].copy()
    lgbm_oof["distance"] = (lgbm_oof["proba_calibrated"] - OPTIMAL_THRESHOLD).abs()
    closest = lgbm_oof.sort_values("distance").iloc[0]
    return int(closest["SK_ID_CURR"]), float(closest["proba_calibrated"]), int(closest["y_true"])


def main():
    model, X_t, feature_names, ids = fit_final_model()
    explainer = shap.TreeExplainer(model)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_t), size=min(GLOBAL_SAMPLE_SIZE, len(X_t)), replace=False)
    X_sample = X_t[sample_idx]
    shap_values_sample = explainer(X_sample)
    shap_values_sample.feature_names = list(feature_names)

    applicant_id, applicant_pd, applicant_target = pick_near_threshold_applicant()
    applicant_row_idx = ids[ids == applicant_id].index[0]
    X_applicant = X_t[[applicant_row_idx]]
    shap_values_applicant = explainer(X_applicant)
    shap_values_applicant.feature_names = list(feature_names)

    raw_margin = model.predict(X_applicant, raw_score=True)[0]
    reconstructed = shap_values_applicant.base_values[0] + shap_values_applicant.values[0].sum()
    assert np.isclose(raw_margin, reconstructed, atol=1e-4), (
        f"SHAP additive consistency violated: raw_margin={raw_margin}, "
        f"base_value+sum(shap)={reconstructed}"
    )

    print(f"Global SHAP: {len(sample_idx):,} sampled applicants, {len(feature_names)} features")
    print(f"Local explanation applicant: SK_ID_CURR={applicant_id}, "
          f"OOF calibrated PD={applicant_pd:.4f} (Phase 3 optimal threshold=0.220), actual TARGET={applicant_target}")
    print(f"Additive consistency check passed: base_value + sum(shap) = {reconstructed:.4f} "
          f"== raw model margin = {raw_margin:.4f}")

    return {
        "shap_values_sample": shap_values_sample,
        "shap_values_applicant": shap_values_applicant,
        "applicant_id": applicant_id,
        "applicant_pd": applicant_pd,
        "applicant_target": applicant_target,
    }


if __name__ == "__main__":
    result = main()
