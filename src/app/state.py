"""Loads every artifact the API needs exactly once, at process startup --
never per-request. Held on `app.state` (see main.py's lifespan) rather
than as module globals so tests/multiple app instances don't share state
by accident.

Two different kinds of "load" happen here, deliberately kept separate:
  - Live-scoring resources (the persisted model pipeline, the applicant
    feature table, the SHAP explainer) -- used to answer a request about
    an arbitrary SK_ID_CURR on demand.
  - Precomputed analysis artifacts (fairness/drift/profit-curve CSVs from
    Phases 3-6) -- these are served as-is, not recomputed per request;
    recomputing e.g. a fairness metric over the full synthetic overlay on
    every API call would be wasteful and these numbers are already the
    project's audited, reported results.
"""

import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import Request

import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parents[0] / "models"
FAIRNESS_DIR = APP_DIR.parents[0] / "fairness"
EXPLAINABILITY_DIR = APP_DIR.parents[0] / "explainability"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(FAIRNESS_DIR))
sys.path.insert(0, str(EXPLAINABILITY_DIR))

from expected_loss import LGD_ASSUMPTION, MARGIN_RATE, profit_at_threshold  # noqa: E402
from fairness_metrics import AGE_BAND_EDGES, AGE_BAND_LABELS, SMALL_GROUP_WARNING_N, summarize_attribute  # noqa: E402
# Reuses Phase 6's own weighting/rescale functions rather than
# reimplementing the formula a second time here.
from weighted_temporal_shap import blend_weights  # noqa: E402
from weighted_temporal_shap import weighted_temporal_shap as apply_weighted_shap  # noqa: E402

DATA_DIR = APP_DIR.parents[1] / "data" / "processed"
DB_PATH = DATA_DIR / "credit_risk.db"
ARTIFACT_PATH = DATA_DIR / "final_model_pipeline.joblib"

FAIRNESS_SYNTHETIC_PATH = DATA_DIR / "phase4_synthetic_bias_detection.csv"
FAIRNESS_ISOLATED_EFFECT_PATH = DATA_DIR / "phase4_synthetic_bias_isolated_effect.csv"
FAIRNESS_REAL_PATH = DATA_DIR / "phase4_real_fairness_metrics.csv"
TRADEOFF_PATHS = {
    "age_band": DATA_DIR / "phase4_fairness_accuracy_tradeoff_age_band.csv",
    "CODE_GENDER": DATA_DIR / "phase4_fairness_accuracy_tradeoff_CODE_GENDER.csv",
}
DRIFT_PATH = DATA_DIR / "phase5_drift_metrics.csv"
EXPECTED_LOSS_PATH = DATA_DIR / "phase3_expected_loss.csv"
WEIGHT_COMPONENTS_PATH = DATA_DIR / "phase6_weight_components.csv"

SIMULATE_MODEL_NAME = "LightGBM"  # Phase 3's winning model -- the one this API actually deploys

# Risk-tier cutoffs for the Segments screen (roadmap 7.2 #2). Not arbitrary
# terciles: the upper cutoff is the deployed decision threshold (0.220)
# itself, so "High" == "would be declined today" -- a business-meaningful
# cut, not a cosmetic one. "Medium" (0.10-0.22) covers ~24% of applicants,
# "Low" (<0.10) the remaining ~73% -- checked against the real OOF
# calibrated-PD distribution before picking these edges.
RISK_TIER_EDGES = [0.0, 0.10, 0.220, 1.0]
RISK_TIER_LABELS = ["Low", "Medium", "High"]


class AppState:
    """Everything the routers need, attached to app.state.app_state at startup."""

    def __init__(self):
        artifact = joblib.load(ARTIFACT_PATH)
        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.calibrator = artifact["calibrator"]
        self.feature_names = artifact["feature_names"]
        self.threshold = artifact["threshold"]
        self.lgd = artifact["lgd"]
        self.margin_rate = artifact["margin_rate"]

        # tree_path_dependent (no explicit background) -- same convention as
        # static_shap.py, since this endpoint explains one applicant at a
        # time against the model's own training distribution, not a
        # period-specific drift background (that's Methods A/B/C's job).
        self.explainer = shap.TreeExplainer(self.model)

        conn = sqlite3.connect(DB_PATH)
        self.applicant_features = pd.read_sql("SELECT * FROM modeling_feature_set", conn).set_index("SK_ID_CURR")
        self.applicant_demo = pd.read_sql(
            "SELECT SK_ID_CURR, AMT_CREDIT AS EAD, CODE_GENDER, NAME_FAMILY_STATUS, DAYS_BIRTH FROM application_train",
            conn,
        ).set_index("SK_ID_CURR")
        conn.close()

        self.fairness_synthetic = pd.read_csv(FAIRNESS_SYNTHETIC_PATH)
        self.fairness_isolated_effect = pd.read_csv(FAIRNESS_ISOLATED_EFFECT_PATH)
        self.fairness_real = pd.read_csv(FAIRNESS_REAL_PATH)
        self.drift = pd.read_csv(DRIFT_PATH)
        self.fairness_tradeoff = {attr: pd.read_csv(path) for attr, path in TRADEOFF_PATHS.items()}

        el = pd.read_csv(EXPECTED_LOSS_PATH)
        self.simulate_population = el[el["model"] == SIMULATE_MODEL_NAME][["SK_ID_CURR", "PD", "TARGET", "EAD"]].copy()

        # Phase 6's precomputed per-period, per-feature w_drift/w_cost --
        # these are period-level (not applicant-specific), so they can be
        # applied live to ANY applicant's own static SHAP values, not just
        # the 500-applicant eval sample Phase 6's own script evaluated.
        self.weight_components = pd.read_csv(WEIGHT_COMPONENTS_PATH).set_index(["period", "feature"])

        # Segments screen (roadmap 7.2 #2): same PD/TARGET/EAD population as
        # simulate_population (Phase 3's OOF calibrated PD for LightGBM,
        # the model this API deploys) -- reuses it rather than recomputing,
        # joined to the same demographic attributes/age-band binning as
        # fairness_metrics.py (imported, not duplicated), plus a risk-tier
        # bucketing of PD. Presentation on top of existing outputs only.
        demo = self.applicant_demo[["CODE_GENDER", "NAME_FAMILY_STATUS", "DAYS_BIRTH"]].copy()
        demo["age_years"] = -demo["DAYS_BIRTH"] / 365.25
        demo["age_band"] = pd.cut(demo["age_years"], bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=False)

        self.segment_population = self.simulate_population.merge(
            demo[["CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"]], on="SK_ID_CURR", how="left"
        )
        self.segment_population["risk_tier"] = pd.cut(
            self.segment_population["PD"], bins=RISK_TIER_EDGES, labels=RISK_TIER_LABELS, right=False, include_lowest=True
        )
        self.segment_population["approved"] = (self.segment_population["PD"] <= self.threshold).astype(int)

    def simulate_profit(self, threshold: float):
        """profit/approval_rate/approved_default_rate are the realized
        backtest numbers (using actual TARGET, per expected_loss.py's own
        convention). total_expected_loss is the separate, ex-ante PD*LGD*EAD
        sum over the approved population -- the two are complementary, not
        duplicates: one says what actually happened in backtest, the other
        says what the model expects going forward."""
        profit, approval_rate, approved_default_rate = profit_at_threshold(self.simulate_population, threshold)
        approved = self.simulate_population[self.simulate_population["PD"] <= threshold]
        total_expected_loss = float((approved["PD"] * self.lgd * approved["EAD"]).sum())
        return {
            "threshold": threshold,
            "profit": profit,
            "approval_rate": approval_rate,
            "approved_default_rate": approved_default_rate,
            "total_expected_loss": total_expected_loss,
        }

    def compute_live_fairness(self, threshold: float, attribute: str):
        """Recomputes DPD/EOD/EqualizedOddsDiff at an arbitrary threshold --
        unlike fairness_real/fairness_synthetic (fixed at Phase 3's optimal
        threshold=0.220), this is what the policy simulator's slider needs.
        DPD/EOD/EqualizedOddsDiff already exclude any group with n <
        SMALL_GROUP_WARNING_N from the max-min spread (fairness_metrics.py's
        own _max_min_diff) -- excluded_groups surfaces WHICH groups were
        dropped and why, so a caller can't mistake "silently excluded" for
        "no small groups exist"."""
        df = self.segment_population.copy()
        df["approved"] = (df["PD"] <= threshold).astype(int)
        summary, selection_rates, _, _ = summarize_attribute(df, attribute)
        small = selection_rates[selection_rates["n"] < SMALL_GROUP_WARNING_N]
        excluded_groups = [f"{group} (n={int(n)})" for group, n in small["n"].items()]
        return summary, excluded_groups

    def weight_vector(self, period: str):
        """w_drift/w_cost for `period`, aligned to self.feature_names order
        (looked up by name, not assumed row order, since the CSV's own row
        order isn't guaranteed to match the joblib artifact's)."""
        sub = self.weight_components.loc[period].reindex(self.feature_names)
        return sub["w_drift"].to_numpy(), sub["w_cost"].to_numpy()

    def blend_weights_for(self, period: str, alpha: float):
        w_drift, w_cost = self.weight_vector(period)
        return blend_weights(w_drift, w_cost, alpha)

    def compute_weighted_shap(self, static_values, base_value, raw_margin, period, alpha):
        """static_values: (54,) raw per-feature SHAP for one applicant (from
        self.explainer, tree_path_dependent -- already exactly additive-
        consistent on its own). Applies Phase 6's actual formula
        (Weighted_SHAP = SHAP x blended weight, then rescaled to exact
        additive consistency) live, reusing weighted_temporal_shap.py's own
        functions rather than reimplementing them."""
        weights = self.blend_weights_for(period, alpha)
        weighted = apply_weighted_shap(
            static_values.reshape(1, -1), np.array([base_value]), np.array([raw_margin]), weights
        )
        return weighted[0]


def get_state(request: Request) -> AppState:
    """FastAPI dependency: `state: AppState = Depends(get_state)` in a router."""
    return request.app.state.app_state
