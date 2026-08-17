"""Phase 8.1: confirm the cost-sensitive threshold produces sensible
profit-curve behavior at the extremes (threshold=0 approves ~nobody,
threshold=1 approves everybody) and that the reported "optimal" threshold
genuinely beats both extremes on backtest profit, for every model in the
comparison table -- not just a visual "the curve looked right" check.
"""

import pandas as pd

from conftest import DATA_DIR
from expected_loss import profit_at_threshold

MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]


def _population_for(model_name):
    el = pd.read_csv(DATA_DIR / "phase3_expected_loss.csv")
    return el[el["model"] == model_name][["SK_ID_CURR", "PD", "TARGET", "EAD"]]


def test_threshold_zero_rejects_almost_everyone():
    df = _population_for("LightGBM")
    profit, approval_rate, _ = profit_at_threshold(df, 0.0)
    assert approval_rate < 0.01
    assert abs(profit) < 1e8  # near-zero profit -- almost nothing was approved to earn or lose on


def test_threshold_one_approves_everyone():
    df = _population_for("LightGBM")
    profit, approval_rate, approved_default_rate = profit_at_threshold(df, 1.0)
    assert approval_rate == 1.0
    assert approved_default_rate == df["TARGET"].mean()
    assert profit > 0  # per CLAUDE.md: positive but suboptimal


def test_optimal_threshold_beats_both_extremes_for_every_model():
    optima = pd.read_csv(DATA_DIR / "phase3_optimal_thresholds.csv").set_index("model")
    for model_name in MODELS:
        df = _population_for(model_name)
        optimal_profit = optima.loc[model_name, "profit"]

        profit_at_zero, _, _ = profit_at_threshold(df, 0.0)
        profit_at_one, _, _ = profit_at_threshold(df, 1.0)

        assert optimal_profit > profit_at_zero, f"{model_name}: optimal threshold should beat threshold=0"
        assert optimal_profit > profit_at_one, f"{model_name}: optimal threshold should beat threshold=1"


def test_optimal_threshold_matches_the_precomputed_profit_curve_argmax():
    """Cross-check phase3_optimal_thresholds.csv against phase3_profit_curves.csv
    directly -- the "optimal" row should actually be at (or extremely near) the
    curve's own maximum, not just plausible-looking."""
    curves = pd.read_csv(DATA_DIR / "phase3_profit_curves.csv")
    optima = pd.read_csv(DATA_DIR / "phase3_optimal_thresholds.csv").set_index("model")

    for model_name in MODELS:
        curve = curves[curves["model"] == model_name]
        curve_max_profit = curve["profit"].max()
        reported_optimal_profit = optima.loc[model_name, "profit"]
        assert reported_optimal_profit == curve_max_profit
