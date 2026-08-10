"""/shap-stability-eval for Phase 5: compares static SHAP vs Methods A/B/C
(src/explainability/adaptive_shap.py) across the synthetic periods, per the
shap-stability-eval skill's spec.

For each method, and for every pair of *consecutive* periods (P0-P1,
P1-P2, P2-P3, P3-P4), computes on the mean-|SHAP| vector across that
period's evaluated applicants:
- cosine similarity (direction/magnitude of feature importance)
- Kendall tau rank correlation (does the *order* of important features hold)
- Jaccard@10 (overlap of the top-10 most important features)

Each metric is then averaged across the 4 consecutive pairs into one
stability score per method -- higher is more stable across drift.

Also reports the fairness-reduction columns (DeltaDPD, DeltaEOD) the skill
requires alongside stability: these are 0 for every method here by
construction (see adaptive_shap.py's docstring) -- none of Methods A/B/C
change the model's decisions, only how those decisions are explained, so
there is no mechanism by which they could move DPD/EOD relative to static
SHAP's baseline. Reported explicitly rather than omitted.

Run: python src/explainability/shap_stability_eval.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase5_adaptive_shap_values.npz"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase5_shap_stability_eval.csv"

PERIODS = ["P0", "P1", "P2", "P3", "P4"]
METHODS = ["static", "method_a", "method_b", "method_c"]
TOP_K = 10


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def jaccard_at_k(a, b, feature_names, k=TOP_K):
    top_a = set(feature_names[np.argsort(-a)[:k]])
    top_b = set(feature_names[np.argsort(-b)[:k]])
    return len(top_a & top_b) / len(top_a | top_b)


def mean_abs_shap_by_period(data, feature_names, method):
    return {period: np.abs(data[f"{period}_{method}"]).mean(axis=0) for period in PERIODS}


def stability_for_method(data, feature_names, method):
    mean_abs = mean_abs_shap_by_period(data, feature_names, method)

    cosine_scores, tau_scores, jaccard_scores = [], [], []
    for p1, p2 in zip(PERIODS[:-1], PERIODS[1:]):
        v1, v2 = mean_abs[p1], mean_abs[p2]
        cosine_scores.append(cosine_similarity(v1, v2))
        tau, _ = kendalltau(v1, v2)
        tau_scores.append(tau)
        jaccard_scores.append(jaccard_at_k(v1, v2, feature_names))

    return {
        "method": method,
        "cosine_mean": np.mean(cosine_scores),
        "kendall_tau_mean": np.mean(tau_scores),
        f"jaccard_at_{TOP_K}_mean": np.mean(jaccard_scores),
        "DeltaDPD": 0.0,
        "DeltaEOD": 0.0,
    }


def main():
    data = np.load(RESULTS_PATH, allow_pickle=True)
    feature_names = data["feature_names"]

    rows = [stability_for_method(data, feature_names, method) for method in METHODS]
    result = pd.DataFrame(rows).set_index("method")
    result.to_csv(OUTPUT_PATH)

    print(result.round(4).to_string())
    print(f"\nSaved: {OUTPUT_PATH}")

    print("\nDeltaDPD / DeltaEOD are 0 for every method by construction: Methods A/B/C only change how "
          "the same fixed model's decisions are *explained*, never the decisions themselves, so there is "
          "no mechanism by which they could move a fairness metric relative to static SHAP's baseline.")

    ranked_by_cosine = result["cosine_mean"].sort_values(ascending=False)
    ranked_by_tau = result["kendall_tau_mean"].sort_values(ascending=False)
    ranked_by_jaccard = result[f"jaccard_at_{TOP_K}_mean"].sort_values(ascending=False)
    print(f"\nMost stable by cosine: {ranked_by_cosine.index[0]} ({ranked_by_cosine.iloc[0]:.4f})")
    print(f"Most stable by Kendall tau: {ranked_by_tau.index[0]} ({ranked_by_tau.iloc[0]:.4f})")
    print(f"Most stable by Jaccard@{TOP_K}: {ranked_by_jaccard.index[0]} ({ranked_by_jaccard.iloc[0]:.4f})")

    return result


if __name__ == "__main__":
    result = main()
