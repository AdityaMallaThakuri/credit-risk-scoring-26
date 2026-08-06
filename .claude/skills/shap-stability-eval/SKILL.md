---
name: shap-stability-eval
description: Compute explanation-stability metrics (cosine similarity, Kendall tau, Jaccard@k) to compare SHAP methods across synthetic periods, and compare fairness reduction. Use when evaluating static SHAP vs Method A/B/C vs Weighted Temporal SHAP in Phases 5-6 of the roadmap.
---

Run this comparison whenever evaluating a SHAP-based explanation method
against a baseline across the synthetic periods.

## Metrics to compute, per pair of consecutive periods

1. **Cosine similarity** between mean absolute SHAP value vectors —
   measures whether the overall direction/magnitude of feature importance
   stays consistent.
2. **Kendall tau** rank correlation between feature importance rankings —
   measures whether the *order* of important features stays stable, which
   can degrade even when cosine similarity looks fine.
3. **Jaccard@k** (default k=10) overlap between top-k important features —
   measures whether the same core features remain influential.

Compute all three for every method under comparison (static SHAP, Method
A, Method B, Method C, and Weighted Temporal SHAP) so results are
directly comparable — use the same period boundaries and same k for
every method in a given run.

## Fairness-reduction comparison (required alongside stability)

For each method, also compute the change in Demographic Parity Difference
(DPD) and Equal Opportunity Difference (EOD) relative to static SHAP's
baseline fairness metrics, using the planted-bias ground truth from the
synthetic overlay (see `synthetic-drift-injection` skill). Stability
alone is not sufficient evidence of improvement — report both together.

## Output format

Produce one comparison table with methods as rows and
{cosine, Kendall tau, Jaccard@10, ΔDPD, ΔEOD} as columns, plus a one or
two sentence written interpretation of which method wins on which axis
and whether stability and fairness improvements move together or
trade off against each other.

If asked to run the α-ablation for Weighted Temporal SHAP specifically,
run this same table at α = 1 (≡ Method A), α = 0 (pure cost-weighting),
and 2-3 intermediate blended values, so the blend's effect is visible
against both endpoints.
