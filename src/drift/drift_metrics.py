"""Phase 5 drift detection: PSI, KS test, and Jensen-Shannon divergence on
the synthetic period overlay (data/synthetic/synthetic_overlay.csv), for
the 4 features the overlay deliberately drifted (EXT_SOURCE_1/2/3,
AMT_INCOME_TOTAL; see data/synthetic/README.md and injection_log.json).

Reference period is always P0 (the clean, undrifted period). Each of
P1-P4 is compared against P0 for each drift feature.

PSI implementation deliberately matches src/features/build_synthetic_
overlay.py's own method exactly (10 quantile bins cut from the P0
reference distribution, -inf/inf-capped outer edges, NaN as its own bin,
eps=1e-6 clipping) -- not a coincidence, but so this independent
re-implementation can be checked against injection_log.json's
`psi_measured` as a correctness cross-check on the drift-injection
pipeline itself, in addition to being the metric used going forward for
drift monitoring (this becomes the shared drift-metrics module the Phase 7
dashboard's drift view will also use). A close match confirms this
module's PSI is computed correctly; it is not expected to match to more
than ~1e-9 precision only because `synthetic_overlay.csv`'s row order is
exactly what build_synthetic_overlay.py measured against -- no additional
resampling happens here.

KS test uses the finite values of each feature directly (scipy.stats.
ks_2samp) -- an exact nonparametric test, no binning, most sensitive to
distributional shifts PSI's coarser 10 bins might smooth over. NaN rows
are dropped for this test only (KS needs continuous samples); each
feature's missingness rate per period is reported separately since a
shift in *missingness itself* is a legitimate form of drift that a
finite-values-only KS test cannot see (PSI's dedicated NaN bin does
capture it).

JS divergence reuses PSI's exact same bin edges/proportions (including
the NaN bin), computed in bits (base-2 log), via
scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2 (that function
returns the JS *distance*, i.e. sqrt(divergence); squaring recovers the
divergence, bounded in [0, 1] with base-2 log). Included alongside PSI
because JS divergence is a proper metric (symmetric, bounded) where PSI is
neither -- useful as a second, differently-shaped cross-check on the same
binned distributions.

Run: python src/drift/drift_metrics.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "credit_risk.db"
SYNTHETIC_OVERLAY_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "synthetic_overlay.csv"
INJECTION_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "injection_log.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase5_drift_metrics.csv"

DRIFT_FEATURES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL"]
REFERENCE_PERIOD = "P0"
COMPARISON_PERIODS = ["P1", "P2", "P3", "P4"]
N_BINS = 10
PSI_EPS = 1e-6


def get_bin_edges(reference_values, n_bins=N_BINS):
    finite = reference_values.dropna()
    _, edges = pd.qcut(finite, q=n_bins, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(values, edges):
    """Integer bin index 0..len(edges)-2, or len(edges)-1 (a dedicated bin) for NaN."""
    n_bins = len(edges) - 1
    codes = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return codes.fillna(n_bins).astype(int).to_numpy()


def bin_proportions(bin_codes, n_bins_plus_missing):
    counts = np.bincount(bin_codes, minlength=n_bins_plus_missing).astype(float)
    return counts / counts.sum()


def population_stability_index(ref_props, comp_props, eps=PSI_EPS):
    e = np.clip(ref_props, eps, None)
    o = np.clip(comp_props, eps, None)
    return float(np.sum((o - e) * np.log(o / e)))


def js_divergence_from_props(ref_props, comp_props):
    distance = jensenshannon(ref_props, comp_props, base=2)
    return float(distance ** 2)


def ks_test(ref_values, comp_values):
    ref_finite = ref_values.dropna()
    comp_finite = comp_values.dropna()
    result = ks_2samp(ref_finite, comp_finite)
    return float(result.statistic), float(result.pvalue)


def compute_feature_drift(ref_values, comp_values, n_bins=N_BINS):
    edges = get_bin_edges(ref_values, n_bins)
    n_bins_plus_missing = (len(edges) - 1) + 1

    ref_props = bin_proportions(assign_bins(ref_values, edges), n_bins_plus_missing)
    comp_props = bin_proportions(assign_bins(comp_values, edges), n_bins_plus_missing)

    psi = population_stability_index(ref_props, comp_props)
    js = js_divergence_from_props(ref_props, comp_props)
    ks_stat, ks_pvalue = ks_test(ref_values, comp_values)

    return {
        "psi": psi,
        "js_divergence": js,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "missing_rate_ref": float(ref_values.isna().mean()),
        "missing_rate_comp": float(comp_values.isna().mean()),
    }


def load_period_features():
    overlay = pd.read_csv(SYNTHETIC_OVERLAY_PATH)
    conn = sqlite3.connect(DB_PATH)
    features = pd.read_sql(
        f"SELECT SK_ID_CURR, {', '.join(DRIFT_FEATURES)} FROM application_train", conn
    )
    conn.close()
    return overlay.merge(features, on="SK_ID_CURR", how="left")


def main():
    df = load_period_features()
    with open(INJECTION_LOG_PATH) as f:
        injection_log = json.load(f)

    ref_df = df[df["period"] == REFERENCE_PERIOD]

    rows = []
    for period in COMPARISON_PERIODS:
        comp_df = df[df["period"] == period]
        print(f"\n=== {period} vs {REFERENCE_PERIOD} (target PSI={injection_log['psi_targets'][period]:.2f}) ===")
        for feat in DRIFT_FEATURES:
            metrics = compute_feature_drift(ref_df[feat], comp_df[feat])
            logged_psi = injection_log["periods"][period]["psi_measured"][feat]

            rows.append({
                "period": period,
                "feature": feat,
                "psi_target": injection_log["psi_targets"][period],
                "psi_logged": logged_psi,
                **metrics,
            })
            print(f"  {feat:<18} PSI={metrics['psi']:.4f} (logged={logged_psi:.4f}, target={injection_log['psi_targets'][period]:.2f})  "
                  f"KS={metrics['ks_statistic']:.4f} (p={metrics['ks_pvalue']:.2e})  JS={metrics['js_divergence']:.4f}")

    result = pd.DataFrame(rows)
    result["psi_vs_logged_diff"] = (result["psi"] - result["psi_logged"]).abs()
    result["psi_vs_target_diff"] = (result["psi"] - result["psi_target"]).abs()
    result.to_csv(RESULTS_PATH, index=False)

    print(f"\nMax |measured PSI - logged PSI| across all period/feature pairs: {result['psi_vs_logged_diff'].max():.6f}")
    print(f"Max |measured PSI - target PSI| across all period/feature pairs: {result['psi_vs_target_diff'].max():.4f}")
    print(f"Saved: {RESULTS_PATH}")

    return df, result


if __name__ == "__main__":
    df, result = main()
