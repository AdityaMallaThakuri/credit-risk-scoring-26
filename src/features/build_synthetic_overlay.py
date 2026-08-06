"""Build the synthetic period/bias overlay for drift + fairness ground truth.

Per docs/roadmap.md Phase 1 and CLAUDE.md's data-strategy rules:
- `application_train` rows are partitioned into 5 synthetic periods (P0
  reference, P1-P4) by row assignment only -- SK_ID_CURR carries no
  chronological meaning in this dataset and is never used as a time proxy.
- Covariate drift is injected via weighted resampling (rejection/importance
  sampling), not by fabricating new feature values.
- Exactly one period (the "shock" period) additionally gets concept drift:
  the income-TARGET relationship itself is flattened there.
- A known bias is planted on CODE_GENDER via label-side injection (see
  BIAS section below for the explicit rationale) in P1-P4 only; P0 stays
  a clean, bias-free reference.
- Output is a derived *assignment* table (SK_ID_CURR + period + replicate
  id + TARGET override), not a duplicated copy of application_train.
  Downstream code re-joins to application_train on SK_ID_CURR.
- `data/raw/` is never written to. Everything here lands in
  `data/synthetic/`.

Run: python src/features/build_synthetic_overlay.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import RAW_DIR, handle_days_employed_anomaly

SEED = 42
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"

PERIODS = ["P0", "P1", "P2", "P3", "P4"]
PSI_TARGETS = {"P0": 0.0, "P1": 0.15, "P2": 0.30, "P3": 0.45, "P4": 0.60}
SHOCK_PERIOD = "P3"  # gets concept drift in addition to covariate drift
CONCEPT_DRIFT_DAMPING = 0.3  # fraction of the natural income->TARGET signal RETAINED in the shock period

DRIFT_FEATURES = {
    # feature: direction of drift with increasing period (+1 = toward higher bins, -1 = toward lower bins)
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    "AMT_INCOME_TOTAL": +1,
}
N_BINS_TARGET = 10

BIAS_ATTRIBUTE = "CODE_GENDER"
BIAS_DISADVANTAGED_GROUP = "F"
BIAS_FLIP_RATE = 0.08  # of TARGET==0 rows in the disadvantaged group, in bias-affected periods
BIAS_PERIODS = ["P1", "P2", "P3", "P4"]  # P0 stays clean


# --------------------------------------------------------------------------- #
# Binning helpers (deterministic PSI via analytic expected proportions, so
# calibration doesn't fight sampling noise -- only the final draw is random)
# --------------------------------------------------------------------------- #

def get_bin_edges(reference_values, n_bins=N_BINS_TARGET):
    finite = reference_values.dropna()
    _, edges = pd.qcut(finite, q=n_bins, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(values, edges):
    """Returns integer bin index 0..len(edges)-2, or n_bins (=len(edges)-1) for NaN."""
    n_bins = len(edges) - 1
    codes = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return codes.fillna(n_bins).astype(int).to_numpy()


def bin_proportions(bin_codes, n_bins_plus_missing, weights=None):
    if weights is None:
        weights = np.ones(len(bin_codes))
    totals = np.zeros(n_bins_plus_missing)
    for b in range(n_bins_plus_missing):
        totals[b] = weights[bin_codes == b].sum()
    return totals / totals.sum()


def psi(expected_props, observed_props, eps=1e-6):
    e = np.clip(expected_props, eps, None)
    o = np.clip(observed_props, eps, None)
    return float(np.sum((o - e) * np.log(o / e)))


# --------------------------------------------------------------------------- #
# Covariate-drift calibration: coordinate-descent bisection on a per-feature
# "strength" so the *analytic expected* PSI (weights, not a random draw)
# hits the period's target for every drift feature simultaneously.
# --------------------------------------------------------------------------- #

def combined_weights(bin_codes_by_feature, n_bins_by_feature, strengths):
    n_rows = len(next(iter(bin_codes_by_feature.values())))
    log_w = np.zeros(n_rows)
    for feat, codes in bin_codes_by_feature.items():
        n_bins = n_bins_by_feature[feat]
        center = (n_bins - 1) / 2
        direction = DRIFT_FEATURES[feat]
        is_missing = codes == n_bins
        exponent = strengths[feat] * direction * (codes - center)
        exponent = np.where(is_missing, 0.0, exponent)
        log_w += np.clip(exponent, -30, 30)
    return np.exp(log_w - log_w.max())


def calibrate_strengths(bin_codes_by_feature, n_bins_by_feature, ref_props_by_feature, target_psi, rounds=4):
    strengths = {f: 0.0 for f in DRIFT_FEATURES}
    for _ in range(rounds):
        for feat in DRIFT_FEATURES:
            def measured_psi(s):
                trial = dict(strengths)
                trial[feat] = s
                w = combined_weights(bin_codes_by_feature, n_bins_by_feature, trial)
                props = bin_proportions(bin_codes_by_feature[feat], n_bins_by_feature[feat] + 1, weights=w)
                return psi(ref_props_by_feature[feat], props)

            lo, hi = 0.0, 1.0
            while measured_psi(hi) < target_psi and hi < 50:
                hi *= 2
            for _ in range(40):
                mid = (lo + hi) / 2
                if measured_psi(mid) < target_psi:
                    lo = mid
                else:
                    hi = mid
            strengths[feat] = (lo + hi) / 2
    return strengths


# --------------------------------------------------------------------------- #
# Concept drift: within the shock period, flatten the income -> TARGET
# relationship by reweighting TARGET within income quartiles toward the
# pool's overall default rate (damped, not fully erased).
# --------------------------------------------------------------------------- #

def concept_drift_weights(income_values, target_values, damping=CONCEPT_DRIFT_DAMPING):
    quartile = pd.qcut(income_values, q=4, labels=False, duplicates="drop")
    overall_rate = target_values.mean()
    weights = np.ones(len(target_values), dtype=float)
    for q in np.unique(quartile[~pd.isna(quartile)]):
        mask_q = quartile == q
        n1 = (target_values[mask_q] == 1).sum()
        n0 = (target_values[mask_q] == 0).sum()
        if n1 == 0 or n0 == 0:
            continue
        natural_rate = n1 / (n1 + n0)
        flattened_rate = overall_rate + damping * (natural_rate - overall_rate)
        w1 = flattened_rate / natural_rate
        w0 = (1 - flattened_rate) / (1 - natural_rate)
        weights[mask_q & (target_values == 1).to_numpy()] = w1
        weights[mask_q & (target_values == 0).to_numpy()] = w0
    return weights


def default_rate_by_income_quartile(income_values, target_values):
    quartile = pd.qcut(income_values, q=4, labels=False, duplicates="drop")
    return pd.Series(target_values).groupby(quartile).mean().to_dict()


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(SEED)

    app = pd.read_csv(RAW_DIR / "HC_application_train.csv")
    app = handle_days_employed_anomaly(app)

    log = {
        "seed": SEED,
        "source_table": "HC_application_train.csv",
        "n_source_rows": len(app),
        "periods": {},
        "drift_features": DRIFT_FEATURES,
        "psi_targets": PSI_TARGETS,
        "shock_period": SHOCK_PERIOD,
        "concept_drift_damping": CONCEPT_DRIFT_DAMPING,
        "bias_attribute": BIAS_ATTRIBUTE,
        "bias_disadvantaged_group": BIAS_DISADVANTAGED_GROUP,
        "bias_flip_rate": BIAS_FLIP_RATE,
        "bias_periods": BIAS_PERIODS,
        "bias_note": (
            "Bias is planted on the LABEL side, not the decision side: for a "
            f"random {BIAS_FLIP_RATE:.0%} of true non-defaulters "
            f"(TARGET==0) in group {BIAS_DISADVANTAGED_GROUP}, "
            "TARGET is flipped to 1 in bias-affected periods only. This models "
            "historical/selection bias (worse recorded outcomes for a group "
            "independent of true creditworthiness), per reading_material.md "
            "Part 6.1. It is a synthetic ground-truth construct for testing "
            "detection methods, not a claim about any real population."
        ),
    }

    # --- 1. Partition into 5 periods, stratified by TARGET so base rate matches pre-injection ---
    period_assignment = pd.Series(index=app.index, dtype=object)
    for target_value, group in app.groupby("TARGET"):
        idx = rng.permutation(group.index.to_numpy())
        splits = np.array_split(idx, len(PERIODS))
        for period, split_idx in zip(PERIODS, splits):
            period_assignment.loc[split_idx] = period
    app["_period_base"] = period_assignment

    # --- 2. Reference bin edges/proportions from P0 ---
    p0 = app[app["_period_base"] == "P0"]
    ref_edges, ref_props, n_bins_by_feature = {}, {}, {}
    for feat in DRIFT_FEATURES:
        edges = get_bin_edges(p0[feat])
        ref_edges[feat] = edges
        n_bins_by_feature[feat] = len(edges) - 1
        codes = assign_bins(p0[feat], edges)
        ref_props[feat] = bin_proportions(codes, n_bins_by_feature[feat] + 1)

    overlay_rows = []
    row_id_counter = 0

    # P0: no injection at all, kept as-is (one-to-one)
    for orig_idx in p0.index:
        overlay_rows.append({
            "synthetic_row_id": row_id_counter,
            "SK_ID_CURR": int(app.loc[orig_idx, "SK_ID_CURR"]),
            "period": "P0",
            "is_shock_period": False,
            "TARGET_original": int(app.loc[orig_idx, "TARGET"]),
            "TARGET": int(app.loc[orig_idx, "TARGET"]),
            "bias_flipped": False,
        })
        row_id_counter += 1
    log["periods"]["P0"] = {"n_rows": len(p0), "psi_target": 0.0, "psi_measured": {f: 0.0 for f in DRIFT_FEATURES}}

    # --- 3. P1-P4: covariate drift (+ concept drift in the shock period) via weighted resample ---
    for period in PERIODS[1:]:
        pool = app[app["_period_base"] == period]
        bin_codes_by_feature = {feat: assign_bins(pool[feat], ref_edges[feat]) for feat in DRIFT_FEATURES}

        strengths = calibrate_strengths(bin_codes_by_feature, n_bins_by_feature, ref_props, PSI_TARGETS[period])
        weights = combined_weights(bin_codes_by_feature, n_bins_by_feature, strengths)

        is_shock = period == SHOCK_PERIOD
        pre_shock_income_default = None
        if is_shock:
            pre_shock_income_default = default_rate_by_income_quartile(pool["AMT_INCOME_TOTAL"], pool["TARGET"])
            concept_w = concept_drift_weights(pool["AMT_INCOME_TOTAL"], pool["TARGET"])
            weights = weights * concept_w

        probs = weights / weights.sum()
        sampled_positions = rng.choice(len(pool), size=len(pool), replace=True, p=probs)
        sampled = pool.iloc[sampled_positions].reset_index(drop=True)

        post_shock_income_default = None
        if is_shock:
            post_shock_income_default = default_rate_by_income_quartile(sampled["AMT_INCOME_TOTAL"], sampled["TARGET"])

        # --- 4. Planted bias: label-flip a fraction of disadvantaged-group non-defaulters ---
        bias_flipped = pd.Series(False, index=sampled.index)
        target = sampled["TARGET"].copy()
        if period in BIAS_PERIODS:
            eligible = (sampled[BIAS_ATTRIBUTE] == BIAS_DISADVANTAGED_GROUP) & (sampled["TARGET"] == 0)
            eligible_idx = sampled.index[eligible]
            n_flip = int(round(len(eligible_idx) * BIAS_FLIP_RATE))
            flip_idx = rng.choice(eligible_idx, size=n_flip, replace=False) if n_flip > 0 else np.array([], dtype=int)
            target.loc[flip_idx] = 1
            bias_flipped.loc[flip_idx] = True

        for i in sampled.index:
            overlay_rows.append({
                "synthetic_row_id": row_id_counter,
                "SK_ID_CURR": int(sampled.loc[i, "SK_ID_CURR"]),
                "period": period,
                "is_shock_period": is_shock,
                "TARGET_original": int(sampled.loc[i, "TARGET"]),
                "TARGET": int(target.loc[i]),
                "bias_flipped": bool(bias_flipped.loc[i]),
            })
            row_id_counter += 1

        measured_psi = {}
        for feat in DRIFT_FEATURES:
            codes = assign_bins(sampled[feat], ref_edges[feat])
            props = bin_proportions(codes, n_bins_by_feature[feat] + 1)
            measured_psi[feat] = psi(ref_props[feat], props)

        log["periods"][period] = {
            "n_rows": len(sampled),
            "psi_target": PSI_TARGETS[period],
            "psi_measured": measured_psi,
            "calibration_strengths": strengths,
            "is_shock_period": is_shock,
        }
        if is_shock:
            log["periods"][period]["income_default_rate_by_quartile_pre_concept_drift"] = pre_shock_income_default
            log["periods"][period]["income_default_rate_by_quartile_post_concept_drift"] = post_shock_income_default

        n_flipped = int(bias_flipped.sum())
        n_eligible = int(eligible.sum()) if period in BIAS_PERIODS else 0
        log["periods"][period]["bias_flips"] = {
            "n_eligible": n_eligible,
            "n_flipped": n_flipped,
            "flip_rate_achieved": (n_flipped / n_eligible) if n_eligible else 0.0,
        }

    overlay = pd.DataFrame(overlay_rows)

    # --- 5. Bias-magnitude validation: default-rate gap by gender, per period ---
    merged = overlay.merge(
        app[["SK_ID_CURR", "CODE_GENDER"]].drop_duplicates("SK_ID_CURR"),
        on="SK_ID_CURR", how="left",
    )
    for period in PERIODS:
        sub = merged[merged["period"] == period]
        rates = sub.groupby("CODE_GENDER")["TARGET"].mean()
        rate_f = rates.get(BIAS_DISADVANTAGED_GROUP, float("nan"))
        other_groups = [g for g in rates.index if g != BIAS_DISADVANTAGED_GROUP and g != "XNA"]
        rate_other = sub[sub["CODE_GENDER"].isin(other_groups)]["TARGET"].mean()
        log["periods"][period]["default_rate_gap_gender"] = {
            "rate_disadvantaged_group": float(rate_f),
            "rate_other_groups": float(rate_other),
            "gap": float(rate_f - rate_other),
        }

    # --- 6. Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OUT_DIR / "synthetic_overlay.parquet"
    log_path = OUT_DIR / "injection_log.json"
    overlay.to_parquet(overlay_path, index=False)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"Saved overlay ({len(overlay):,} rows) -> {overlay_path}")
    print(f"Saved injection log -> {log_path}")
    print()
    print_validation_report(log)


def print_validation_report(log):
    print("=== Validation: target vs measured PSI ===")
    for period in PERIODS:
        info = log["periods"][period]
        print(f"\n{period} (target PSI={info['psi_target']:.2f}, n={info['n_rows']:,}"
              f"{', SHOCK PERIOD' if info.get('is_shock_period') else ''})")
        for feat, val in info["psi_measured"].items():
            print(f"    {feat:20s} measured PSI = {val:.3f}")

    shock_info = log["periods"][SHOCK_PERIOD]
    print(f"\n=== Concept drift check ({SHOCK_PERIOD}): income-quartile default rates ===")
    print(f"  pre-injection (natural):  {shock_info['income_default_rate_by_quartile_pre_concept_drift']}")
    print(f"  post-injection (flattened): {shock_info['income_default_rate_by_quartile_post_concept_drift']}")

    print("\n=== Planted bias: default-rate gap, disadvantaged group vs rest ===")
    for period in PERIODS:
        gap_info = log["periods"][period]["default_rate_gap_gender"]
        flips = log["periods"][period]["bias_flips"]
        print(f"{period}: gap={gap_info['gap']:+.4f}  "
              f"(disadvantaged={gap_info['rate_disadvantaged_group']:.4f}, "
              f"other={gap_info['rate_other_groups']:.4f})  "
              f"flips={flips['n_flipped']:,}/{flips['n_eligible']:,} "
              f"(target rate={BIAS_FLIP_RATE:.0%}, achieved={flips['flip_rate_achieved']:.2%})")


if __name__ == "__main__":
    main()
