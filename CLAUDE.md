# Credit Risk FYP — Project Instructions

Master's FYP: explainable, fair, drift-aware credit risk scoring system.
Student: Pujan Malla Thakuri. Full context lives in `docs/` — read
`docs/roadmap.md` before starting any new phase of work, and
`docs/reading_material.md` for concept explanations if unfamiliar with a
term (SHAP, PSI, Expected Loss, etc.).

## Data strategy (do not deviate without explicit confirmation)

- **Real data**: Home Credit Default Risk dataset in `data/raw/`. This is
  the foundation for all core modeling, SQL pipeline, and static SHAP work.
- **No absolute calendar time exists in this data.** Every date field is
  relative to each applicant's own application. `SK_ID_CURR` does NOT
  encode chronological order — this was tested empirically and confirmed
  false. Never use `SK_ID_CURR` order as a time proxy.
- **Synthetic overlay**: a separate, clearly-logged layer in
  `data/synthetic/` that assigns real rows to synthetic "periods" with
  deliberately injected drift (via reweighting/resampling, not new fake
  rows) and a planted bias with known ground truth. Used ONLY for:
  drift monitoring, fairness-over-time tracking, and evaluating the
  Weighted Temporal SHAP method. Never used for headline model-comparison
  numbers (Phase 3) — those use real data with stratified k-fold only.
- **Never edit files in `data/raw/`.** Treat as read-only. All derived
  data goes in `data/processed/` or `data/synthetic/`.

## Hard rules (violating these invalidates results)

- **Split before resample.** Always: train/test split first, then apply
  SMOTE (or any resampling) only to the training fold. Never resample
  before splitting — this causes data leakage and has been found
  repeatedly in reference/naive implementations reviewed for this project.
- **Fit scalers/imputers on training data only.** Use `.fit()` on train,
  `.transform()` (not `.fit_transform()`) on validation/test.
- **SHAP additive consistency.** Any modified SHAP values (e.g. Weighted
  Temporal SHAP) must be renormalized so they still sum to
  `prediction − base_value`. Verify this in tests, not just visually.
- **No feature may use information unavailable at the actual moment of
  the lending decision.** Check every engineered feature against this
  before adding it to the pipeline.

## Reference notebooks — read carefully

Two notebooks under review (`Preprocessing_merged.ipynb`, `EDA_merged.ipynb`
and four model notebooks) are **third-party bootcamp project work**, not
mine. They are reference-only, for understanding a naive baseline
approach. Never copy code, variable names, or structure from them
directly. They contain a confirmed leakage bug (SMOTE before split) —
flag this pattern if it ever appears anywhere in this repo's code.

## Project layout

- `sql/` — feature-extraction pipeline (CTEs, window functions,
  cohort aggregation) against the real Home Credit tables
- `src/features/` — feature engineering (DTI, repayment velocity,
  utilization trend, ratio stacking, etc.)
- `src/models/` — training/validation for LogReg, RF, XGBoost, LightGBM;
  calibration; cost-sensitive threshold + Expected Loss
- `src/explainability/` — static SHAP, Weighted Temporal SHAP, the
  replicated Methods A/B/C
- `src/fairness/` — DPD/EOD/Equalized Odds metrics, fairness-accuracy
  trade-off reporting
- `src/app/` — the dashboard application (portfolio, segments, SHAP
  panel, fairness-over-time, drift monitor, policy simulator)
- `tests/` — see `.claude/skills/leakage-check` before adding any new
  modeling code
- `docs/` — roadmap, reading material, and any written dissertation/
  paper sections as they're drafted

## Working conventions

- Python, pandas/scikit-learn/xgboost/lightgbm/shap. SQL pipeline can be
  SQLite for local dev.
- One shared feature-engineering pipeline — do not duplicate preprocessing
  logic per model notebook (this was a specific weakness identified in
  the reference project's structure).
- Before marking a roadmap phase complete, check its exit criteria in
  `docs/roadmap.md` explicitly — don't just move on because code runs.
- When in doubt about a data-strategy or scope decision (real vs.
  synthetic, what a phase's exit criteria require), ask rather than
  assume — several of these decisions were made deliberately after
  testing alternatives that didn't work.

## Skills available in this repo

- `synthetic-drift-injection` — the exact procedure for building the
  synthetic period/bias overlay with logged ground truth
- `leakage-check` — checklist to run before finalizing any model or
  preprocessing step
- `shap-stability-eval` — computes cosine similarity / Kendall tau /
  Jaccard@k to compare SHAP explanation stability across methods
