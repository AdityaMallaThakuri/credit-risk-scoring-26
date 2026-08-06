---
name: leakage-check
description: Checklist for data leakage before finalizing any preprocessing, resampling, or model-training code in this project. Use whenever writing or reviewing code that involves train/test splitting, SMOTE or other resampling, scaling, imputation, or feature construction from multi-table joins.
---

Apply this checklist to any preprocessing or modeling code before treating
it as final. This project has a documented history of exactly these bugs
appearing in naive/reference implementations — check explicitly rather
than assuming they're absent.

## Checklist

1. **Split order.** Confirm `train_test_split` (or the expanding-window /
   stratified-k-fold equivalent) happens BEFORE any resampling (SMOTE,
   over/undersampling). If you see `smote.fit_resample(X, y)` followed
   later by a split on the resampled output, that's leakage — fix by
   splitting first, resampling only `X_train`/`y_train`.

2. **Scaler/imputer fitting.** Confirm `.fit()` or `.fit_transform()` is
   called only on training data. Validation/test data should only see
   `.transform()`. Flag any `scaler.fit_transform(full_dataframe)` called
   before a split exists.

3. **Temporal validity.** For any feature, ask: "would this value actually
   be known at the moment of the real lending decision?" Aggregate
   features built from `previous_application`, `bureau`, `installments`,
   etc. must only use records that predate the current application in a
   way the data actually supports — check this explicitly for Home Credit
   given there's no absolute timestamp (see CLAUDE.md's data strategy
   section).

4. **Target leakage in joins.** When merging TARGET onto row-level
   auxiliary tables (bureau, installments) for EDA/plotting, confirm this
   merged frame is never fed into actual feature construction or model
   training — it's fine for descriptive plots only.

5. **SHAP additive consistency.** If SHAP values have been modified
   (reweighted, rebaselined, or blended as in Weighted Temporal SHAP),
   confirm `sum(shap_values) == prediction - base_value` still holds
   after renormalization, within floating-point tolerance.

6. **Synthetic/real separation.** Confirm no code path lets the synthetic
   period/bias overlay leak into the Phase 3 real-data baseline numbers,
   and vice versa — these two are deliberately kept separate per the
   project's data strategy.

Report any violation found with the specific line/cell and the fix,
rather than silently correcting it — the person should know what was
wrong and why.
