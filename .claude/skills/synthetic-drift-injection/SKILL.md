---
name: synthetic-drift-injection
description: Build the synthetic period/bias overlay on top of real Home Credit data, with logged ground truth for drift magnitude and planted bias. Use when constructing or modifying the synthetic drift/fairness evaluation layer for Phases 4-6 of the roadmap.
disable-model-invocation: true
---

Build or modify the synthetic overlay following this exact procedure.
Never edit `data/raw/` — this overlay is a derived table only, stored in
`data/synthetic/`.

## Procedure

1. **Partition, don't reorder.** Partition `application_train` rows into
   5 synthetic periods (P0 reference, P1-P4 "later") by row assignment,
   not by any real column. Stratify so baseline TARGET rate matches
   across periods unless a period is deliberately a shock period.

2. **Covariate drift via reweighting.** For P1-P4, reweight/resample
   specific feature distributions against P0 using rejection sampling or
   importance weighting, targeting explicit PSI values per period
   (e.g. PSI ≈ 0.15, 0.30, 0.45, 0.60 across P1→P4). Target features:
   `EXT_SOURCE_1/2/3`, `AMT_INCOME_TOTAL`, and the engineered DTI/
   utilization ratio features.

3. **Concept drift in one shock period.** In exactly one designated shock
   period, inject concept drift (not just covariate drift): resample
   within TARGET-conditional strata so the feature-outcome relationship
   itself shifts (e.g. weaken the income-default relationship by
   oversampling high-income defaulters in that period only). This matters
   because static SHAP breaks specifically when the feature-outcome
   relationship moves, not just when marginals move.

4. **Plant a known bias.** Layer a synthetic bias mechanism onto a real
   attribute (start with `CODE_GENDER` or an age band) at an explicit,
   logged magnitude — this gives the fairness-reduction claim in Phase 5/6
   a verifiable ground truth instead of relying on whatever unmeasured
   bias already exists in the real data.

5. **Log everything.** Every injection gets recorded: which features,
   which period, target PSI, covariate-only vs. covariate+concept, and
   the planted bias attribute/magnitude. Store this log alongside the
   overlay table in `data/synthetic/` — it's what lets later phases
   sanity-check "does detection find what I planted."

6. **Validate before proceeding.** After building, confirm actual
   measured PSI/JS on the injected features lands near the target values,
   and confirm the planted bias is measurable at roughly the intended
   magnitude via standard fairness metrics (DPD/EOD). If either check
   fails, fix the injection mechanism — do not proceed to Phase 5/6 on
   an unvalidated overlay.

Report the validation results explicitly (target vs. measured PSI,
target vs. measured bias magnitude) rather than assuming success.
