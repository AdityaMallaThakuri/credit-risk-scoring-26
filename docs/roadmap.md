# Master's Final Year Project — Complete Roadmap
## Explainable, Fair, and Drift-Aware Credit Risk Scoring System

**Student:** Pujan Malla Thakuri
**Deliverables:** (1) a working credit-risk application, (2) a dissertation, (3) a conference/workshop paper draft
**Data strategy locked in:** Home Credit Default Risk as the real-data foundation, with a clearly-labeled synthetic period/bias overlay for drift and fairness ground truth (see prior discussion for rationale)
**Assumed duration:** 24 weeks — rescale proportionally to your actual deadline; the phase *order* and *dependencies* stay fixed regardless of total length

---

## 0. What "done" looks like at the end

- A deployed (or locally-runnable, demoable) dashboard application where a user can view portfolio risk, inspect SHAP explanations, see fairness metrics over synthetic periods, and move a threshold slider to see live Expected Loss / approval-rate / fairness impact
- A trained, validated, calibrated model pipeline behind it, benchmarked across 4 algorithms
- A working implementation of Weighted Temporal SHAP, evaluated against static SHAP and the replicated baseline methods
- A dissertation document covering background, methodology, results, and limitations
- A paper draft in a submittable format for a workshop/conference venue

---

## 1. Timeline Overview

| Phase | Weeks | Focus | Key Deliverable |
|---|---|---|---|
| 1. Data Foundation | 1–3 | Real-data pipeline + synthetic overlay design | Clean, validated dataset + documented synthetic-period/bias mechanism |
| 2. Feature Engineering | 4–5 | SQL/feature pipeline on real data | Full feature set, leakage-checked |
| 3. Modeling Core | 6–8 | 4 models, validation, calibration, EL | Benchmarked models with calibration + cost-sensitive thresholds |
| 4. Explainability + Fairness Baseline | 9–10 | Static SHAP, fairness metrics, sanity checks, fairness-accuracy trade-off report | Verified static baseline, confirmed synthetic bias is detectable, standalone trade-off table |
| 5. Drift Replication | 11–12 | Methods A/B/C on the synthetic overlay | Replication results table |
| 6. Weighted Temporal SHAP | 13–15 | Your algorithmic contribution | Working implementation + ablation results |
| 7. Application Architecture & Build | 16–18 | Backend + frontend + integration | Working application (local or deployed) |
| 8. Testing & Validation | 19–20 | Model tests, app tests, stress tests | Test suite + validation report |
| 9. Paper & Dissertation Writing | throughout, finalized 21–23 | Continuous writing, not a single phase | Draft dissertation + paper |
| 10. Final Polish & Submission | 24 | Fixes, packaging, submission | Submitted dissertation, paper, app |

**Important:** Phase 9 (writing) is listed as its own row but should run *continuously* from Phase 1 onward, not start at week 21. Building and writing at the same time is what keeps this achievable — see Section 4.

---

## 2. Detailed Phase Breakdown

### Phase 1 — Data Foundation (Weeks 1–3)

**Tasks:**
1. Load and clean the 7 Home Credit tables; handle the known `DAYS_EMPLOYED` sentinel-value anomaly (365243 placeholder, ~17.4% of rows) explicitly
2. Design the synthetic period-assignment mechanism: how many periods, which features get reweighted, target PSI values per period, which period gets concept-drift (not just covariate-drift) injection
3. Design the synthetic bias-planting mechanism: which attribute (real, like gender, or a separate synthetic proxy), what magnitude, logged explicitly
4. Build the derived "period/bias assignment" table — **kept separate from raw data**, never edited into the original CSVs
5. Validate the synthetic design: confirm injected PSI values land where intended, confirm planted bias is measurable at the magnitude you set

**Exit criteria:** raw data untouched and versioned separately; synthetic overlay table exists with a full, reviewable log of what was injected, where, and at what magnitude.

---

### Phase 2 — Feature Engineering (Weeks 4–5)

**Tasks:**
1. Build the SQL pipeline: CTEs and window functions for rolling utilization, aggregation of bureau/previous-application/installments tables up to `SK_ID_CURR`
2. Engineer the feature set: DTI, repayment velocity, delinquency recency, spending volatility, credit utilization trend, application-time signals (`WEEKDAY_APPR_PROCESS_START`, `HOUR_APPR_PROCESS_START`), ratio-stacked features
3. Run a leakage check pass: confirm no feature uses information that wouldn't be available at the actual moment of the lending decision
4. Run a multicollinearity check (VIF) on the final feature set

**Exit criteria:** a single, versioned feature-engineering pipeline (one script/notebook, not duplicated logic across models) producing a clean feature table ready for modeling.

---

### Phase 3 — Modeling Core (Weeks 6–8)

**Tasks:**
1. Implement stratified k-fold validation for the real-data baseline (per our data-strategy decision — real headline numbers, no synthetic dependency here)
2. Train and tune Logistic Regression, Random Forest, XGBoost, LightGBM
3. Class imbalance handling: **split first, then apply SMOTE only to the training fold** — this is a hard rule, confirmed necessary after finding this exact bug pervasive in the reference notebooks
4. Calibration: reliability diagrams for each model; apply Platt scaling or isotonic regression if needed
5. Cost-sensitive threshold selection + Expected Loss (PD × LGD × EAD), with LGD/EAD simulated at plausible values since Home Credit doesn't provide them directly

**Exit criteria:** a comparison table across all 4 models (AUC, F1, KS, calibration quality, EL at optimal threshold) — this becomes your dissertation's baseline-modeling results section.

---

### Phase 4 — Explainability + Fairness Baseline (Weeks 9–10)

**Tasks:**
1. Static SHAP (global, local, interaction values) on your best-performing calibrated model
2. Fairness metrics (DPD, EOD, Equalized Odds) computed on the real attributes (gender, family status, age bands) as a baseline, **and** on your synthetic-period data to confirm the planted bias is detectable
3. Sanity-check: does your fairness pipeline actually find the bias you planted, at roughly the magnitude you set? If not, fix the injection mechanism before proceeding — everything after this phase assumes this works
4. **Fairness-accuracy trade-off report** (this fulfils your original Objective 5 explicitly, not just implicitly): for each fairness-mitigation step or threshold adjustment you test, record the accuracy/AUC/Expected-Loss cost alongside the fairness gain. Produce a short table or chart showing, in concrete numbers, what you gave up in model performance to achieve each unit of fairness improvement — this is a named deliverable, not an incidental by-product of Phase 4's metrics

**Exit criteria:** documented evidence that your synthetic ground truth (drift + bias) is detectable by standard methods, validating that your evaluation framework is sound before you test your own method against it — **plus** a standalone fairness-accuracy trade-off table/report with real numbers from your own models.

---

### Phase 5 — Drift Replication (Weeks 11–12)

**Tasks:**
1. Implement PSI, KS test, JS divergence drift detection on your synthetic period overlay
2. Implement and run Methods A (drift-weighted), B (sliding background), C (Ridge surrogate) — the concept-drift paper's methods — on your synthetic periods
3. Compute stability metrics: cosine similarity, Kendall tau, Jaccard@k, same as the original paper, so your results are directly comparable

**Exit criteria:** a replication table — does Method B still win on your independently-constructed data? Report honestly either way; a different result is still a valid, interesting finding.

---

### Phase 6 — Weighted Temporal SHAP (Weeks 13–15)

**Tasks:**
1. Implement the cost-aware blended weighting: `Weighted_SHAP(j,t) = SHAP(j,t) × [α·w_drift(j,t) + (1−α)·w_cost(j,t)]`
2. Implement `w_cost` via counterfactual Expected Loss re-simulation
3. Implement additive-consistency renormalization (SHAP values must still sum to prediction − base value)
4. Run the ablation: α = 1 (≡ Method A), α = 0 (pure cost), 2–3 blended values
5. Run the two comparisons that make this a contribution: stability vs. static/Method B, and **fairness reduction** vs. Method B — using your planted-bias ground truth from Phase 4 to verify the fairness claim is real, not incidental

**Exit criteria:** working implementation, completed ablation table, a one-sentence, stateable finding (e.g., "cost-aware weighting improves fairness reduction by X% over Method B at comparable explanation stability").

---

### Phase 7 — Application Architecture & Build (Weeks 16–18)

This is the part that makes it a Master's-level *application*, not just a research notebook. Treat this as its own engineering project with its own design decisions.

**7.1 Architecture decisions (make these explicit, write them down for your dissertation's system-design section):**
- **Backend:** a Python service (FastAPI or Flask) exposing endpoints for: applicant scoring, SHAP explanation retrieval, fairness metrics by period, drift metrics by period, Expected Loss/threshold simulation
- **Frontend/dashboard:** Streamlit or Dash for fastest build time, or a lightweight React frontend if you want a more "product-like" look and already have frontend experience — don't over-invest in frontend polish at the expense of the modeling work behind it
- **Data layer:** your SQL pipeline output persisted (SQLite is sufficient for a single-user demo; Postgres if you want to show more "production-like" thinking)
- **Model serving:** load the trained/calibrated model + SHAP explainer once at startup, not per-request, for reasonable responsiveness

**7.2 Core screens to build, in this order (build and demo incrementally, don't attempt all at once):**
1. Portfolio overview (aggregate stats, approval/default rates)
2. **Segments view** (fulfils your original Objective 7's "segments" requirement explicitly): breaks the portfolio down by risk tier (e.g., low/medium/high predicted risk, bucketed from model output) and by demographic group — reuses data already computed for fairness metrics and model scores, so this is presentation work on top of existing outputs, not new modeling
3. Individual applicant view with local SHAP explanation in plain language
4. Fairness-over-time view across your synthetic periods, including the fairness-accuracy trade-off numbers from Phase 4
5. Drift monitoring view (PSI/KS/JS by feature, by period)
6. **The integrated policy simulator** — threshold slider that live-updates approval rate, Expected Loss/revenue, fairness metrics, and a sample applicant's drift-adjusted SHAP explanation simultaneously — this is your centerpiece screen

**Exit criteria:** a working, runnable application covering all 6 screens, even if visual polish is still rough — functional completeness before cosmetic polish.

---

### Phase 8 — Testing & Validation (Weeks 19–20)

A Master's application deliverable needs actual testing, not just "it ran once and worked."

**8.1 Model validation tests:**
- Confirm calibration holds on a held-out fold, not just training data
- Confirm the cost-sensitive threshold produces sensible behavior at the extremes (threshold=0 approves everyone, threshold=1 approves no one — sanity-check the profit curve shape)
- Confirm SHAP values satisfy additive consistency (sum to prediction − base value) after your Weighted Temporal SHAP renormalization

**8.2 Application tests:**
- Basic functional tests for each backend endpoint (does it return the right shape of data, does it handle a missing/edge-case applicant gracefully)
- Manual walkthrough testing of the policy simulator — does every linked metric actually update when the slider moves, or does something silently go stale

**8.3 Stress tests:**
- Very small subgroup sizes (fairness metrics get noisy/unreliable with few samples — does your dashboard handle or flag this, rather than showing a misleadingly confident number)
- Extreme/missing feature values

**Exit criteria:** a short validation report documenting what was tested and what passed/failed, included as an appendix in your dissertation — this is what makes the "reliable enough to show a bank" claim credible rather than asserted.

---

### Phase 9 — Paper & Dissertation Writing (continuous, finalized Weeks 21–23)

**Do not treat this as a single block at the end.** Write each section as its corresponding phase completes — by week 20 you should have rough drafts of every section already, and weeks 21–23 are about integration, polish, and coherence, not first-drafting everything from scratch.

**Writing schedule aligned to build phases:**
- After Phase 1: data section, synthetic-overlay methodology and justification
- After Phase 2: feature engineering section
- After Phase 3: baseline modeling results section
- After Phase 4: explainability + fairness baseline section, **including the standalone fairness-accuracy trade-off write-up** (your original Objective 5's explicit deliverable — write this as its own subsection with a real table/chart from your results, not folded silently into the fairness metrics discussion)
- After Phase 5: drift replication results section
- After Phase 6: your core algorithm section — method + experiments (the heart of the paper)
- After Phase 7: system design/architecture section
- After Phase 8: validation/testing section, honest limitations section

**Paper-specific additions (on top of the dissertation content):**
- Trim to workshop paper length/format once the dissertation draft is complete — the paper is a condensed, reframed subset of the dissertation, not a separate writing effort from scratch
- Related work section (your five anchor papers), formal contribution statement, abstract — write these last, once you know exactly what you're claiming

**Exit criteria by week 23:** full dissertation draft reviewed by supervisor at least once; paper draft ready for a peer read before submission.

---

### Phase 10 — Final Polish & Submission (Week 24)

- Incorporate supervisor/peer feedback
- Final application packaging: clean repo, README, reproducibility instructions
- Final proofread and formatting pass on both documents
- Submit dissertation; submit paper to chosen venue (or arXiv + university showcase if timing doesn't align with a conference deadline)

---

## 3. Application Deliverable — What "Proper" Means Here

Since this is explicitly a Master's FYP requiring a real application, hold yourself to these standards, not just "it technically runs":

- [ ] Runs from a clean install with documented setup steps (not just "it works on my machine")
- [ ] Has basic error handling — doesn't crash on missing/malformed input
- [ ] Has a validation/testing record (Phase 8), not just anecdotal "I tried it and it worked"
- [ ] The policy simulator's linked updates actually work live, not just as separate static screens
- [ ] Includes a short technical README explaining architecture decisions, suitable for a technical reviewer or examiner to read in five minutes

**Original proposal objective coverage check (all 7 objectives, confirmed):**
1. SQL pipeline (CTEs, window functions, cohort analysis) → Phase 2 ✅
2. Behavioral/financial features (DTI, repayment velocity, delinquency recency, spending volatility) → Phase 2 ✅
3. Multi-model benchmarking (LogReg, RF, XGBoost, LightGBM) on academic + business metrics → Phase 3 ✅
4. SHAP explainability, global + local, in an interactive dashboard → Phase 4 + Phase 7 screen 3 ✅
5. Fairness/bias analysis with explicit fairness-accuracy trade-off report → Phase 4 (now includes the standalone trade-off deliverable) ✅
6. Policy simulation (threshold variation, default rate/approval rate/EL/revenue) → Phase 3 + Phase 7 screen 6 ✅
7. Interactive dashboard covering portfolio overview, **segments**, SHAP panels, threshold simulation → Phase 7 screens 1–6 (segments view now explicit) ✅

---

## 4. Managing the Two Workloads Together (Build + Write)

The single biggest risk in a project like this is treating writing as something that happens "after" building. Concretely, block your calendar like this:
- **Within every 2-week phase block above:** spend roughly 80% of time building, 20% writing the corresponding section, in the same window — not sequentially
- **Never let more than one phase pass without its section drafted** — if you're at the start of Phase 5 and Phase 3's results section still isn't drafted, stop and write it before continuing; catching up on 3 phases of unwritten results at once is where dissertations go over deadline

---

## 5. Immediate Next Action

Phase 1 is next, and its first concrete task is the one we haven't finalized yet: the exact synthetic period-assignment and bias-planting design (number of periods, which features, target PSI values, magnitude of planted bias). That design needs to be locked before any pipeline code gets written, since Phases 2 through 6 all build on top of it.
