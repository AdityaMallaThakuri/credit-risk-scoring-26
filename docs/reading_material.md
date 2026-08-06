# Credit Risk Assessment System — Complete Reading & Reference Guide

**Student:** Pujan Malla Thakuri
**Purpose of this document:** a single place that explains *every* technical concept your project touches, in plain language, in the order you'll actually need it. Read it once start to finish before writing any code, then use it as a reference while building.

---

## How to use this document

This is organized in the order you will **build**, not the order of your original objectives list. Each part explains:
- **What it is** (plain language)
- **Why your project needs it**
- **How it connects to the part before and after it**

By the end, Part 8 lays out your own proposed contribution, and Part 9 gives you a week-by-week-style build order.

---

# PART 1 — The Business Problem (Background)

## 1.1 What "unsecured lending" actually means

A loan is either **secured** (backed by collateral — a house, a car — the lender can seize if you don't pay) or **unsecured** (backed by nothing but your promise — credit cards, personal loans, most fintech lending). Unsecured lending is riskier for the lender because there's nothing to repossess if the borrower defaults. This is exactly why the decision of *who to lend to* has to be made carefully, and it's why this domain has driven most of the innovation in credit risk modeling.

## 1.2 How lenders have traditionally made this decision

**Credit scorecards.** A point system: your income, job stability, existing debt, and repayment history get converted into a single number (your "credit score"). Above a cutoff → approved. Below → rejected. Built using logistic regression — a simple statistical model that assumes each factor affects risk in a straight, additive line.

**Why this breaks down:**
1. **Non-linearity.** Real risk doesn't move in a straight line. E.g., a small amount of debt relative to income might be harmless, but risk might explode past a certain debt-to-income ratio rather than rising smoothly — logistic regression can't capture this without manual feature engineering.
2. **Bias.** Historical repayment data reflects historical lending patterns, which may have already been biased (e.g., against certain neighborhoods, demographics). A model trained on this data can inherit and amplify that bias, even without ever looking at race or gender directly — through **proxy variables** (features correlated with a protected attribute, like zip code standing in for race).
3. **Opacity.** A rejected applicant is often told "your score was too low" with no real explanation. Regulators increasingly find this unacceptable.

## 1.3 The regulatory backdrop (why explainability isn't optional)

- **ECOA (US)** — Equal Credit Opportunity Act: if you reject someone, you must give the *specific* principal reasons ("adverse action notice").
- **GDPR (EU)** — gives individuals a "right to explanation" for automated decisions that significantly affect them.
- **Basel Accords / IFRS 9** — require banks to estimate **Probability of Default (PD)**, **Loss Given Default (LGD)**, and **Exposure at Default (EAD)** for regulatory capital purposes — this is where your "Expected Loss" business layer comes from; it's not invented for this project, it's industry-standard.

## 1.4 What your project is actually trying to prove

Not "can I build a model that predicts default" — lots of student projects do that. Your project's actual thesis is:

> **A credit model that predicts well the day it's built will not necessarily predict well, explain itself well, or treat groups fairly two years later — and a well-designed system should be able to show you when and why that's happening, not just quietly decay.**

Everything downstream — the synthetic multi-year data, the drift page, the fairness-over-time tracking — exists to support that thesis. Keep this sentence in mind; it's your compass when you're not sure why a step matters.

---

# PART 2 — Data Engineering Foundations

## 2.1 Why you're building synthetic data (and why this is a legitimate research choice, not a shortcut)

Real bank data is confidential and you have no access to it. But more importantly for your project: **real data doesn't come with ground truth about bias or drift.** If you build synthetic data where *you* control exactly how much drift and how much bias exists, you can test whether your fairness/drift detection actually finds what you planted. This is a stronger validation approach than using a real dataset where you'd never know the "true" answer.

## 2.2 The core design decision: a multi-year panel, not a snapshot

Instead of one table of "applicants," you build a table that spans several **simulated years** (e.g., 2018–2026), where:
- Applicant volume and profile shift gradually year to year (normal economic drift)
- You inject **one or two deliberate shocks** — e.g., a simulated recession year where unemployment spikes and default rates jump
- You **plant a small, known bias** in a subset of years — e.g., a synthetic protected attribute (a stand-in demographic label) that has a slightly different approval rate for reasons unrelated to actual risk, in specific years only

This single design choice is what makes every later "advanced" objective possible — without it, there's no drift to detect and no bias to find.

## 2.3 SQL concepts you need, explained

**CTE (Common Table Expression)** — a named, temporary result set you can reference like a table within one query. Lets you break a complex query into readable, logical steps instead of one giant nested subquery.
```sql
WITH monthly_utilization AS (
    SELECT applicant_id, month, balance / credit_limit AS utilization
    FROM credit_card_activity
)
SELECT applicant_id, AVG(utilization) AS avg_utilization
FROM monthly_utilization
GROUP BY applicant_id;
```

**Window functions** — calculations across a set of rows *related to* the current row, without collapsing them into one row (unlike `GROUP BY`). Essential for anything "over time."
- `LAG()` / `LEAD()` — get the previous/next row's value (e.g., last month's balance)
- `AVG() OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN ...)` — rolling average
- `ROW_NUMBER()`, `RANK()` — sequencing within groups

```sql
SELECT applicant_id, month, utilization,
       AVG(utilization) OVER (
           PARTITION BY applicant_id
           ORDER BY month
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) AS rolling_3mo_utilization
FROM monthly_utilization;
```
This is exactly how you'll compute **credit utilization trend** — not just "what's utilization now" but "is it climbing."

**Cohort analysis** — grouping applicants by a shared starting point (e.g., "applied in Q1 2021") and tracking their behavior over time relative to that point. This is how you'll structure the panel dataset itself and how you'll later validate models with expanding windows (Part 3).

## 2.4 Feature engineering — what each feature is and why it matters

| Feature | What it captures | Why it's not just "raw data" |
|---|---|---|
| **DTI (Debt-to-Income) ratio** | Monthly debt payments ÷ monthly income | A ratio, not a raw number — normalizes across income levels |
| **Repayment velocity** | How quickly someone pays down balances relative to minimum required | Behavioral, not static — same balance, different velocity = different risk |
| **Delinquency recency** | Time since last missed payment | Recent problems matter more than old ones — "recency" beats a simple count |
| **Spending volatility** | Standard deviation of monthly spending | Erratic spenders may be riskier even with the same average spend |
| **Credit utilization trend** | Direction of utilization over rolling window (via window functions above) | *Rising* utilization is a distress signal even if current level looks fine |
| **Application-time behavioral signals** | Time of day/week applied, number of applications in last 30/90 days | Captures desperation/fraud-adjacent signals invisible in bureau snapshots |
| **Ratio stacking** | Combining ratios, e.g. DTI × utilization | Interactions between ratios often carry more signal than either alone — and this is *exactly* what SHAP interaction values (Part 5) will later examine |

---

# PART 3 — Machine Learning Modeling

## 3.1 The prediction task, precisely

This is **binary classification**: predict whether an applicant will default (1) or not (0) within some fixed window (e.g., 12 months). The model outputs a probability between 0 and 1 — this probability *is* your PD (Probability of Default), which becomes central later.

## 3.2 The four models, in plain terms

- **Logistic Regression** — the traditional scorecard model. Assumes a straight-line (linear, in log-odds space) relationship between features and risk. Fast, fully transparent, but can't capture non-linear patterns or interactions well.
- **Random Forest** — builds many decision trees on random subsets of data/features and averages their votes. Captures non-linearity and interactions automatically. Harder to interpret directly (hence needing SHAP).
- **XGBoost** — builds trees *sequentially*, where each new tree corrects the errors of the previous ones ("gradient boosting"). Usually the strongest performer on tabular data like this. Industry-standard in credit risk.
- **LightGBM** — same gradient boosting idea as XGBoost, but grows trees leaf-by-leaf instead of level-by-level, making it faster on large datasets. Often near-identical accuracy to XGBoost.

You benchmark all four so you can honestly say "we compared a traditional model against three modern alternatives" rather than just picking the fanciest one.

## 3.3 Why a random train/test split is wrong for this project — and what to use instead

A random split shuffles all years together, so your test set contains applicants from *before and after* your simulated shocks, mixed in with training data from the same period. This **hides drift completely** — the exact thing your project is trying to study.

**Expanding-window validation** (the method used in the concept-drift paper you read): train on all years up to year *t*, test on year *t+1*, then expand the training window and repeat. This mimics how a real bank actually operates — retraining periodically as new data comes in — and it's the only validation strategy that lets you *see* drift happening, because each year's test performance is genuinely "the future" relative to training.

## 3.4 Evaluation metrics — and the one most projects miss

- **AUC-ROC** — how well the model *ranks* risk (does it put actual defaulters above actual non-defaulters), regardless of the exact probability values. Standard in credit risk.
- **Precision / Recall / F1** — standard classification metrics; watch out for class imbalance (defaults are usually a small minority of applicants), which makes raw accuracy meaningless and inflates apparent F1 unless handled properly (see 3.5).
- **KS statistic** — the maximum separation between the cumulative distributions of predicted scores for defaulters vs. non-defaulters. A credit-industry-specific companion to AUC.

**Calibration (the one most projects skip):** AUC and F1 tell you if the model *orders* risk correctly. They tell you **nothing** about whether a predicted "20% probability of default" actually corresponds to roughly 20% of such applicants defaulting in reality. A model can have great AUC and still be badly calibrated. You check this with a **reliability diagram** (bin predictions, plot predicted probability vs. actual observed default rate per bin — should sit near the diagonal). If it doesn't, apply **Platt scaling** (fits a logistic curve to correct probabilities) or **isotonic regression** (a more flexible, non-parametric correction).

**Why this matters more than it sounds:** your Expected Loss formula in Part 4 multiplies PD directly. A model that ranks well but outputs wildly wrong probabilities will produce a wildly wrong Expected Loss, even though its AUC looked great.

## 3.5 Class imbalance

Defaults are rare. Two common fixes:
- **SMOTE** (Synthetic Minority Oversampling Technique) — generates synthetic minority-class (default) examples to balance the training set.
- **Cost-sensitive learning** — instead of rebalancing the data, you tell the model that misclassifying a default is more "costly" than misclassifying a non-default, directly in the loss function. This connects naturally to Part 4's cost-sensitive thresholds — arguably cleaner than SMOTE because it keeps the cost logic explicit and business-driven rather than baked silently into resampled data.

## 3.6 Stacked/blended ensembles (stretch goal)

Train a simple model (often logistic regression) on the *out-of-fold predictions* of your base models (RF, XGBoost, LightGBM) as its input features. This "meta-learner" often squeezes out a small accuracy gain by learning when to trust which base model. Worth doing only after the four base models are solid and well understood — it's a nice-to-have, not core to your thesis.

---

# PART 4 — The Business Layer: Expected Loss

## 4.1 Why PD alone isn't enough

Two applicants can have the identical 10% PD but represent very different financial exposure — one wants a $500 loan, another wants a $50,000 loan. A pure PD-based decision ignores this. Real credit risk teams think in terms of **money at risk**, not just probability.

## 4.2 The Expected Loss formula

```
Expected Loss (EL) = PD × LGD × EAD
```
- **PD (Probability of Default)** — your model's calibrated output (Part 3.4 — this is why calibration matters so much)
- **LGD (Loss Given Default)** — the fraction of exposure actually lost if default happens (rarely 100% — some amount is often recovered through collections). You'll simulate this, e.g., LGD ~ Beta distribution centered around a plausible recovery rate.
- **EAD (Exposure at Default)** — how much money is actually outstanding at the moment of default (for a loan, often close to the original amount minus any repayments made).

This is the formula banks actually use for provisioning under IFRS 9 — implementing it properly (not just using PD) is one of the clearest signals in your write-up that you understand production credit risk, not just a Kaggle-style prediction task.

## 4.3 Cost-sensitive threshold selection

Instead of the default 0.5 probability cutoff (which optimizes for nothing business-relevant), define:
- Cost of a **false negative** (approving someone who defaults) ≈ the loan amount lost
- Cost of a **false positive** (rejecting someone who would have repaid) ≈ the lost interest/profit margin

Then choose the threshold that minimizes total expected cost (or maximizes expected profit) across the applicant pool — not the threshold that maximizes accuracy or F1. Plot **expected profit against threshold** to find the peak. This is the exact mechanism from the profit-based loan-default papers you read.

---

# PART 5 — Explainability

## 5.1 Why black-box models need this at all

XGBoost/LightGBM/RF don't give you a clean "+10 for income, −15 for missed payment" story the way logistic regression does. Without an explanation layer, you can't satisfy ECOA-style adverse action requirements, and you can't spot when the model is relying on something inappropriate (like a proxy for a protected attribute).

## 5.2 SHAP — the core idea, without the heavy math

SHAP (SHapley Additive exPlanations) is based on a concept from cooperative game theory: imagine each feature as a "player" contributing to the final prediction. SHAP fairly distributes the difference between the model's prediction and its average prediction among all the features, based on how much each one actually changed the outcome, averaged over every possible order in which features could be "added."

Two flavors you'll use:
- **Global SHAP** — average absolute SHAP value per feature across all applicants → tells you overall what drives the model.
- **Local SHAP** — SHAP values for one specific applicant → tells you exactly why *this person* got *this* decision (this is your adverse-action-notice content).

## 5.3 LIME (brief — you likely won't need this heavily)

LIME explains one prediction by fitting a simple, local, interpretable model (usually linear) around that single data point. Faster than SHAP but less theoretically grounded and less stable. Mention it in your literature review as an alternative, but SHAP is the stronger choice for your dashboard.

## 5.4 SHAP interaction values

Regular SHAP gives each feature's *individual* contribution. Interaction values show how *pairs* of features jointly move the prediction beyond what you'd expect from each alone — e.g., does high DTI matter much more when utilization is *also* high, versus when it's low? This is where your **ratio-stacked features** from Part 2.4 pay off — interaction values are only interesting when features actually interact.

## 5.5 Counterfactual explanations

Instead of "why was I rejected," a counterfactual answers "what would need to change for me to be approved" — e.g., "if your DTI were 3% lower, this would have been approved." This is often *more* useful to a rejected applicant than a SHAP breakdown, and it maps directly onto the legal requirement to state specific, actionable reasons for an adverse decision.

---

# PART 6 — Fairness

## 6.1 Where bias actually comes from (it's rarely intentional)

- **Disparate treatment** — using a protected attribute directly (illegal in most jurisdictions, and you shouldn't do this anyway)
- **Proxy bias** — a "neutral" feature (zip code, employer name) is statistically correlated with a protected attribute and smuggles bias in indirectly
- **Selection/historical bias** — the training data itself reflects unequal past treatment, so the model learns to repeat it

## 6.2 Fairness metrics you'll compute

- **Demographic Parity Difference (DPD)** — does the approval *rate* differ across groups?
- **Equal Opportunity Difference (EOD)** — among people who *would* repay, is the approval rate equal across groups? (True positive rate parity.)
- **Equalized Odds Difference** — extends EOD to also require equal false-positive rates across groups.
- **Disparate Impact ratio** — the classic legal/regulatory metric: (approval rate for group A) ÷ (approval rate for group B); values below ~0.8 are a common (though not legally binding) red flag threshold in US employment law, often borrowed into credit contexts too.

## 6.3 The fairness-accuracy trade-off (a real, provable limit — not just a practical annoyance)

A well-known result (Pleiss et al., 2017 — the paper cited earlier) shows that except in special cases, you generally **cannot** simultaneously have equal calibration *and* equal error rates across groups — improving one typically costs you some of the other. Your write-up should acknowledge this explicitly rather than presenting fairness fixes as a free lunch.

## 6.4 Mitigation strategies (categorized by *when* they intervene)

- **Pre-processing** — fix the data before training (reweighting samples, removing/transforming biased features)
- **In-processing** — build fairness constraints directly into the model's training objective
- **Post-processing** — adjust the *model's outputs* after training (e.g., different thresholds per group to equalize a chosen metric) — this is the category your planned Method B-style recalibration falls into, and it's usually the easiest to implement and explain.

---

# PART 7 — Concept Drift and Monitoring

## 7.1 What "drift" actually means, precisely

- **Covariate drift** — the distribution of *input features* changes (e.g., average income shifts) — detected with PSI/KS/JS
- **Prior drift** — the proportion of the *target class* changes (e.g., overall default rate rises in a recession)
- **Concept drift** (the deep version) — the actual *relationship* between features and outcome changes (e.g., income mattered less for risk during the pandemic than it did before)

## 7.2 The tools for detecting each

- **PSI (Population Stability Index)** — bins a numeric feature and compares the proportion of applicants in each bin now vs. at baseline:
  ```
  PSI = Σ (Observed% − Expected%) × ln(Observed% / Expected%)
  ```
  Rule of thumb: PSI < 0.1 = stable, 0.1–0.25 = moderate shift, > 0.25 = significant drift (this threshold is exactly what the concept-drift paper you read used).
- **KS test** — statistically compares two distributions' shapes for numeric features.
- **Jensen-Shannon divergence** — an information-theoretic distance between two probability distributions; used for categorical features.
- **Chi-square test** — tests whether category frequencies (categorical features) have changed significantly.

## 7.3 Why static SHAP breaks under drift (the actual mechanism, not just "it gets old")

SHAP compares each prediction against a **fixed background dataset** — usually a sample of the original training data — to compute what a feature's "average" contribution looks like. If the population has moved on (income distribution shifted, new borrower types appeared), that background no longer represents reality. The SHAP values you compute are technically still correct math, but they're answering "how does this compare to a population that no longer exists" — which quietly produces misleading explanations even while the underlying model's accuracy looks fine.

## 7.4 The three adaptive SHAP methods (from the concept-drift paper — you'll replicate these)

- **Method A — Drift-Weighted SHAP Adjustment:** reweight each feature's SHAP contribution by how much that feature's distribution has drifted (using PSI/JS). A feature that's drifted a lot gets its explanatory weight adjusted so it doesn't dominate or distort the picture.
- **Method B — Sliding Background Sampling:** replace the fixed background dataset with a rolling window of recent data (e.g., "last 6 months") — the SHAP background stays current automatically. In the original paper, this was the strongest and most consistent performer.
- **Method C — Surrogate Ridge Recalibration:** keep a small Ridge regression model that continuously tracks how the big model's SHAP attributions behave over time, and flags/corrects when they deviate from established patterns — cheaper than fully recomputing SHAP on the large model constantly, at the cost of ongoing retraining overhead.

---

# PART 8 — Your Project's Own Contribution

## 8.1 The honest gap in the literature

The SHAP papers explain models at one point in time. The fairness papers audit bias at one point in time. The profit/EL papers optimize business impact but never touch explainability or fairness. The concept-drift paper is the closest match — it connects drift, fairness, and explainability — but it has **no business/profit layer at all**, and its recalibration was done manually offline, not as a live, interactive tool. It explicitly lists production/live deployment as future work.

## 8.2 Your contribution, stated precisely

> Existing work treats drift-aware explainability, longitudinal fairness, and profit-based threshold optimization as separate problems, solved with separate tools. This project integrates them into a single interactive system where a threshold change is immediately reflected in the drift-adjusted SHAP explanation, the current fairness metrics, *and* the Expected Loss/revenue projection — together, not in three disconnected reports.

## 8.3 The specific technical extension: Weighted Temporal SHAP

Pick **one** clear differentiator from Method A and commit to it (don't blend both, or reviewers will ask which one actually mattered):

- **Option 1 — Cost-aware weighting:** Method A weights SHAP contributions by *statistical* drift magnitude alone. Your version weights by drift magnitude **combined with** how much that drift actually moves Expected Loss — a feature that drifted statistically but barely affects EL gets less explanatory weight than one whose drift is small statistically but large financially. This directly justifies why your SHAP layer and your business layer need to be built together.
- **Option 2 — Adaptive window length:** Method B always uses a fixed sliding window (e.g., always "last 6 months"). Your version shortens the window automatically when drift is detected to be fast-moving, and lengthens it during stable periods — using your PSI/KS drift signal itself to control the window.

Whichever you choose, your evaluation should directly compare: static SHAP vs. Method B (replicated baseline) vs. your extension — on both **explanation stability** (cosine similarity, Kendall tau — same metrics the original paper used) **and** **fairness reduction** (does your version reduce demographic parity difference more than the plain baseline?). That second comparison is what makes your extension a genuine addition rather than a cosmetic tweak.

## 8.4 The Policy Simulator — where it all comes together

A dashboard component where moving a single threshold slider live-updates:
1. Approval rate / default rate (from the model)
2. Expected Loss and projected revenue (Part 4)
3. Current fairness metrics at that threshold (Part 6)
4. The drift-adjusted SHAP explanation for a sample applicant near the threshold (Parts 7–8.3)

No paper you read built this as one interactive artifact — this is your actual deliverable-level contribution, even more than the SHAP extension itself.

---

# PART 9 — Build Roadmap

**Recommended sequence (roughly a third of total time per group):**

1. **Foundation (SQL + synthetic data):** design and build the multi-year panel dataset with engineered shocks and planted bias; write the SQL feature-extraction pipeline (window functions, ratio stacking, cohort structure)
2. **Modeling core:** expanding-window validation harness → train all 4 models → calibration check → cost-sensitive threshold + Expected Loss (PD×LGD×EAD)
3. **Explainability + fairness baseline:** static SHAP (global/local/interaction) → fairness metrics computed per year → confirm your planted bias and shocks are actually detectable (sanity check your own data design)
4. **Drift replication:** implement PSI/KS/JS drift detection → replicate Methods A/B/C → confirm Method B outperforms, same as the paper (or report honestly if it doesn't — that's still a valid finding)
5. **Your extension:** implement Weighted Temporal SHAP (pick Option 1 or 2 from 8.3) → compare against baseline on stability + fairness
6. **Dashboard/simulator:** portfolio overview → SHAP panel → drift page → integrated policy simulator with live slider
7. **Stretch, only if time remains:** counterfactual explanations, stacked ensemble

**Suggested tech stack:** Python (pandas, scikit-learn, xgboost, lightgbm, shap library), SQL (SQLite or PostgreSQL for the pipeline), Streamlit or Dash for the interactive dashboard.

---

# PART 10 — Glossary (quick reference)

| Term | One-line meaning |
|---|---|
| PD | Probability of Default — model's core output |
| LGD | Loss Given Default — % of exposure lost when default happens |
| EAD | Exposure at Default — amount outstanding when default happens |
| EL | Expected Loss = PD × LGD × EAD |
| AUC | How well the model ranks risk, ignoring exact probabilities |
| Calibration | Whether predicted probabilities match real-world frequencies |
| SHAP | Game-theory-based method for attributing predictions to features |
| PSI | Population Stability Index — measures distribution shift |
| Covariate drift | Input feature distributions change |
| Concept drift | The relationship between features and outcome changes |
| DPD | Demographic Parity Difference — fairness metric on approval rates |
| EOD | Equal Opportunity Difference — fairness metric on true positive rates |
| Proxy variable | A "neutral" feature that secretly correlates with a protected attribute |

---

# PART 11 — Your Core Reading List (recap)

1. Kumar (2025) — *SHAP for Credit Risk in Retail Lending* — explainability template
2. Bias and Fairness in AI-Based Credit Scoring (2025) — fairness metrics on synthetic data
3. Credit risk prediction based on loan profit — Chinese SMEs (2023) — cost-sensitive threshold mechanics
4. Explaining Adverse Actions Using Shapley Decomposition (2022) — counterfactuals + regulatory link
5. **Fair and Explainable Credit-Scoring under Concept Drift (2025)** — your closest match and main methodological anchor; read this one twice.

---

*End of reading material. Once you've been through this, the next concrete step is designing the exact schema for the synthetic multi-year dataset — that's the dependency everything else in Part 9 sits on top of.*
