# Credit Risk 

Explainable, fair, and drift-aware credit risk scoring system.

## Setup on a new machine

This repo's `.gitignore` deliberately excludes large/regenerable data
(`data/raw/`, `data/processed/`, `data/synthetic/`) and, as a project
decision, `CLAUDE.md` and `docs/` — carry those two over by hand
(copy the files directly) if you need them on the new machine; a plain
`git clone` will not include them.

### 1. Clone and install dependencies

```
git clone <repo-url>
cd credit-risk-fyp
python -m venv .venv          # or use conda
.venv\Scripts\activate         # Windows; `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this project was built and
validated against. `shap` requires `numpy` to stay compatible with
`numba` — if you change any pin, reinstall `shap` last and re-verify
`import shap` works before trusting any explainability output.

### 2. Get the raw data

The 7 raw Home Credit tables are **not** in this repo (too large, and
`data/raw/` is treated as read-only/external to the project). Download
the "Home Credit Default Risk" dataset from Kaggle
(`kaggle.com/c/home-credit-default-risk`, requires a free Kaggle
account), then rename and place each file under `data/raw/` exactly as:

| Kaggle filename | Place as |
|---|---|
| `application_train.csv` | `data/raw/HC_application_train.csv` |
| `bureau.csv` | `data/raw/HC_bureau.csv` |
| `bureau_balance.csv` | `data/raw/HC_bureau_balance.csv` |
| `credit_card_balance.csv` | `data/raw/HC_credit_card_balance.csv` |
| `installments_payments.csv` | `data/raw/HC_installments_payments.csv` |
| `POS_CASH_balance.csv` | `data/raw/HC_POS_CASH_balance.csv` |
| `previous_application.csv` | `data/raw/HC_previous_application.csv` |

(`application_test.csv`, `sample_submission.csv`, and the column
description file from Kaggle are not used by this project.)

### 3. Rebuild the derived data, in order

Everything below writes only to `data/processed/` and `data/synthetic/`
— `data/raw/` is never touched. Run from the repo root:

```
python src/features/run_sql_pipeline.py          # ~25 min full run; add --sample 5000 for a fast dev-scale run
python src/features/engineer_features.py
python src/features/vif_check.py                 # report-only, prints to console
python src/features/build_modeling_feature_set.py
python src/features/build_synthetic_overlay.py
python src/models/cv_split.py
python src/models/train_baseline_models.py
python src/models/calibration.py
python src/models/expected_loss.py
```

Each script is idempotent (safe to re-run) and reads/writes
`data/processed/credit_risk.db` (SQLite) plus a handful of CSVs under
`data/processed/`. After this, `src/explainability/`, `src/fairness/`,
and `src/drift/` scripts can be run the same way — see each phase's
notebook under `notebooks/` for the exact call order.

### 4. Explore via the notebooks

```
jupyter lab
```

Open any notebook under `notebooks/` — each one `%run`s the
corresponding `src/` script(s) and adds plots/interpretation on top.
`notebooks/phase1_eda.ipynb` and `phase2_feature_engineering.ipynb` are
read-only against `credit_risk.db` and safe to open first to sanity-check
the setup worked before running heavier phases (modeling, SHAP).

**If you're on a memory-constrained machine (8GB RAM or less):** restart
the Jupyter kernel between phases rather than running everything in one
long-lived session — the SQL pipeline, SHAP, and SMOTE-based training
steps each hold a full copy of a 300K-row table (or several) in memory
at once, and stale DataFrames from earlier cells add up fast.
