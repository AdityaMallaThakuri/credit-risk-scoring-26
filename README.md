# Credit Risk 

Explainable, fair, and drift-aware credit risk scoring system.

## Architecture

Three layers, kept deliberately separate:

- **Data layer** — a SQLite database (`data/processed/credit_risk.db`)
  built by a SQL feature-extraction pipeline (`sql/`, run via
  `src/features/run_sql_pipeline.py`) plus a set of precomputed CSV
  artifacts from later analysis phases (calibration, fairness, drift,
  Weighted Temporal SHAP — all under `data/processed/`).
- **Backend** (`src/app/`, FastAPI) — the only process that touches
  modeling code. At startup it loads one persisted pipeline
  (preprocessor + LightGBM + isotonic calibrator, built by
  `src/models/train_final_model.py`) and a SHAP explainer exactly once,
  not per request, then serves ~15 endpoints across five routers:
  applicant scoring/explanation, portfolio segments, fairness metrics
  (including a live-threshold recompute), drift metrics, and Expected
  Loss/profit simulation.
- **Dashboard** (`src/app/dashboard/`, Streamlit) — six screens
  (Portfolio Overview, Segments, Applicant SHAP, Fairness Over Time,
  Drift Monitoring, and the Policy Simulator — an integrated threshold
  slider that live-updates approval rate, Expected Loss, fairness
  metrics, and a sample applicant's explanation together). This process
  holds no state of its own and never imports modeling code — every
  number it shows comes from an HTTP call to the backend, so the two
  processes can be restarted or redeployed independently.

Setup and run instructions for all three layers are below (step 3 for
the data layer, steps 4–5 for the backend/dashboard).

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

### 4. Build the deployable model artifact (required before running the app)

The application (below) loads one persisted pipeline rather than
refitting a model per request. Build it once, after step 3 above:

```
python src/models/train_final_model.py
```

This fits a fresh preprocessor + LightGBM + isotonic calibrator (an
80/20 stratified train/calibration split, same split-before-resample
discipline as everywhere else in this project) and writes
`data/processed/final_model_pipeline.joblib`. Re-run it any time
`modeling_feature_set` changes; it's the only artifact the app depends
on that isn't already produced by step 3.

### 5. Run the application

Two processes, in two terminals, both from a repo with step 4 already
done:

```
# Terminal 1 — backend API
cd src/app
uvicorn main:app --host 127.0.0.1 --port 8000
```

```
# Terminal 2 — dashboard
cd src/app/dashboard
streamlit run Home.py --server.port 8501
```

- Backend: interactive API docs at `http://127.0.0.1:8000/docs`
- Dashboard: `http://localhost:8501` — Streamlit auto-discovers the six
  screens under `pages/` (Segments, Applicant SHAP, Fairness Over Time,
  Drift Monitoring, Policy Simulator) as sidebar entries

The dashboard talks to the backend over plain HTTP (`API_BASE_URL` env
var, default `http://127.0.0.1:8000`) — it holds no state of its own and
never imports modeling code directly, so the backend must already be
running before the dashboard can show real data.

### 6. Run the test suite

```
python -m pytest tests/ -v
```

63 tests covering calibration on held-out data, cost-sensitive threshold
sanity at the extremes, SHAP additive consistency after Weighted
Temporal SHAP's renormalization, and a functional smoke test for every
backend endpoint (via FastAPI's in-process test client — no running
server required). Takes about 20 seconds; the first test that touches
the API pays a one-time ~15s cost to build the full application state
(persisted model, feature table, SHAP explainer), shared across the rest
of that test session.

### 7. Explore via the notebooks

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
