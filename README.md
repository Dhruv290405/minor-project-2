# Distributed Diabetes Risk Assessment System

A full-stack, **distributed** diabetes-risk screening system whose machine-learning core uses a
**custom, cost-sensitive XGBoost loss function** that deliberately penalizes *false negatives*
(missed diabetics) far more than false positives — because in clinical screening, telling a sick
patient they are healthy is the dangerous, expensive error.

The system pairs this safety-tuned model with **explainability** (per-patient SHAP risk/protective
factors) and serves it through four independent tiers: a React UI → a Node.js API gateway → a
Python/FastAPI ML microservice → MongoDB.

> ⚠️ **Disclaimer:** Research/educational project on the public Pima dataset. **Not a medical
> device** and not for clinical use. The served `risk_percentage` is a cost-sensitive *risk score*,
> not a calibrated probability.

---

## Table of Contents

- [Key Idea](#key-idea)
- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup & Running](#setup--running)
- [The Machine Learning Core](#the-machine-learning-core)
- [API Reference](#api-reference)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Generating the Report](#generating-the-report)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Idea

Standard XGBoost minimizes binary log-loss, whose gradient (`p − y`) and hessian (`p(1−p)`) treat a
false negative and a false positive **symmetrically**. On an imbalanced screening dataset this
maximizes accuracy while quietly accepting many missed diabetics.

XGBoost is **gradient-agnostic** — it only needs the per-sample gradient and hessian of *any* loss.
This project supplies three custom objectives that make missed positives much costlier:

| Objective | Mechanism | Intuition |
|---|---|---|
| **Weighted cross-entropy** | `weight = w if y==1 else 1` | Positives cost `w`× more (we use `w=3`). |
| **Focal loss** (Lin et al. 2017) | `(1−pₜ)^γ` modulating factor | Focuses training on *hard* positives (the FNs). |
| **Exponential FN penalty** | `L = exp(γ·(1−p))·(−log p)` for positives | Penalty grows exponentially as a diabetic is confidently called healthy. |

The **Focal** objective is deployed as the production model. Full math (gradients/hessians) lives in
[`custom_objectives.py`](custom_objectives.py) and [`PAPER.md`](PAPER.md).

---

## Results

5-fold cross-validation × 5 seeds (25 runs/model), identical hyperparameters — **the loss is the
only variable**:

| Model | Recall ↑ | F2 ↑ | Precision | Specificity | ROC-AUC | FN/fold ↓ |
|---|---|---|---|---|---|---|
| Standard | 0.610 | 0.620 | 0.671 | 0.838 | 0.826 | 20.9 |
| Weighted CE | 0.756 | 0.720 | 0.612 | 0.742 | 0.827 | 13.1 |
| **Focal (deployed)** | **0.770** | **0.728** | 0.602 | 0.725 | 0.825 | **12.3** |
| Exponential | 0.703 | 0.686 | 0.628 | 0.775 | 0.821 | 15.9 |

**Recall rises 0.61 → 0.77 and false negatives nearly halve, while ROC-AUC is unchanged** — the
model isn't worse, it's re-tuned to a safer operating point. Figures in [`figures/`](figures/).

---

## Architecture

The browser never talks to the ML service directly; all traffic flows through the Node gateway,
which orchestrates the ML call and persistence. Each tier is an independently deployable process.

```
Browser (React SPA, :5173)
        │  POST /api/assess  { patientName, ...vitals }
        ▼
Node.js / Express API Gateway (:5000)
        │  POST /predict { ...vitals }              ┌─────────────────────┐
        ▼                                           │  MongoDB (:27018)   │
FastAPI ML Engine (:8000)  ── XGBoost + SHAP ──►    │  save assessment    │
        │  { risk_percentage, explanation }         └─────────────────────┘
        ▲────────────────────────────────────────────────────┘
```

**Request flow:**
1. Clinician submits the form; SPA POSTs name + 8 vitals to `/api/assess`.
2. Gateway validates the name, strips it, relays the 8 vitals to the ML `/predict`.
3. ML service applies zero-as-missing → predicts margin → applies sigmoid → runs SHAP.
4. Gateway saves `{ patient, vitals, result }` to MongoDB, returns the result to the SPA.
5. SPA renders the colour-coded risk + risk/protective factors. History tab reads back paginated records.

---

## Project Structure

```
minor/
├── main.py                        # FastAPI ML service: /predict (scoring + SHAP)
├── custom_objectives.py           # The 3 custom cost-sensitive XGBoost losses (grad/hess)
├── train_model.py                 # Standard log-loss baseline trainer (control)
├── train_cost_sensitive_model.py  # Trains + deploys the winning (Focal) model
├── compare_models.py              # Experiment harness: CV×seeds, metrics, figures, PAPER.md
├── xgboost_diabetes_model.pkl     # Deployed model (cost-sensitive Focal)
├── PAPER.md                       # Research write-up (methods, results, discussion)
├── pima.csv                       # Cached dataset
├── figures/                       # Generated result plots
├── results/                       # metrics.csv, summary.csv, winner.txt
│
├── backend/                       # Node.js / Express API gateway
│   ├── server.js                  #   /api/assess (orchestrator), /api/history
│   ├── models/Prediction.js       #   Mongoose schema
│   ├── docker-compose.yml         #   Local MongoDB
│   └── .env                       #   MONGO_URI
│
├── frontend/                      # React + Vite + TypeScript + Tailwind SPA
│   ├── src/App.tsx                #   Assess form, result panel, paginated history
│   └── vite.config.ts             #   /api proxy → :5000
│
└── report/                        # Minor-project report (AITR/RGPV template)
    ├── build_report.py            #   Generator (→ .docx + REPORT.md)
    ├── Distributed_..._Report.docx
    └── REPORT.md
```

---

## Tech Stack

| Tier | Technology | Why |
|---|---|---|
| ML service | Python 3.14, FastAPI, Uvicorn | Typed validation (Pydantic), async, minimal code |
| Model | XGBoost 3.2, SHAP 0.51, scikit-learn 1.8 | Custom objective support + per-prediction explanations |
| Gateway | Node.js, Express, Axios, Mongoose | Lightweight orchestration / persistence boundary |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS | Fast, type-safe SPA |
| Database | MongoDB (Docker) | Schema-flexible JSON assessment records |

---

## Prerequisites

- **Python 3.11+** with the virtualenv at `./venv` (already present in this repo)
- **Node.js 18+** and npm
- **Docker** + Docker Compose (for MongoDB)

Python dependencies (already installed in `./venv`): `fastapi uvicorn xgboost shap scikit-learn
pandas numpy matplotlib joblib python-docx`.

---

## Setup & Running

Start the four services **in this order**. Use four terminals (or background each).

### 1. MongoDB
```bash
docker compose -f backend/docker-compose.yml up -d
```

### 2. ML Engine (FastAPI, port 8000)
```bash
./venv/bin/uvicorn main:app --port 8000
```
On startup it loads `xgboost_diabetes_model.pkl` and builds the SHAP explainer.

### 3. API Gateway (Node, port 5000)
```bash
cd backend
npm install        # first time only
node server.js
```

### 4. Frontend (Vite, port 5173)
```bash
cd frontend
npm install        # first time only
npm run dev
```

Open the printed Vite URL (default http://localhost:5173). Enter a patient name and the eight
vitals (leave a field at **0** if it was not measured), then click **Assess Risk**. The **History**
tab lists past assessments.

---

## The Machine Learning Core

### Dataset & preprocessing
- **Pima Indians Diabetes** (768 patients, 268 diabetic ≈ 34.9%), 8 numeric features.
- In `Glucose, BloodPressure, SkinThickness, Insulin, BMI`, a value of **0 is physiologically
  impossible** and means *missing* → converted to `NaN`. XGBoost handles `NaN` natively (it learns a
  default branch direction), so no imputation is done. **The identical transform is applied at
  serving time** so inference sees data exactly as training did.

### ⚠️ The serving pitfall (important)
With a custom objective, XGBoost has **no link function**, so:
- `predict_proba` is **unreliable**, and
- `predict` returns the **raw margin**, not a probability.

Every prediction site must apply the sigmoid manually:
```python
proba = sigmoid(model.predict(X, output_margin=True))
```
This is handled consistently in both `compare_models.py` and `main.py`.

### Stability protocol
- Stratified 5-fold CV × 5 seeds `{42, 0, 7, 13, 21}` = 25 runs/model, reporting **mean ± std**.
- All models share one hyperparameter set (`n_estimators=200, max_depth=4, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, base_score=0.5`) so any difference is
  causally attributable to the loss.
- Hessians floored at `1e-6`; the exponential gradient is bounded → stable training.

---

## API Reference

### `POST /api/assess` (gateway)
**Request:**
```json
{
  "patientName": "Jane Doe",
  "Pregnancies": 4, "Glucose": 145, "BloodPressure": 80, "SkinThickness": 30,
  "Insulin": 0, "BMI": 33.6, "Pedigree": 0.55, "Age": 50
}
```
**Response:**
```json
{
  "success": true,
  "message": "Assessment complete",
  "data": {
    "status": "success",
    "risk_percentage": 70.04,
    "explanation": {
      "primary_risk_factors": ["Glucose of 145.0 significantly increases risk.", "..."],
      "protective_factors": []
    }
  }
}
```

### `GET /api/history?page=1` (gateway)
Returns paginated past assessments (10/page): `{ success, data: [...], pagination: { page, limit, total, totalPages } }`.

### `POST /predict` (ML service, internal)
Body = the 8 vitals only (no `patientName`). Returns `{ status, risk_percentage, explanation }`.

| Status | Meaning |
|---|---|
| `400` | Missing/blank patient name (gateway) |
| `503` | ML service unreachable (`ECONNREFUSED`) |
| `500` | Internal error |

---

## Reproducing the Experiment

```bash
# Run the full comparison: CV×seeds, metrics table, figures, and inject results into PAPER.md
./venv/bin/python compare_models.py

# Train and deploy the winning cost-sensitive model (overwrites xgboost_diabetes_model.pkl)
./venv/bin/python train_cost_sensitive_model.py

# (Optional) retrain the standard baseline
./venv/bin/python train_model.py
```
Outputs land in `results/` (CSV + `winner.txt`), `figures/` (5 PNGs), and the Results section of
`PAPER.md`. The harness **asserts the thesis** (a cost-sensitive model must beat the baseline on
recall *and* false negatives) and fails loudly otherwise.

---

## Generating the Report

The AITR/RGPV minor-project report is generated from a single source of truth:
```bash
./venv/bin/python report/build_report.py
```
Produces `report/Distributed_Diabetes_Risk_Assessment_System_Report.docx` (submittable) and
`report/REPORT.md`. Personal fields are left as `<<placeholders>>` — fill in names/roll numbers in
the `<<…>>` constants near the bottom of `build_report.py`, then re-run.

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| MongoDB URI | `backend/.env` (`MONGO_URI`) | `mongodb://admin:supersecret@localhost:27018/minor?authSource=admin` |
| Gateway port | `backend/server.js` (`PORT`) | `5000` |
| ML service port | `uvicorn --port` | `8000` (gateway expects `127.0.0.1:8000`) |
| API proxy | `frontend/vite.config.ts` | `/api → http://localhost:5000` |
| Cost weights | `compare_models.py` | `w=3.0`, focal `γ=2, α=0.75`, exp `γ=2` |

---

## Troubleshooting

- **`Machine Learning service is currently unavailable` (503):** the FastAPI service (step 2) isn't
  running on port 8000. Start it before submitting an assessment.
- **MongoDB connection error:** ensure Docker is running and `docker compose ... up -d` succeeded;
  the container maps host port **27018** → container 27017.
- **History/assess fails in the browser:** confirm the gateway (`:5000`) is up and the Vite proxy in
  `vite.config.ts` points at it.
- **Model file missing:** run `./venv/bin/python train_cost_sensitive_model.py` to regenerate
  `xgboost_diabetes_model.pkl`.
- **Risk score looks "too high":** by design — cost-sensitive training skews positive scores up. It's
  a monotonic risk score for triage, not a calibrated probability (see [`PAPER.md`](PAPER.md)).

---

## License

Educational / research use. Dataset: UCI Pima Indians Diabetes. Not for clinical deployment.
