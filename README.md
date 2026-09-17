# ChurnShield

**Don't predict who leaves. Predict who you can save.**

A retention console for the customer-churn problem statement. Most entries stop at
"who is likely to churn". ChurnShield goes one step further and estimates, per
customer and per offer, whether an intervention would actually *change the outcome* —
then allocates a finite budget against that estimate.

---

## The idea in one table

A churn model ranks by risk. Ranking by risk spends money on two groups it cannot help:

| Quadrant | Meaning | Correct action |
|---|---|---|
| **Persuadable** | Stays only if treated | **Spend here — this is the entire ROI** |
| Sure thing | Stays either way | Don't spend; you're discounting a loyal customer |
| Lost cause | Leaves either way | Don't spend; money burned |
| **Sleeping dog** | Leaves *because* you contacted them | **Contacting them costs you the customer** |

Separating those four requires a causal treatment effect, not a prediction. That is
what the EconML layer does, and it is the whole differentiator.

---

## Stack

| Layer | Library | Why |
|---|---|---|
| Classification | scikit-learn, XGBoost | Churn probability, calibrated |
| Explainability | SHAP | Global drivers and per-customer attribution |
| **Causal uplift** | **EconML** (Microsoft Research) | Per-customer treatment effect with confidence intervals |
| **Refutation** | **DoWhy** (Microsoft) | Attacks its own causal estimate three ways |
| **Fairness** | **Fairlearn** (Microsoft) | Group metrics and parity gates |
| API | FastAPI | Serves precomputed artifacts + live recompute |
| UI | React + Vite + Tailwind + Recharts | Fluent 2 themed console |

EconML, DoWhy and Fairlearn are free open-source Microsoft projects — no Azure
account required. Azure is only the deployment target (see *Deployment* below).

---

## The four treatments

Every offer maps to a **real column in the Telco dataset**. Nothing is invented, so
re-scoring a customer under a changed feature is meaningful rather than theatre.

| Offer | Column | Intervention |
|---|---|---|
| Contract lock-in | `Contract` | Month-to-month → One/Two year |
| Tech support add-on | `TechSupport` | No → Yes |
| Security bundle | `OnlineSecurity` | No → Yes |
| Auto-pay migration | `PaymentMethod` | Electronic check → automatic |

---

## Getting started

### 1. Dataset

Download **IBM Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 rows)
and drop it into `backend/data/`.

> Without it the pipeline generates a Telco-shaped synthetic sample so everything still
> runs. Synthetic mode is flagged in the sidebar and never presented as real data.

### 2. Backend

```powershell
cd backend
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m src.build_artifacts      # trains everything, writes artifacts/
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

`build_artifacts` is the whole pipeline: clean → train five classifiers → SHAP →
estimate CATE for four treatments → economics → fairness, refutation, error analysis.
It prints each stage and writes JSON into `backend/artifacts/`.

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to the backend, so no CORS setup.

---

## Layout

```
churnshield/
├── backend/
│   ├── src/
│   │   ├── config.py            paths, treatments, economic assumptions
│   │   ├── data.py              load + clean Telco (synthetic fallback)
│   │   ├── train.py             5 classifiers, calibration, confusion
│   │   ├── explain.py           SHAP global + local
│   │   ├── causal.py            EconML learners, CATE, Qini/AUUC
│   │   ├── economics.py         CLV, net value, knapsack, targeting curves
│   │   ├── rai.py               Fairlearn, DoWhy refutation, error analysis
│   │   └── build_artifacts.py   orchestrator
│   ├── app/
│   │   ├── main.py              FastAPI
│   │   └── copilot.py           intent router over parameterised queries
│   ├── artifacts/               generated JSON + scored table
│   └── models/                  churn_model.pkl
└── frontend/src/
    ├── components/              Shell (Fluent nav) + UI primitives
    └── pages/                   the seven screens
```

---

## The seven screens

1. **Command Center** — *Revenue at risk* beside *Savable revenue*. The gap is the thesis.
2. **Action List** — a worklist, not a risk table: who to call, with what, ranked by expected net value.
3. **Customer 360** — SHAP drivers, four treatment effects with 90% CIs, what-if on real features only.
4. **Budget Optimizer** — drag the budget; the knapsack re-solves and the Qini chart shows the lift over risk targeting.
5. **Model Lab** — every classifier including a dummy baseline, every uplift learner, calibration, confusion matrix.
6. **Responsible AI** — Fairlearn by group, DoWhy refutation results, auto-discovered error cohorts.
7. **Churn Copilot** — natural-language questions, each answer showing the intent and parameters that ran.

---

## Demo path (about 90 seconds)

1. **Command Center** — point at the gap between revenue at risk and savable revenue.
2. Click a red dot in the lower-right of the scatter.
3. **Customer 360** — a ~90% churn risk customer whose best uplift is *negative*. Every
   risk-ranked model in the room would spend money making this worse.
4. **Budget Optimizer** — drag the budget slider, land on the "+N% over risk targeting" tile.
5. **Responsible AI** — the fairness gate and three surviving refutation tests.

---

## What this does not claim

- The Telco data is **observational**. Customers chose their own contracts; nobody
  randomised them. Every causal estimate assumes no unmeasured confounding. That
  assumption is *tested* by the refutation battery, not proven.
- The what-if simulator shows a **model-predicted change**, never a guaranteed outcome.
- Budget returns assume the customer **accepts** the offer. Real uptake is lower, so
  treat the figures as an upper bound and scale by observed conversion.
- No offer is sent automatically. Every row is a recommendation for a human agent.

---

## Deployment

Built to run locally first — that is deliberate, because losing a day to cloud config
is a bigger risk than not being deployed. Target architecture:

- **Azure Container Apps** — the FastAPI backend
- **Azure Static Web Apps** — the built frontend (`npm run build` → `dist/`)
- **Azure ML** — model registry for `churn_model.pkl`
- **Azure OpenAI** — optional, to phrase Copilot answers (the intent router stays in code;
  the model never touches the data directly)

Free tiers plus an Azure for Students credit cover all of it.
