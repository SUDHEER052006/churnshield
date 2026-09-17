"""ChurnShield API.

Precomputed artifacts are served straight from disk so the UI is instant.
Anything that depends on a user-supplied assumption -- budget allocation,
what-if simulation -- is recomputed live against the trained model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C            # noqa: E402
from src import economics              # noqa: E402
from app.copilot import answer         # noqa: E402

app = FastAPI(title="ChurnShield API", version="0.1.0",
              description="Predict -> explain -> estimate uplift -> allocate budget")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE: dict = {}


def _load_json(name):
    p = C.ARTIFACT_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.on_event("startup")
def load_artifacts():
    pq, csv = C.ARTIFACT_DIR / "scored.parquet", C.ARTIFACT_DIR / "scored.csv"
    if pq.exists():
        STATE["scored"] = pd.read_parquet(pq)
    elif csv.exists():
        STATE["scored"] = pd.read_csv(csv)
    else:
        STATE["scored"] = None

    for name in ("summary", "models", "rai", "curves", "shap_local"):
        STATE[name] = _load_json(name)

    mp = C.MODEL_DIR / "churn_model.pkl"
    STATE["model"] = joblib.load(mp) if mp.exists() else None
    STATE["ready"] = STATE["scored"] is not None and STATE["summary"] is not None


def require_ready():
    if not STATE.get("ready"):
        raise HTTPException(
            503, "Artifacts not built yet. Run:  python -m src.build_artifacts")


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ready": bool(STATE.get("ready")),
            "rows": int(len(STATE["scored"])) if STATE.get("scored") is not None else 0,
            "model_loaded": STATE.get("model") is not None}


@app.get("/api/summary")
def summary():
    require_ready()
    return STATE["summary"]


@app.get("/api/models")
def models():
    require_ready()
    return STATE["models"]


@app.get("/api/rai")
def responsible_ai():
    require_ready()
    return STATE["rai"]


# ---------------------------------------------------------------------------
@app.get("/api/customers")
def customers(quadrant: str | None = None, offer: str | None = None,
              q: str | None = None, limit: int = Query(60, le=500),
              budget: float | None = None, margin: float | None = None,
              horizon: int | None = None):
    require_ready()
    a = economics.merge_assumptions(
        {"budget": budget, "gross_margin": margin, "horizon_months": horizon})
    df = _rescore(a)

    if quadrant and quadrant != "all":
        df = df[df["quadrant"] == quadrant]
    else:
        df = df[(df["quadrant"] != "Sleeping dog") & (df["best_net"] > 0)]
    if offer and offer != "all":
        df = df[df["best_offer"] == offer]
    if q:
        df = df[df[C.ID_COL].str.contains(q, case=False, na=False)]

    total = int(len(df))
    df = df.sort_values("best_net", ascending=False).head(limit)
    return {"total": total, "assumptions": a,
            "rows": [_row(r) for _, r in df.iterrows()]}


@app.get("/api/customers/{customer_id}")
def customer(customer_id: str, budget: float | None = None,
             margin: float | None = None, horizon: int | None = None):
    require_ready()
    a = economics.merge_assumptions(
        {"budget": budget, "gross_margin": margin, "horizon_months": horizon})
    df = _rescore(a)
    hit = df[df[C.ID_COL] == customer_id]
    if hit.empty:
        raise HTTPException(404, f"No customer {customer_id}")
    r = hit.iloc[0]

    treatments = []
    for key, spec in C.TREATMENTS.items():
        lo, hi = float(r[f"cate_{key}_lo"]), float(r[f"cate_{key}_hi"])
        treatments.append({
            "key": key, "label": spec["label"], "detail": spec["detail"],
            "effect": float(r[f"cate_{key}"]), "lo": lo, "hi": hi,
            "significant": bool(lo > 0),
            "caveat": spec.get("caveat"),
            "cost": float(r[f"cost_{key}"]), "net": float(r[f"net_{key}"]),
        })
    treatments.sort(key=lambda t: -t["net"])

    return {
        **_row(r),
        "profile": {k: _clean(r[k]) for k in C.PROFILE_FIELDS if k in r},
        "editable": _editable_fields(),
        "clv": float(r["clv"]),
        "shap": (STATE["shap_local"] or {}).get(customer_id, []),
        "treatments": treatments,
        "assumptions": a,
    }


@app.get("/api/allocate")
def allocate(budget: float | None = None, margin: float | None = None,
             horizon: int | None = None):
    require_ready()
    a = economics.merge_assumptions(
        {"budget": budget, "gross_margin": margin, "horizon_months": horizon})
    df = _rescore(a)
    result = economics.allocate(df, a)
    result["assumptions"] = a
    result["curves"] = economics.targeting_curves(df)["points"]
    return result


class SimulateRequest(BaseModel):
    """Only columns the model was actually trained on may be changed.

    `changes` is a free-form map validated against the model's own feature
    list, so the simulator works for any dataset profile without a bespoke
    schema per dataset. Anything not in the feature list is rejected rather
    than silently ignored.
    """
    customer_id: str
    changes: dict[str, object] = Field(default_factory=dict)


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Re-score the customer through the real trained model.

    Only columns the model was trained on can be changed -- there is no
    invented 'discount' field, because the model never saw one.
    """
    require_ready()
    if STATE["model"] is None:
        raise HTTPException(503, "Model not loaded")

    df = STATE["scored"]
    hit = df[df[C.ID_COL] == req.customer_id]
    if hit.empty:
        raise HTTPException(404, f"No customer {req.customer_id}")

    row = hit.iloc[0].to_dict()
    frame = _full_feature_row(row)
    before = float(STATE["model"].predict_proba(frame)[:, 1][0])

    allowed = set(C.NUMERIC_FEATURES) | set(C.CATEGORICAL_FEATURES)
    unknown = [k for k in req.changes if k not in allowed]
    if unknown:
        raise HTTPException(
            400, f"Not model features, so changing them would be meaningless: "
                 f"{', '.join(unknown)}")

    changed = {}
    for field, val in req.changes.items():
        if val is None:
            continue
        if field in C.NUMERIC_FEATURES:
            val = float(val)
        if str(val) != str(frame.at[0, field]):
            frame.at[0, field] = val
            changed[field] = val
    after = float(STATE["model"].predict_proba(frame)[:, 1][0])

    return {"customer_id": req.customer_id, "before": before, "after": after,
            "delta": after - before, "changed": changed,
            "note": "Model-predicted change under the modified features. "
                    "Not a causal guarantee of customer behaviour."}


class AskRequest(BaseModel):
    question: str


@app.post("/api/copilot")
def copilot(req: AskRequest):
    require_ready()
    return answer(req.question, STATE)


# ---------------------------------------------------------------------------
def _rescore(a: dict) -> pd.DataFrame:
    """Recompute costs / net values under the caller's assumptions."""
    base = STATE["scored"]
    if (a["gross_margin"] == C.DEFAULT_ASSUMPTIONS["gross_margin"]
            and a["horizon_months"] == C.DEFAULT_ASSUMPTIONS["horizon_months"]):
        return base
    return economics.score_table(base, a)


def _editable_fields() -> list[dict]:
    """Which what-if controls the UI should render.

    Derived from the active profile rather than hardcoded, and restricted to
    real model features plus the revenue column. Nothing the model never saw
    can appear as a slider.
    """
    base = STATE["scored"]
    out = []
    for key, spec in C.TREATMENTS.items():
        col = spec["column"]
        if col in C.CATEGORICAL_FEATURES and col in base.columns:
            out.append({"field": col, "label": spec["label"], "type": "choice",
                        "options": sorted(base[col].dropna().astype(str).unique().tolist())})
    rev = C.REVENUE_COL
    if rev in C.NUMERIC_FEATURES and rev in base.columns:
        out.append({"field": rev, "label": "Monthly revenue", "type": "number",
                    "min": float(base[rev].quantile(0.01)),
                    "max": float(base[rev].quantile(0.99))})
    return out


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


def _row(r) -> dict:
    """The shape every table and card in the UI consumes.

    `tenure` and `monthly` are deliberately generic names mapped from the
    active profile's columns, so the frontend never learns a dataset's
    vocabulary.
    """
    return {
        "id": r[C.ID_COL],
        "churn_prob": float(r["churn_prob"]),
        "quadrant": r["quadrant"],
        "best_uplift": float(r["best_uplift"]),
        "offer": r["best_offer"],
        "offer_label": C.TREATMENTS[r["best_offer"]]["label"],
        "offer_short": C.TREATMENTS[r["best_offer"]]["short"],
        "offer_effect": float(r["best_effect_for_offer"]),
        "cost": float(r["best_cost"]),
        "net": float(r["best_net"]),
        "tenure": float(r[C.TENURE_COL]),
        "monthly": float(r[C.REVENUE_COL]),
    }


def _full_feature_row(row: dict) -> pd.DataFrame:
    """Rebuild a complete model input row, filling any column the scored
    artifact trimmed away with a sane per-column default."""
    base = STATE["scored"]
    data = {}
    for col in C.NUMERIC_FEATURES:
        if col in row and pd.notna(row.get(col)):
            data[col] = row[col]
        else:
            data[col] = float(base[col].median()) if col in base.columns else 0.0
    for col in C.CATEGORICAL_FEATURES:
        if col in row and row.get(col) is not None:
            data[col] = row[col]
        elif col in base.columns:
            data[col] = base[col].mode().iloc[0]
        else:
            data[col] = "Unknown"
    return pd.DataFrame([data])[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES]
