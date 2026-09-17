"""SHAP explainability -- global importance and per-customer local attribution.

Global importance says what matters across the base. Local attribution says
what drove one prediction. The dashboard keeps them clearly separate, because
conflating the two is the most common way churn dashboards mislead.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

try:
    import shap
    HAS_SHAP = True
except Exception:                                          # pragma: no cover
    HAS_SHAP = False


PRETTY = {
    "tenure": "Tenure",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen",
    "PaperlessBilling": "Paperless billing",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "TechSupport": "Tech support",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "MultipleLines": "Multiple lines",
    "PhoneService": "Phone service",
    "Contract": "Contract",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "gender": "Gender",
}


def _feature_names(pipe) -> list[str]:
    prep = pipe.named_steps["prep"]
    names = list(C.NUMERIC_FEATURES)
    ohe = prep.named_transformers_["cat"]
    for col, cats in zip(C.CATEGORICAL_FEATURES, ohe.categories_):
        names += [f"{col}={c}" for c in cats]
    return names


def _root(name: str) -> str:
    base = name.split("=")[0]
    return PRETTY.get(base, base)


def _label(name: str) -> str:
    if "=" not in name:
        return PRETTY.get(name, name)
    col, val = name.split("=", 1)
    return f"{PRETTY.get(col, col)} - {val}"


def compute(pipe, X: pd.DataFrame, max_local: int = 10):
    """Returns (global_importance, local_matrix, feature_names, base_value)."""
    names = _feature_names(pipe)
    Xt = pipe.named_steps["prep"].transform(X)
    model = pipe.named_steps["clf"]

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(Xt)
            if isinstance(vals, list):
                vals = vals[-1]
            vals = np.asarray(vals)
            base = float(np.atleast_1d(explainer.expected_value)[-1])
        except Exception:
            vals, base = _linear_fallback(model, Xt), 0.0
    else:
        vals, base = _linear_fallback(model, Xt), 0.0

    # Global: mean |SHAP|, aggregated from one-hot columns back to the
    # original feature so the chart reads in business terms.
    mean_abs = np.abs(vals).mean(axis=0)
    agg: dict[str, float] = {}
    for n, v in zip(names, mean_abs):
        agg[_root(n)] = agg.get(_root(n), 0.0) + float(v)
    global_imp = [{"feature": k, "value": v}
                  for k, v in sorted(agg.items(), key=lambda kv: -kv[1])]

    return global_imp, vals, names, base


def _linear_fallback(model, Xt):
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return Xt * np.asarray(coef).ravel()
    imp = getattr(model, "feature_importances_", np.ones(Xt.shape[1]))
    return Xt * np.asarray(imp).ravel()


def local_top(vals_row: np.ndarray, names: list[str], row: pd.Series, k: int = 8):
    """Top-k signed contributions for one customer, with readable labels."""
    order = np.argsort(-np.abs(vals_row))[:k]
    out = []
    for i in order:
        name = names[i]
        if "=" in name:
            col, val = name.split("=", 1)
            if str(row.get(col)) != val:
                continue            # this one-hot column is 0 for this customer
            label = f"{PRETTY.get(col, col)} - {val}"
        else:
            v = row.get(name)
            shown = f"{v:.0f}" if isinstance(v, (int, float, np.integer, np.floating)) else v
            label = f"{PRETTY.get(name, name)} - {shown}"
        out.append({"feature": label, "value": float(vals_row[i])})
    return out[:k]
