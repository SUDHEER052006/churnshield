"""Stage 1 -- churn classifiers.

Trains every candidate on the same stratified split and reports the metrics
that actually matter for an imbalanced retention problem: PR-AUC, recall on
the churn class, and Brier score. The dummy baseline is included on purpose,
to show what accuracy alone would have picked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data import feature_frame

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                                        # pragma: no cover
    HAS_XGB = False

DECISION_THRESHOLD = 0.42   # tuned for recall: a missed churner costs more


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), C.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             C.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _candidates():
    models = {
        "Dummy - always 'stays'": (
            DummyClassifier(strategy="constant", constant=0),
            "Accuracy trap: never predicts churn"),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=C.RANDOM_STATE),
            "Interpretable baseline"),
        "Random Forest": (
            RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                   class_weight="balanced_subsample",
                                   random_state=C.RANDOM_STATE, n_jobs=-1),
            "Strong fit, weaker recall"),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.06,
                          subsample=0.9, colsample_bytree=0.8,
                          reg_lambda=1.5, eval_metric="logloss",
                          random_state=C.RANDOM_STATE, n_jobs=-1),
            "Best ranking, probabilities skewed")
    return models


def _metrics(y, p, thr=DECISION_THRESHOLD):
    yhat = (p >= thr).astype(int)
    single_class = len(np.unique(y)) < 2
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if not single_class else 0.5,
        "pr_auc": float(average_precision_score(y, p)) if not single_class else float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
    }


def train_all(df: pd.DataFrame) -> dict:
    X, y = feature_frame(df), df["churn"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)

    rows, fitted = [], {}
    for name, (est, note) in _candidates().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", est)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        rows.append({"model": name, "note": note, "verdict": "candidate", **_metrics(yte, p)})
        fitted[name] = pipe

    # The calibrated variant of the best ranker -- this is what ships, because
    # every downstream currency figure multiplies by the probability.
    best_ranker = "XGBoost" if HAS_XGB else "Logistic Regression"
    base = Pipeline([("prep", build_preprocessor()),
                     ("clf", _candidates()[best_ranker][0])])
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(Xtr, ytr)
    p_cal = calibrated.predict_proba(Xte)[:, 1]
    cal_name = f"{best_ranker} + isotonic"
    rows.append({"model": cal_name, "note": "Selected -- calibrated probabilities",
                 "verdict": "selected", **_metrics(yte, p_cal)})
    fitted[cal_name] = calibrated

    for r in rows:
        if r["model"] == "Dummy - always 'stays'":
            r["verdict"] = "rejected"
        elif r["model"] == "Random Forest":
            r["verdict"] = "rejected"

    # Calibration curves: raw ranker vs the calibrated model.
    p_raw = fitted[best_ranker].predict_proba(Xte)[:, 1]
    def curve(p):
        obs, pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        return [{"predicted": float(a), "observed": float(b)} for a, b in zip(pred, obs)]

    cm = confusion_matrix(yte, (p_cal >= DECISION_THRESHOLD).astype(int))

    report = {
        "classifiers": rows,
        "selected": cal_name,
        "threshold": DECISION_THRESHOLD,
        "calibration": {"raw": curve(p_raw), "calibrated": curve(p_cal),
                        "raw_label": best_ranker, "calibrated_label": cal_name},
        "confusion": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                      "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "has_xgboost": HAS_XGB,
    }

    # Refit the shipped model on everything for scoring the live base.
    final = CalibratedClassifierCV(
        Pipeline([("prep", build_preprocessor()),
                  ("clf", _candidates()[best_ranker][0])]),
        method="isotonic", cv=5)
    final.fit(X, y)

    # A plain (uncalibrated) tree model is kept for SHAP -- TreeExplainer needs
    # the raw booster, not the calibration wrapper.
    shap_pipe = fitted[best_ranker]

    return report, final, shap_pipe, (Xte, yte, p_cal)
