"""Load and clean the active dataset profile.

If the real CSV is absent, a profile-shaped synthetic frame is generated so
the whole pipeline still runs end to end. Synthetic mode is flagged loudly
and surfaced in the UI -- it is never passed off as real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .synthetic import synth_cell2cell, synth_telco


def _find_csv():
    for name in C.FILENAMES:
        p = C.DATA_DIR / name
        if p.exists():
            return p
    for p in sorted(C.DATA_DIR.glob("*.csv")):
        try:
            head = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        if C.TARGET in head.columns and C.REVENUE_COL in head.columns:
            return p
    return None


def load_raw() -> tuple[pd.DataFrame, dict]:
    path = _find_csv()
    if path is not None:
        df = pd.read_csv(path)
        meta = {"source": f"{C.DATASET_DISPLAY} ({path.name})", "synthetic": False}
    else:
        df = synth_cell2cell() if C.DATASET == "cell2cell" else synth_telco()
        meta = {"source": f"{C.DATASET_DISPLAY} - synthetic stand-in", "synthetic": True}
    meta["dataset"] = C.DATASET
    meta["rows_raw"] = int(len(df))
    return df, meta


def clean(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Target -> 0/1. Handles "Yes"/"No", "True"/"False" and 1/0 alike.
    raw_target = df[C.TARGET].astype(str).str.strip().str.lower()
    df["churn"] = raw_target.isin(["yes", "true", "1", "churn"]).astype(int)

    # Numerics arrive as strings often enough to be worth forcing every time
    # (Telco's TotalCharges is blank for tenure-0 customers; Cell2Cell ships
    # genuine NaNs in the revenue and usage columns).
    for col in C.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    essential = [C.REVENUE_COL, C.TENURE_COL]
    n_before = len(df)
    df = df.dropna(subset=[c for c in essential if c in df.columns])
    meta["dropped_missing_essential"] = int(n_before - len(df))

    # Remaining numeric gaps are filled with the column median rather than
    # dropped -- Cell2Cell would lose thousands of rows otherwise.
    filled = 0
    for col in C.NUMERIC_FEATURES:
        n_na = int(df[col].isna().sum())
        if n_na:
            df[col] = df[col].fillna(df[col].median())
            filled += n_na
    meta["numeric_values_imputed"] = filled

    for col in C.CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown"})

    # Columns a treatment definition needs but that are not model features.
    for extra in ("RetentionOffersAccepted", "RetentionCalls", "AgeHH1"):
        if extra in df.columns:
            df[extra] = pd.to_numeric(df[extra], errors="coerce").fillna(0)

    for name, fn in C.DERIVED.items():
        try:
            df[name] = fn(df)
        except Exception:
            df[name] = "Unknown"

    if C.ID_COL not in df.columns:
        df[C.ID_COL] = [f"{i:06d}" for i in range(len(df))]
    df[C.ID_COL] = df[C.ID_COL].astype(str)

    n_before = len(df)
    df = df.drop_duplicates(subset=[C.ID_COL])
    meta["dropped_duplicates"] = int(n_before - len(df))

    df = df.reset_index(drop=True)
    meta["rows_clean"] = int(len(df))
    meta["churn_base_rate"] = float(df["churn"].mean())
    return df, meta


def load() -> tuple[pd.DataFrame, dict]:
    raw, meta = load_raw()
    return clean(raw, meta)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
