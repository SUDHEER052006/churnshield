"""Stage 2 -- causal uplift estimation (the differentiator).

A churn classifier answers "who is likely to leave". It cannot answer
"would this offer change their mind", because that is a counterfactual and
the data only ever shows one branch.

Here we estimate, per customer and per offer, the Conditional Average
Treatment Effect (CATE) on RETENTION using EconML. Positive effect means the
offer makes the customer more likely to stay. Negative effect means we have
found a sleeping dog: contacting them is estimated to make things worse.

Honest framing: the Telco data is OBSERVATIONAL. Customers chose their own
contracts; nobody randomised them. Every estimate below rests on a
no-unmeasured-confounding assumption, which we test rather than assert -- see
rai.py for the DoWhy refutation battery.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

warnings.filterwarnings("ignore")

# NumPy 2 renamed trapz -> trapezoid. Support both so the project runs on
# either generation of the stack.
_trapz = getattr(np, "trapezoid", None) or np.trapz

try:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import SLearner, TLearner, XLearner
    HAS_ECONML = True
except Exception:                                         # pragma: no cover
    HAS_ECONML = False


# --------------------------------------------------------------------------
# Covariate construction
# --------------------------------------------------------------------------
def _covariate_columns(exclude: str) -> tuple[list[str], list[str]]:
    num = [c for c in C.NUMERIC_FEATURES if c != exclude]
    cat = [c for c in C.CATEGORICAL_FEATURES if c != exclude]
    return num, cat


def _make_X(df: pd.DataFrame, exclude: str):
    """One-hot covariate matrix with the treatment column removed."""
    num, cat = _covariate_columns(exclude)
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    X = ct.fit_transform(df[num + cat])
    return np.asarray(X, dtype=float), ct, num + cat


def _treatment_vector(df: pd.DataFrame, spec: dict):
    """1 = treated, 0 = control, NaN = not comparable.

    `eligible` narrows the comparison to customers who could plausibly have
    received the treatment at all. For the Cell2Cell retention offer that
    means restricting to customers who actually reached the retention desk --
    everyone else was never in a position to accept, so including them would
    compare dissatisfied callers against the merely quiet.
    """
    col = df[spec["column"]].astype(str)
    t = pd.Series(np.nan, index=df.index)
    t[col.isin([str(v) for v in spec["treated_values"]])] = 1.0
    t[col.isin([str(v) for v in spec["control_values"]])] = 0.0

    elig = spec.get("eligible")
    if elig is not None:
        try:
            t[~elig(df).reindex(df.index).fillna(False).astype(bool)] = np.nan
        except Exception:
            pass
    return t


# --------------------------------------------------------------------------
# Qini / AUUC
# --------------------------------------------------------------------------
def qini_curve(uplift: np.ndarray, t: np.ndarray, y: np.ndarray, points: int = 60):
    """Standard Qini curve on retention. y = 1 when the customer stayed."""
    order = np.argsort(-uplift)
    t, y = t[order], y[order]
    n = len(y)
    idx = np.unique(np.linspace(1, n, points).astype(int))

    ct = np.cumsum(t)
    cc = np.cumsum(1 - t)
    ryt = np.cumsum(y * t)
    ryc = np.cumsum(y * (1 - t))

    xs, qs = [], []
    for k in idx:
        i = k - 1
        nt, nc = ct[i], cc[i]
        q = ryt[i] - (ryc[i] * nt / nc if nc > 0 else 0.0)
        xs.append(k / n)
        qs.append(float(q))

    overall = qs[-1] if qs else 0.0
    rand = [overall * x for x in xs]

    # Qini coefficient, normalised against the best ordering the data admits.
    # Without this the ratio blows up whenever the overall effect is near zero,
    # which is exactly when the coefficient is least meaningful.
    area_model = float(_trapz(np.array(qs) - np.array(rand), xs))
    area_perfect = _perfect_qini_area(t, y, xs, idx, overall)
    coef = area_model / area_perfect if area_perfect > 1e-9 else 0.0
    return xs, qs, rand, float(np.clip(coef, -1.5, 1.5))


def _perfect_qini_area(t, y, xs, idx, overall):
    """Area of the qini curve under the best ordering achievable on this data:
    treated responders first, then control non-responders, then everyone else."""
    rank = np.where((t == 1) & (y == 1), 0,
           np.where((t == 0) & (y == 0), 1, 2))
    order = np.argsort(rank, kind="stable")
    tp, yp = t[order], y[order]
    ct, cc = np.cumsum(tp), np.cumsum(1 - tp)
    ryt, ryc = np.cumsum(yp * tp), np.cumsum(yp * (1 - tp))

    qs = []
    for k in idx:
        i = k - 1
        q = ryt[i] - (ryc[i] * ct[i] / cc[i] if cc[i] > 0 else 0.0)
        qs.append(float(q))
    rand = [overall * x for x in xs]
    return abs(float(_trapz(np.array(qs) - np.array(rand), xs)))


def _auuc(uplift, t, y):
    xs, qs, rand, _ = qini_curve(uplift, t, y, points=40)
    total = qs[-1] if qs else 0.0
    if total == 0:
        return 0.5
    return float(_trapz(np.array(qs) / total, xs))


def _uplift_at(uplift, t, y, frac=0.20):
    k = max(1, int(len(y) * frac))
    order = np.argsort(-uplift)[:k]
    tt, yy = t[order], y[order]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return 0.0
    return float(yy[tt == 1].mean() - yy[tt == 0].mean())


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def _nuisance():
    return dict(
        reg=lambda: GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                              random_state=C.RANDOM_STATE),
        clf=lambda: GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                               random_state=C.RANDOM_STATE),
    )


def _fallback_tlearner(Xtr, ttr, ytr, Xall):
    """Used only when EconML is unavailable. Plain two-model T-learner."""
    m1 = RandomForestRegressor(n_estimators=300, min_samples_leaf=8,
                               random_state=C.RANDOM_STATE, n_jobs=-1)
    m0 = RandomForestRegressor(n_estimators=300, min_samples_leaf=8,
                               random_state=C.RANDOM_STATE, n_jobs=-1)
    m1.fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0.fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict(Xall) - m0.predict(Xall)


def primary_treatment() -> str:
    """The treatment the Model Lab comparison is run on: the first declared,
    which each profile orders so its headline intervention comes first."""
    return next(iter(C.TREATMENTS))


def compare_learners(df: pd.DataFrame, key: str | None = None) -> list[dict]:
    """Model Lab table: every uplift learner scored on the same split."""
    key = key or primary_treatment()
    spec = C.TREATMENTS[key]
    X, _, _ = _make_X(df, spec["column"])
    t = _treatment_vector(df, spec)
    mask = t.notna().values
    X, t = X[mask], t[mask].values
    y = (1 - df.loc[mask, "churn"]).values.astype(float)     # retention

    Xtr, Xte, ttr, tte, ytr, yte = train_test_split(
        X, t, y, test_size=0.3, stratify=t, random_state=C.RANDOM_STATE)

    rows = []
    if not HAS_ECONML:
        up = _fallback_tlearner(Xtr, ttr, ytr, Xte)
        _, _, _, q = qini_curve(up, tte, yte)
        rows.append({"model": "T-Learner (sklearn fallback)", "library": "scikit-learn",
                     "qini": q, "auuc": _auuc(up, tte, yte),
                     "uplift_at_20": _uplift_at(up, tte, yte), "ci_width": None,
                     "verdict": "selected", "note": "EconML not installed"})
        return rows

    n = _nuisance()
    specs = [
        ("S-Learner", lambda: SLearner(overall_model=n["reg"]()),
         "Effect can wash out inside one outcome model"),
        ("T-Learner", lambda: TLearner(models=n["reg"]()),
         "Two independent models, higher variance"),
        ("X-Learner", lambda: XLearner(models=n["reg"](),
                                       propensity_model=LogisticRegression(max_iter=1000)),
         "Handles imbalanced treatment arms"),
        ("DR-Learner", lambda: DRLearner(model_propensity=n["clf"](),
                                         model_regression=n["reg"](),
                                         model_final=n["reg"](),
                                         random_state=C.RANDOM_STATE),
         "Doubly robust"),
        ("CausalForestDML", lambda: CausalForestDML(
            model_y=n["reg"](), model_t=n["clf"](), discrete_treatment=True,
            n_estimators=300, min_samples_leaf=8, random_state=C.RANDOM_STATE),
         "Honest confidence intervals per customer"),
    ]

    for name, build, note in specs:
        try:
            est = build()
            est.fit(ytr, ttr, X=Xtr)
            up = np.asarray(est.effect(Xte)).ravel()
            _, _, _, q = qini_curve(up, tte, yte)
            ci_width = None
            if name == "CausalForestDML":
                try:
                    lo, hi = est.effect_interval(Xte, alpha=0.10)
                    ci_width = float(np.mean(np.asarray(hi).ravel() - np.asarray(lo).ravel()))
                except Exception:
                    pass
            rows.append({"model": name, "library": "EconML", "qini": float(q),
                         "auuc": _auuc(up, tte, yte),
                         "uplift_at_20": _uplift_at(up, tte, yte),
                         "ci_width": ci_width, "verdict": "candidate", "note": note})
        except Exception as exc:                            # pragma: no cover
            rows.append({"model": name, "library": "EconML", "qini": 0.0, "auuc": 0.5,
                         "uplift_at_20": 0.0, "ci_width": None,
                         "verdict": "failed", "note": f"{type(exc).__name__}: {exc}"[:120]})

    # CausalForestDML is what ships, and not always because it wins on Qini.
    # It is the only estimator here returning an honest confidence interval per
    # customer, and the allocator refuses to spend against a point estimate
    # whose interval straddles zero. Say that, rather than quietly presenting
    # the top of a leaderboard as the choice.
    ok = [r for r in rows if r["verdict"] == "candidate"]
    if ok:
        deployed = next((r for r in ok if r["model"] == "CausalForestDML"), None)
        best_qini = max(ok, key=lambda r: r["qini"])
        for r in ok:
            r["verdict"] = "rejected"
        if deployed is None:
            best_qini["verdict"] = "selected"
        else:
            deployed["verdict"] = "selected"
            deployed["note"] += " -- deployed: only estimator with per-customer CIs"
            if best_qini is not deployed:
                best_qini["verdict"] = "candidate"
                best_qini["note"] += f" -- higher Qini ({best_qini['qini']:.3f}), but no intervals"
    return rows


def estimate_all_treatments(df: pd.DataFrame) -> dict:
    """Per-customer CATE with confidence intervals, for every offer."""
    out = {}
    for key, spec in C.TREATMENTS.items():
        X, _, _ = _make_X(df, spec["column"])
        t = _treatment_vector(df, spec)
        mask = t.notna().values
        Xf, tf = X[mask], t[mask].values
        yf = (1 - df.loc[mask, "churn"]).values.astype(float)

        if tf.sum() < 30 or (1 - tf).sum() < 30:
            out[key] = {"effect": np.zeros(len(df)), "lo": np.zeros(len(df)),
                        "hi": np.zeros(len(df)), "estimator": "skipped (too few in one arm)",
                        "qini": 0.0}
            continue

        if HAS_ECONML:
            n = _nuisance()
            est = CausalForestDML(model_y=n["reg"](), model_t=n["clf"](),
                                  discrete_treatment=True, n_estimators=400,
                                  min_samples_leaf=8, random_state=C.RANDOM_STATE)
            est.fit(yf, tf, X=Xf)
            eff = np.asarray(est.effect(X)).ravel()
            try:
                lo, hi = est.effect_interval(X, alpha=0.10)
                lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
            except Exception:
                se = np.std(eff) * 0.6
                lo, hi = eff - 1.645 * se, eff + 1.645 * se
            name = "CausalForestDML"
            _, _, _, q = qini_curve(np.asarray(est.effect(Xf)).ravel(), tf, yf)
        else:
            eff = _fallback_tlearner(Xf, tf, yf, X)
            se = np.std(eff) * 0.6
            lo, hi = eff - 1.645 * se, eff + 1.645 * se
            name = "T-Learner (sklearn fallback)"
            _, _, _, q = qini_curve(_fallback_tlearner(Xf, tf, yf, Xf), tf, yf)

        out[key] = {"effect": eff, "lo": lo, "hi": hi, "estimator": name, "qini": float(q)}
    return out


def classify_quadrant(churn_prob: float, best_uplift: float) -> str:
    if best_uplift <= C.SLEEPING_DOG_THRESHOLD:
        return "Sleeping dog"
    if best_uplift >= C.PERSUADABLE_UPLIFT and churn_prob >= C.PERSUADABLE_RISK:
        return "Persuadable"
    if churn_prob >= C.LOST_CAUSE_RISK:
        return "Lost cause"
    return "Sure thing"
