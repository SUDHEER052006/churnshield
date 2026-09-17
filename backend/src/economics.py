"""Turning a treatment effect into a business decision.

    expected net value = CATE (extra probability of staying)
                         x customer lifetime value at risk
                         - cost of the offer

Then a budget-constrained allocation over those net values. Every constant
here comes from config.DEFAULT_ASSUMPTIONS and is overridable at request time,
which is why the Budget page can say honestly that nothing is hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def merge_assumptions(overrides: dict | None = None) -> dict:
    a = dict(C.DEFAULT_ASSUMPTIONS)
    if overrides:
        a.update({k: v for k, v in overrides.items() if v is not None})
    return a


def customer_value(row, a: dict) -> float:
    """Gross margin at risk over the value horizon."""
    return float(row[C.REVENUE_COL]) * a["horizon_months"] * a["gross_margin"]


def offer_cost(row, key: str, a: dict) -> float:
    return float(C.TREATMENTS[key]["cost"](row, a))


def score_table(scored: pd.DataFrame, a: dict) -> pd.DataFrame:
    """Adds cost / value / net columns for every offer, plus the best play."""
    df = scored.copy()
    df["clv"] = df.apply(lambda r: customer_value(r, a), axis=1)

    best_key, best_net, best_cost, best_eff = [], [], [], []
    for _, r in df.iterrows():
        bk, bn, bc, be = None, -np.inf, 0.0, 0.0
        for key in C.TREATMENTS:
            eff = float(r[f"cate_{key}"])
            cost = offer_cost(r, key, a)
            net = eff * r["clv"] - cost
            df.at[_, f"cost_{key}"] = cost
            df.at[_, f"net_{key}"] = net
            if net > bn:
                bk, bn, bc, be = key, net, cost, eff
        best_key.append(bk); best_net.append(bn); best_cost.append(bc); best_eff.append(be)

    df["best_offer"] = best_key
    df["best_net"] = best_net
    df["best_cost"] = best_cost
    df["best_effect_for_offer"] = best_eff
    return df


def allocate(df: pd.DataFrame, a: dict) -> dict:
    """Greedy knapsack by net value per unit of spend.

    Sleeping dogs are excluded structurally, not by ranking -- they must never
    be reachable by a budget increase.
    """
    pool = df[(df["quadrant"] != "Sleeping dog") & (df["best_net"] > 0)].copy()
    pool["ratio"] = pool["best_net"] / pool["best_cost"].clip(lower=1e-6)
    pool = pool.sort_values("ratio", ascending=False)

    budget = float(a["budget"])
    spend, rows = 0.0, []
    for _, r in pool.iterrows():
        if spend + r["best_cost"] > budget:
            continue
        spend += float(r["best_cost"])
        rows.append(r)
    chosen = pd.DataFrame(rows) if rows else pool.head(0)

    retained = float((chosen["best_uplift"] if "best_uplift" in chosen else
                      chosen.get("best_effect_for_offer", pd.Series(dtype=float))).sum()) \
        if len(chosen) else 0.0
    margin = float((chosen["best_effect_for_offer"] * chosen["clv"]).sum()) if len(chosen) else 0.0

    # Counterfactual: the same budget spent on the highest-risk customers.
    naive = df.sort_values("churn_prob", ascending=False)
    n_spend, n_ret, n_margin = 0.0, 0.0, 0.0
    for _, r in naive.iterrows():
        if n_spend + r["best_cost"] > budget:
            continue
        n_spend += float(r["best_cost"])
        n_ret += float(r["best_effect_for_offer"])
        n_margin += float(r["best_effect_for_offer"] * r["clv"])

    mix = []
    if len(chosen):
        for key in C.TREATMENTS:
            g = chosen[chosen["best_offer"] == key]
            if len(g):
                mix.append({"offer": key, "label": C.TREATMENTS[key]["label"],
                            "customers": int(len(g)), "spend": float(g["best_cost"].sum())})
        mix.sort(key=lambda m: -m["spend"])

    return {
        "targeted": int(len(chosen)),
        "spend": spend,
        "budget": budget,
        "expected_customers_retained": retained,
        "expected_margin_retained": margin,
        "roi": (margin / spend) if spend > 0 else 0.0,
        "naive": {"spend": n_spend, "expected_customers_retained": n_ret,
                  "expected_margin_retained": n_margin},
        "lift_vs_risk_targeting": ((retained / n_ret - 1.0) if n_ret > 0 else 0.0),
        "offer_mix": mix,
        "ids": chosen[C.ID_COL].tolist()[:500] if len(chosen) else [],
    }


def targeting_curves(df: pd.DataFrame, points: int = 60) -> dict:
    """Cumulative expected retention as you work down each ranking.

    This is the chart that carries the whole argument: uplift ranking versus
    risk ranking versus random, on the same axis.
    """
    n = len(df)
    by_uplift = df.sort_values("best_effect_for_offer", ascending=False)["best_effect_for_offer"].values
    by_risk = df.sort_values("churn_prob", ascending=False)["best_effect_for_offer"].values
    rng = np.random.default_rng(C.RANDOM_STATE)
    by_rand = df["best_effect_for_offer"].sample(frac=1.0, random_state=C.RANDOM_STATE).values

    cu, cr, cd = np.cumsum(by_uplift), np.cumsum(by_risk), np.cumsum(by_rand)
    idx = np.unique(np.linspace(1, n, points).astype(int)) - 1
    return {
        "points": [
            {"share": float((i + 1) / n), "uplift": float(cu[i]),
             "risk": float(cr[i]), "random": float(cd[i])}
            for i in idx
        ]
    }
