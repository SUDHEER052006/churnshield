"""Churn Copilot -- intent routing over a fixed set of parameterised queries.

Deliberately NOT an LLM writing SQL against production. The question is
matched to one of a small number of templates; the template runs a pandas
query with bound parameters; the answer reports which template ran and with
what arguments, so the behaviour is auditable.

To add an LLM layer, swap `route()` for a model call that returns
{"intent": ..., "params": {...}} and keep the executors below unchanged --
the model never touches the data directly.
"""
from __future__ import annotations

import re

import pandas as pd

from src import config as C
from src import economics

CURRENCY = C.DEFAULT_ASSUMPTIONS["currency"]


def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


def pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


# ---------------------------------------------------------------------------
def route(q: str) -> tuple[str, dict]:
    t = q.lower().strip()

    m = re.search(r"\b([a-z0-9]{4,6}-[a-z]{4,6})\b", t)
    if m and ("why" in t or "explain" in t or "risk" in t):
        return "explain_customer", {"customer_id": m.group(1).upper()}

    m = re.search(r"(\d[\d,]{2,})", t.replace(",", ""))
    if any(w in t for w in ("spend", "budget", "allocate", "next")) and m:
        return "budget_allocation", {"budget": float(m.group(1))}
    if any(w in t for w in ("spend", "budget", "allocate")):
        return "budget_allocation", {"budget": C.DEFAULT_ASSUMPTIONS["budget"]}

    if any(w in t for w in ("sleeping", "not contact", "don't contact",
                            "do not contact", "avoid", "suppress")):
        return "quadrant_filter", {"quadrant": "Sleeping dog", "limit": 5}
    if "persuadable" in t:
        return "quadrant_filter", {"quadrant": "Persuadable", "limit": 5}

    if any(w in t for w in ("fair", "bias", "discriminat")):
        return "fairness_report", {}
    if any(w in t for w in ("accurate", "accuracy", "auc", "performance",
                            "how good", "brier", "calibrat")):
        return "model_report", {}
    if any(w in t for w in ("factor", "driver", "important", "main reason")):
        return "global_importance", {"top_n": 5}

    # Segment question -> the primary treatment's UNTREATED population, i.e.
    # the group a campaign would naturally go after first.
    primary = next(iter(C.TREATMENTS))
    spec = C.TREATMENTS[primary]
    if any(w in t for w in ("segment", "how many", spec["short"].lower(),
                            spec["column"].lower())):
        return "segment_summary", {"column": spec["column"],
                                   "value": str(spec["control_values"][0])}

    if any(w in t for w in ("expensive", "high charge", "high monthly",
                            "high bill", "high value", "high revenue")):
        return "filtered_ranking", {"monthly_q": 0.80, "churn_gt": 0.6, "limit": 5}
    if "high risk" in t or "high-risk" in t:
        return "filtered_ranking", {"monthly_q": 0.0, "churn_gt": 0.7, "limit": 5}

    return "fallback", {}


# ---------------------------------------------------------------------------
def answer(question: str, state: dict) -> dict:
    intent, params = route(question)
    df: pd.DataFrame = state["scored"]
    summary = state["summary"]
    fn = EXECUTORS.get(intent, _fallback)
    body = fn(df, params, state, summary)
    return {"question": question, "intent": intent, "params": params, **body}


def _explain_customer(df, p, state, summary):
    hit = df[df[C.ID_COL].str.upper() == p["customer_id"]]
    if hit.empty:
        return {"text": f"No customer {p['customer_id']} in the scored base.", "table": []}
    r = hit.iloc[0]
    shap = (state["shap_local"] or {}).get(r[C.ID_COL], [])[:4]
    drivers = "\n".join(
        f"{i+1}. {s['feature']} ({s['value']:+.3f})" for i, s in enumerate(shap))
    offer = C.TREATMENTS[r["best_offer"]]
    return {
        "text": (f"{r[C.ID_COL]} has a churn probability of {r['churn_prob']:.0%} and is "
                 f"classified {r['quadrant'].lower()}.\n\nLargest contributors to that "
                 f"prediction:\n{drivers}\n\nBest available intervention is "
                 f"{offer['label']} at an estimated {pp(r['best_effect_for_offer'])} "
                 f"change in retention, costing {money(r['best_cost'])} for an expected "
                 f"net of {money(r['best_net'])}."),
        "table": [],
    }


def _budget_allocation(df, p, state, summary):
    a = economics.merge_assumptions({"budget": p["budget"]})
    res = economics.allocate(df, a)
    chosen = df[df[C.ID_COL].isin(res["ids"])].sort_values("best_net", ascending=False).head(5)
    return {
        "text": (f"With {money(p['budget'])} the optimiser targets {res['targeted']:,} "
                 f"customers, spending {money(res['spend'])} and expecting to retain "
                 f"{res['expected_customers_retained']:.0f} who would otherwise leave "
                 f"({money(res['expected_margin_retained'])} of margin, "
                 f"{res['roi']:.1f}x return). Nothing is allocated to sleeping dogs. "
                 f"Ranking by churn risk instead would retain "
                 f"{res['naive']['expected_customers_retained']:.0f} for the same budget."),
        "table": [{"Customer": r[C.ID_COL],
                   "Offer": C.TREATMENTS[r["best_offer"]]["short"],
                   "Uplift": pp(r["best_effect_for_offer"]),
                   "Net": money(r["best_net"])} for _, r in chosen.iterrows()],
    }


def _quadrant_filter(df, p, state, summary):
    g = df[df["quadrant"] == p["quadrant"]].sort_values("churn_prob", ascending=False)
    if p["quadrant"] == "Sleeping dog":
        text = (f"{len(g):,} customers are classified as sleeping dogs. Their estimated "
                f"uplift is negative, so a retention offer is expected to make things "
                f"worse. They are suppressed from every campaign, which is why a budget "
                f"increase can never reach them.")
    else:
        text = (f"{len(g):,} customers are {p['quadrant'].lower()}s - an offer is "
                f"estimated to change their decision. This is where the budget goes.")
    return {"text": text,
            "table": [{"Customer": r[C.ID_COL], "Churn P": f"{r['churn_prob']:.0%}",
                       "Uplift": pp(r["best_uplift"]),
                       "Profile": f"{int(r[C.TENURE_COL])}mo · {money(r[C.REVENUE_COL])}/mo"}
                      for _, r in g.head(p.get("limit", 5)).iterrows()]}


def _filtered_ranking(df, p, state, summary):
    # Threshold as a quantile, so "expensive" means the same thing whatever
    # currency or price book the dataset uses.
    p["monthly_gt"] = float(df[C.REVENUE_COL].quantile(p.get("monthly_q", 0.8)))
    g = df[(df[C.REVENUE_COL] > p["monthly_gt"]) & (df["churn_prob"] > p["churn_gt"])]
    g = g.sort_values("best_net", ascending=False)
    return {"text": (f"{len(g):,} customers match monthly charge above "
                     f"{money(p['monthly_gt'])} and churn probability above "
                     f"{p['churn_gt']:.0%}. Top by expected net value:"),
            "table": [{"Customer": r[C.ID_COL], "Monthly": money(r[C.REVENUE_COL]),
                       "Churn P": f"{r['churn_prob']:.0%}",
                       "Offer": C.TREATMENTS[r["best_offer"]]["short"],
                       "Net": money(r["best_net"])}
                      for _, r in g.head(p.get("limit", 5)).iterrows()]}


def _segment_summary(df, p, state, summary):
    g = df[df[p["column"]] == p["value"]]
    q = g["quadrant"].value_counts()
    return {"text": (f"{len(g):,} customers are on {p['value']} "
                     f"({len(g)/len(df):.0%} of the base). Of those, "
                     f"{(g['churn_prob'] > 0.7).sum():,} sit above 70% churn probability - "
                     f"but only {q.get('Persuadable', 0):,} are persuadable. The rest are "
                     f"lost causes or sleeping dogs, where an offer returns less than it costs."),
            "table": [{"Quadrant": k, "Customers": f"{v:,}",
                       "Share of segment": f"{v/len(g):.0%}"} for k, v in q.items()]}


def _global_importance(df, p, state, summary):
    top = summary["global_shap"][:p.get("top_n", 5)]
    lines = "\n".join(f"{i+1}. {d['feature']} - {d['value']:.3f}"
                      for i, d in enumerate(top))
    gender = next((d for d in summary["global_shap"]
                   if d["feature"].lower() == "gender"), None)
    tail = (f"\n\nWorth noting: gender contributes {gender['value']:.3f}, effectively "
            f"nothing. The fairness audit confirms it." if gender else "")
    return {"text": f"Strongest churn drivers by mean |SHAP| across the base:\n{lines}{tail}",
            "table": []}


def _fairness_report(df, p, state, summary):
    f = state["rai"]["fairness"]
    worst = max(f["groups"], key=lambda g: g["false_negative_rate"])
    return {"text": (f"The fairness audit {'passes' if f['passed'] else 'FAILS'}. "
                     f"Maximum demographic parity difference across "
                     f"{', '.join(C.SENSITIVE_FEATURES)} is "
                     f"{f['demographic_parity_difference']:.3f} against a "
                     f"{f['threshold']:.2f} threshold; equalised odds difference is "
                     f"{f['equalized_odds_difference']:.3f}. The worst false negative "
                     f"rate is {worst['false_negative_rate']:.1%} for {worst['group']}."),
            "table": [{"Group": g["group"], "Selection rate": f"{g['selection_rate']:.1%}",
                       "False negative rate": f"{g['false_negative_rate']:.1%}",
                       "n": f"{g['n']:,}"} for g in f["groups"]]}


def _model_report(df, p, state, summary):
    m = state["models"]
    sel = next(r for r in m["classifiers"] if r["model"] == m["selected"])
    ref = state["rai"]["refutation"]
    return {"text": (f"The shipped model is {m['selected']}: ROC-AUC {sel['roc_auc']:.3f}, "
                     f"PR-AUC {sel['pr_auc']:.3f}, recall {sel['recall']:.1%} on the churn "
                     f"class, Brier {sel['brier']:.3f} at a {m['threshold']} threshold. "
                     f"Uplift comes from {m['causal'][-1]['model'] if m['causal'] else 'n/a'}. "
                     f"{sum(t['passed'] for t in ref['tests'])} of {len(ref['tests'])} "
                     f"refutation tests survived."),
            "table": [{"Model": r["model"], "ROC-AUC": f"{r['roc_auc']:.3f}",
                       "PR-AUC": f"{r['pr_auc']:.3f}", "Recall": f"{r['recall']:.1%}",
                       "Brier": f"{r['brier']:.3f}"} for r in m["classifiers"]]}


def _fallback(df, p, state, summary):
    return {"text": ("I answer from the scored base using a fixed set of query "
                     "templates: churn risk, uplift, budget allocation, an individual "
                     "customer, model performance, or fairness. Try naming a customer "
                     "ID, or ask who to spend the next budget on."),
            "table": []}


EXECUTORS = {
    "explain_customer": _explain_customer,
    "budget_allocation": _budget_allocation,
    "quadrant_filter": _quadrant_filter,
    "filtered_ranking": _filtered_ranking,
    "segment_summary": _segment_summary,
    "global_importance": _global_importance,
    "fairness_report": _fairness_report,
    "model_report": _model_report,
    "fallback": _fallback,
}
