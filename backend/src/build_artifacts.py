"""Run the whole pipeline once and write everything the API serves.

    python -m src.build_artifacts

Order matters and mirrors the project's own story:
  data -> classifier -> SHAP -> causal uplift -> economics -> responsible AI
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from . import causal, config as C, data, economics, explain, rai, train


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def write(name: str, payload) -> None:
    path = C.ARTIFACT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"  wrote {path.relative_to(C.ROOT)}")


def main() -> None:
    t0 = time.time()
    print("\n=== ChurnShield pipeline ===\n")

    # ---- 1. data -------------------------------------------------------
    print("[1/6] Loading data")
    df, meta = data.load()
    print(f"      {meta['source']} - {meta['rows_clean']} rows, "
          f"base churn rate {meta['churn_base_rate']:.1%}")
    if meta["synthetic"]:
        print(f"      !! SYNTHETIC DATA. Drop the real {C.DATASET_DISPLAY} CSV into "
              f"backend/data/ and re-run for real figures.")

    # ---- 2. classifiers ------------------------------------------------
    print("[2/6] Training classifiers")
    report, final_model, shap_pipe, holdout = train.train_all(df)
    print(f"      selected: {report['selected']}")
    joblib.dump(final_model, C.MODEL_DIR / "churn_model.pkl")

    X = data.feature_frame(df)
    churn_prob = final_model.predict_proba(X)[:, 1]
    df["churn_prob"] = churn_prob

    # ---- 3. SHAP -------------------------------------------------------
    print("[3/6] Computing SHAP values")
    global_imp, shap_vals, shap_names, base_value = explain.compute(shap_pipe, X)
    print(f"      top driver: {global_imp[0]['feature']}")

    # ---- 4. causal uplift ----------------------------------------------
    print("[4/6] Estimating causal uplift (this is the slow one)")
    effects = causal.estimate_all_treatments(df)
    for key, e in effects.items():
        df[f"cate_{key}"] = e["effect"]
        df[f"cate_{key}_lo"] = e["lo"]
        df[f"cate_{key}_hi"] = e["hi"]
        print(f"      {key:9s} {e['estimator']:22s} mean uplift "
              f"{np.mean(e['effect']):+.4f}  qini {e['qini']:.3f}")

    cate_cols = [f"cate_{k}" for k in C.TREATMENTS]
    df["best_uplift"] = df[cate_cols].max(axis=1)
    df["best_uplift_offer"] = df[cate_cols].idxmax(axis=1).str.replace("cate_", "", regex=False)
    df["quadrant"] = [causal.classify_quadrant(p, u)
                      for p, u in zip(df["churn_prob"], df["best_uplift"])]
    print("      quadrants: " + ", ".join(
        f"{k} {v}" for k, v in df["quadrant"].value_counts().items()))

    print("      comparing uplift learners")
    causal_table = causal.compare_learners(df)

    # ---- 5. economics --------------------------------------------------
    print("[5/6] Scoring economics")
    a = economics.merge_assumptions()
    scored = economics.score_table(df, a)

    # ---- 6. responsible AI ---------------------------------------------
    print("[6/6] Responsible AI checks")
    y_pred = (churn_prob >= train.DECISION_THRESHOLD).astype(int)
    y_true = df["churn"].values
    fair = rai.fairness(df, y_true, y_pred)
    print(f"      fairness ({fair['library']}): DP diff {fair['demographic_parity_difference']:.3f}"
          f" -> {'PASS' if fair['passed'] else 'FAIL'}")
    ref = rai.refute(df)
    print(f"      refutation ({ref['library']}): "
          f"{sum(t['passed'] for t in ref['tests'])}/{len(ref['tests'])} survived")
    errs = rai.error_analysis(df, y_true, y_pred)
    print(f"      worst cohort: {errs['cohorts'][0]['cohort']} "
          f"({errs['cohorts'][0]['error_rate']:.1%})")

    # ---- persist -------------------------------------------------------
    print("\nWriting artifacts")
    keep = [C.ID_COL] + C.PROFILE_FIELDS + [
        "churn", "churn_prob", "quadrant", "best_uplift", "best_offer",
        "best_net", "best_cost", "best_effect_for_offer", "clv"]
    keep += [t["column"] for t in C.TREATMENTS.values()]
    keep += [c for c in scored.columns if c.startswith(("cate_", "cost_", "net_"))]
    keep = list(dict.fromkeys(c for c in keep if c in scored.columns))
    scored[keep].to_parquet(C.ARTIFACT_DIR / "scored.parquet", index=False) \
        if _parquet_ok() else scored[keep].to_csv(C.ARTIFACT_DIR / "scored.csv", index=False)

    # local SHAP for every customer, trimmed to the top contributors
    local = {}
    for i in range(len(df)):
        local[str(df.at[i, C.ID_COL])] = explain.local_top(
            shap_vals[i], shap_names, X.iloc[i], k=8)
    write("shap_local", local)

    curves = economics.targeting_curves(scored)
    primary = causal.primary_treatment()
    t_primary = causal._treatment_vector(scored, C.TREATMENTS[primary])
    mask = t_primary.notna().values
    qx, qy, qr, qc = causal.qini_curve(
        scored.loc[mask, "best_effect_for_offer"].values,
        t_primary[mask].values.astype(float),
        (1 - scored.loc[mask, "churn"]).values.astype(float))

    write("summary", {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": meta,
        "base_rate": meta["churn_base_rate"],
        "selected_model": report["selected"],
        "roc_auc": next(r["roc_auc"] for r in report["classifiers"]
                        if r["model"] == report["selected"]),
        "brier": next(r["brier"] for r in report["classifiers"]
                      if r["model"] == report["selected"]),
        "causal_estimator": effects[causal.primary_treatment()]["estimator"],
        "qini": max(e["qini"] for e in effects.values()),
        "refutations_passed": int(sum(t["passed"] for t in ref["tests"])),
        "refutations_total": len(ref["tests"]),
        "fairness_passed": fair["passed"],
        "runtime_seconds": round(time.time() - t0, 1),
        "global_shap": global_imp,
        "quadrants": [
            {"quadrant": q,
             "customers": int((scored["quadrant"] == q).sum()),
             "share": float((scored["quadrant"] == q).mean()),
             "annual_revenue": float(scored.loc[scored["quadrant"] == q,
                                                C.REVENUE_COL].sum() * 12)}
            for q in ["Persuadable", "Sure thing", "Lost cause", "Sleeping dog"]],
        "revenue_at_risk": float(scored.loc[scored["churn_prob"] >= 0.5,
                                            C.REVENUE_COL].sum() * 12),
        "customers_at_risk": int((scored["churn_prob"] >= 0.5).sum()),
        # How many customers each offer would actively HARM. A campaign that
        # blasts one offer at the high-risk tail hits every one of these.
        "harmful_by_offer": [
            {"offer": k, "label": C.TREATMENTS[k]["label"],
             "customers": int((scored[f"cate_{k}"] < C.SLEEPING_DOG_THRESHOLD).sum()),
             "annual_revenue": float(
                 scored.loc[scored[f"cate_{k}"] < C.SLEEPING_DOG_THRESHOLD,
                            C.REVENUE_COL].sum() * 12)}
            for k in C.TREATMENTS],
        "savable_revenue": float((scored.loc[scored["quadrant"] == "Persuadable",
                                             "best_effect_for_offer"] *
                                  scored.loc[scored["quadrant"] == "Persuadable",
                                             C.REVENUE_COL] * 12).sum()),
        "assumptions": {k: v for k, v in a.items()},
        "treatments": {k: {**{kk: v[kk] for kk in ("label", "short", "detail", "column")},
                                  "caveat": v.get("caveat")}
                       for k, v in C.TREATMENTS.items()},
        "scatter": [
            {"id": r[C.ID_COL], "churn": round(float(r["churn_prob"]), 4),
             "uplift": round(float(r["best_uplift"]), 4), "quadrant": r["quadrant"]}
            for _, r in scored.sample(min(900, len(scored)),
                                      random_state=C.RANDOM_STATE).iterrows()],
    })

    write("models", {**report, "causal": causal_table,
                     "qini_curve": [{"share": x, "qini": y, "random": r}
                                    for x, y, r in zip(qx, qy, qr)],
                     "qini_coefficient": qc})
    write("rai", {"fairness": fair, "refutation": ref, "errors": errs})
    write("curves", curves)

    print(f"\nDone in {time.time() - t0:.1f}s. Start the API with:")
    print("  .venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000\n")


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
