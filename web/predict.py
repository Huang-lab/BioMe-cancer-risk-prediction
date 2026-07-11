"""Framework-agnostic inference for the BioMe CRC risk model.

Loads the fitted sklearn Pipeline (.pkl) + model_metadata.json, scores a single
patient dict, buckets into LOW/MODERATE/HIGH via the tuned thresholds, and returns
the top contributing factors (SHAP for tree models; signed standardized
contributions for logistic). Import this from Streamlit now, or from FastAPI later
— the contract mirrors the reference repo's predict.py so the swap is trivial.

No PHI: inference runs on manually-entered values only.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_dir: str, tag: str = ""):
    model = joblib.load(os.path.join(model_dir, f"model{tag}.pkl"))
    with open(os.path.join(model_dir, f"model_metadata{tag}.json")) as fh:
        meta = json.load(fh)
    return model, meta


def _clean(name: str) -> str:
    return name.split("__", 1)[-1]


def top_factors(model, df_row: pd.DataFrame, meta: dict, k: int = 5):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    Xt = pre.transform(df_row)
    Xt = np.asarray(Xt.todense()) if hasattr(Xt, "todense") else np.asarray(Xt)
    names = list(pre.get_feature_names_out())

    contrib = None
    if meta.get("is_tree"):
        try:
            import shap
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xt)
            if isinstance(sv, list):          # older shap: [class0, class1]
                sv = sv[-1]
            sv = np.asarray(sv)
            if sv.ndim == 3:                  # shap>=0.5x: (n, features, classes)
                sv = sv[:, :, -1]
            contrib = sv[0]
        except Exception:
            contrib = None
    if contrib is None and hasattr(clf, "coef_"):
        contrib = clf.coef_[0] * Xt[0]
    if contrib is None:
        contrib = np.zeros(len(names))

    order = np.argsort(np.abs(contrib))[::-1][:k]
    return [{"feature": _clean(names[i]),
             "impact": float(contrib[i]),
             "direction": "increases risk" if contrib[i] > 0 else "decreases risk"}
            for i in order]


def predict_risk(patient: dict, model, meta: dict, k: int = 5) -> dict:
    """Score one patient. `patient` maps feature_cols -> values (missing -> defaults)."""
    defaults = meta.get("defaults", {})
    row = {c: patient.get(c, defaults.get(c)) for c in meta["feature_cols"]}
    df = pd.DataFrame([row])

    proba = float(model.predict_proba(df)[0][1])
    thr, high = meta["threshold"], meta["high_threshold"]
    level = "HIGH" if proba >= high else "MODERATE" if proba >= thr else "LOW"
    return {
        "risk_score": round(proba * 100, 1),
        "probability": proba,
        "risk_level": level,
        "threshold": thr,
        "high_threshold": high,
        "top_factors": top_factors(model, df, meta, k),
    }
