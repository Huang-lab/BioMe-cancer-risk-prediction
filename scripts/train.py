#!/usr/bin/env python3
"""Stage 6 — train, tune, and export the model(s).

For each population (matched = PRIMARY, full = comparison):
  - RandomizedSearchCV over LR / RandomForest / XGBoost (PR-AUC scoring)
  - StratifiedGroupKFold keyed on matched_set (matched) so a case + its controls
    never split across folds; StratifiedKFold for the full cohort
  - recall-oriented threshold subject to a precision floor, from OOF predictions
  - export the fitted Pipeline (.pkl) + model_metadata.json (feature contract,
    thresholds, tiers, categorical levels for the web form)

  python scripts/train.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, modeling, util  # noqa: E402

LOG = util.get_logger("train")
HIGH_TIER = 0.6


def tune_threshold(y, proba, min_precision):
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    ok = prec >= min_precision
    if ok.any():
        idx_candidates = np.where(ok)[0]
        best = idx_candidates[np.argmax(rec[idx_candidates])]
    else:
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best = int(np.argmax(f1))
    return float(thr[best]), float(prec[best]), float(rec[best])


def get_cv(population, n_splits, seed):
    if population == "matched":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def train_population(cfg, df, spec, population, out):
    seed = cfg.get("random_state", 42)
    mcfg = cfg["model"]
    feature_cols = spec["feature_cols"]
    X = df[feature_cols]
    y = df[spec["label"]].astype(int).to_numpy()
    groups = df[spec["group_col"]].to_numpy() if population == "matched" else None

    imbalance = mcfg["imbalance"][population]
    class_weight = "balanced" if imbalance == "class_weight" else None
    spw = modeling.pos_weight(y) if imbalance == "class_weight" else None

    pre = modeling.make_preprocessor(spec["numeric"], spec["categorical"])
    cv = get_cv(population, mcfg["cv_folds"], seed)
    n_splits = mcfg["cv_folds"]

    best = None  # (score, name, fitted_search)
    for name in mcfg["candidates"]:
        est, params = modeling.candidate(name, class_weight, spw, seed)
        if est is None:
            LOG.warning("  %s unavailable (xgboost not installed), skipping", name)
            continue
        pipe = Pipeline([("pre", pre), ("clf", est)])
        search = RandomizedSearchCV(
            pipe, params, n_iter=mcfg["search_iters"], scoring="average_precision",
            cv=cv, random_state=seed, n_jobs=-1, error_score="raise", refit=True)
        search.fit(X, y, groups=groups)
        LOG.info("  [%s] %s CV PR-AUC=%.3f", population, name, search.best_score_)
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, name, search)

    score, name, search = best
    LOG.info("[%s] selected %s (CV PR-AUC=%.3f)", population, name, score)

    # honest OOF predictions for metrics + threshold
    oof = cross_val_predict(clone(search.best_estimator_), X, y, cv=cv,
                            groups=groups, method="predict_proba", n_jobs=-1)[:, 1]
    auc = float(roc_auc_score(y, oof))
    pr_auc = float(average_precision_score(y, oof))
    thr, prec, rec = tune_threshold(y, oof, mcfg["threshold_tuning"]["min_precision"])
    LOG.info("[%s] OOF AUC=%.3f PR-AUC=%.3f | threshold=%.3f (precision=%.2f recall=%.2f)",
             population, auc, pr_auc, thr, prec, rec)

    model = search.best_estimator_  # refit on all data
    cat_levels = {c: sorted(df[c].dropna().astype(str).unique().tolist())
                  for c in spec["categorical"]}
    defaults = {}
    for c in spec["numeric"]:
        defaults[c] = float(pd.to_numeric(df[c], errors="coerce").median())
    for c in spec["categorical"]:
        m = df[c].dropna().astype(str)
        defaults[c] = m.mode().iloc[0] if not m.empty else ""

    tag = "" if population == "matched" else "_full"
    joblib.dump(model, os.path.join(out, f"model{tag}.pkl"))
    meta = {
        "cancer": cfg["cancer"], "label": cfg["label"], "population": population,
        "model_name": name, "version": f"v1_{population}",
        "feature_cols": feature_cols, "numeric": spec["numeric"],
        "categorical": spec["categorical"], "categorical_levels": cat_levels,
        "pc_cols": spec["pc_cols"], "defaults": defaults,
        "threshold": thr, "high_threshold": HIGH_TIER,
        "metrics": {"auc": auc, "pr_auc": pr_auc, "cv_pr_auc": score,
                    "precision_at_threshold": prec, "recall_at_threshold": rec,
                    "n": int(len(df)), "n_cases": int(y.sum())},
        "cv": {"scheme": "stratified_group" if population == "matched" else "stratified",
               "n_splits": n_splits},
        "is_tree": name in modeling.TREE_MODELS,
    }
    util.save_json(meta, os.path.join(out, f"model_metadata{tag}.json"))
    LOG.info("[%s] wrote model%s.pkl + model_metadata%s.json", population, tag, tag)
    return meta


def main():
    ap = cli.base_parser("Train + tune models")
    ap.add_argument("--population", choices=["matched", "full", "both"], default="both")
    args = ap.parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    spec = util.load_json(os.path.join(out, "feature_spec.json"))

    train_cohorts = (cfg.get("cohort_strategy") or {}).get("train_cohorts")
    pops = ["matched", "full"] if args.population == "both" else [args.population]
    for population in pops:
        path = os.path.join(out, f"dataset_{population}.csv")
        df = pd.read_csv(path)
        if train_cohorts:
            before = len(df)
            df = df[df[spec["cohort_col"]].isin(train_cohorts)].copy()
            LOG.info("restricting training to cohorts %s: %d -> %d subjects",
                     train_cohorts, before, len(df))
        LOG.info("=== population=%s: %d subjects, %d cases ===",
                 population, len(df), int(df[spec["label"]].sum()))
        train_population(cfg, df, spec, population, out)


if __name__ == "__main__":
    main()
