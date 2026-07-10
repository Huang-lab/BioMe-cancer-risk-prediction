#!/usr/bin/env python3
"""Stage 7 — evaluation.

For the PRIMARY (matched) model:
  - honest out-of-fold metrics: AUC, PR-AUC, Brier
  - ancestry-stratified calibration (per Newgroups: AUC, Brier, ECE, cal slope/intercept)
  - global feature importance (SHAP for trees; |standardized coef| for logistic)
  - per-gene carrier enrichment (case vs control freq, ratio, Fisher p)
  - cross-cohort validation (train I -> test II, and reverse)
Writes evaluation.json + CSVs + PNGs under outdir.

  python scripts/evaluate.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, modeling, util  # noqa: E402
from train import get_cv  # reuse the exact CV scheme  # noqa: E402

LOG = util.get_logger("evaluate")


def calibration_stats(y, p, bins):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y, p))
    # calibration slope/intercept: y ~ logit(p)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = intercept = np.nan
    if len(np.unique(y)) == 2:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
        slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    # ECE
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return {"brier": brier, "cal_slope": slope, "cal_intercept": intercept, "ece": float(ece)}


def stratified(y, p, groups, bins):
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            out[str(g)] = {"n": int(m.sum()), "note": "too few / single-class; skipped"}
            continue
        s = {"n": int(m.sum()), "n_cases": int(y[m].sum()),
             "auc": float(roc_auc_score(y[m], p[m])),
             "pr_auc": float(average_precision_score(y[m], p[m]))}
        s.update(calibration_stats(y[m], p[m], bins))
        out[str(g)] = s
    return out


def global_importance(model, X, feature_names, is_tree, top_k):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    Xt = np.asarray(Xt.todense()) if hasattr(Xt, "todense") else np.asarray(Xt)
    if is_tree:
        try:
            import shap
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            imp = np.abs(sv).mean(axis=0)
        except Exception as e:  # pragma: no cover
            LOG.warning("SHAP failed (%s); using tree feature_importances_", e)
            imp = getattr(clf, "feature_importances_", np.zeros(len(names)))
    else:
        imp = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else np.zeros(len(names))
    df = pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names], "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def carrier_enrichment(df, panel, aggregates, label):
    rows = []
    for gene in panel + aggregates:
        col = f"{gene}_any" if f"{gene}_any" in df.columns else gene
        if col not in df.columns:
            continue
        case = df[df[label] == 1][col]
        ctrl = df[df[label] == 0][col]
        a, b = int(case.sum()), int(len(case) - case.sum())
        c, d = int(ctrl.sum()), int(len(ctrl) - ctrl.sum())
        fcase, fctrl = (a / len(case) if len(case) else 0), (c / len(ctrl) if len(ctrl) else 0)
        try:
            _, p = fisher_exact([[a, b], [c, d]])
        except Exception:
            p = np.nan
        ratio = (fcase / fctrl) if fctrl > 0 else np.inf
        rows.append({"gene": col.replace("_any", ""), "case_freq": round(fcase, 4),
                     "control_freq": round(fctrl, 4), "enrichment": round(ratio, 2),
                     "p_value": p, "n_case_carriers": a, "n_control_carriers": c})
    return pd.DataFrame(rows).sort_values("p_value")


def cross_cohort(cfg, df_full, spec, seed):
    cohorts = list(pd.unique(df_full[spec["cohort_col"]]))
    if len(cohorts) < 2:
        return {"note": "single cohort; cross-cohort validation skipped"}
    name = util.load_json  # placeholder to avoid lint; not used
    results = {}
    pre = modeling.make_preprocessor(spec["numeric"], spec["categorical"])
    # use logistic as a stable default for the generalization check
    for train_c in cohorts:
        for test_c in cohorts:
            if train_c == test_c:
                continue
            tr = df_full[df_full[spec["cohort_col"]] == train_c]
            te = df_full[df_full[spec["cohort_col"]] == test_c]
            if len(np.unique(te[spec["label"]])) < 2:
                continue
            from sklearn.pipeline import Pipeline
            est = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
            pipe = Pipeline([("pre", clone(pre)), ("clf", est)])
            pipe.fit(tr[spec["feature_cols"]], tr[spec["label"]].astype(int))
            p = pipe.predict_proba(te[spec["feature_cols"]])[:, 1]
            y = te[spec["label"]].astype(int).to_numpy()
            results[f"train_{train_c}__test_{test_c}"] = {
                "auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p)),
                "n_train": int(len(tr)), "n_test": int(len(te))}
    return results


def main():
    args = cli.base_parser("Evaluation").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    seed = cfg.get("random_state", 42)
    ev = cfg["evaluation"]
    spec = util.load_json(os.path.join(out, "feature_spec.json"))

    meta = util.load_json(os.path.join(out, "model_metadata.json"))
    model = joblib.load(os.path.join(out, "model.pkl"))
    df = pd.read_csv(os.path.join(out, "dataset_matched.csv"))
    df_full = pd.read_csv(os.path.join(out, "dataset_full.csv"))

    X = df[spec["feature_cols"]]
    y = df[spec["label"]].astype(int).to_numpy()
    groups = df[spec["group_col"]].to_numpy()
    cv = get_cv("matched", cfg["model"]["cv_folds"], seed)

    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(clone(model), X, y, cv=cv, groups=groups,
                            method="predict_proba", n_jobs=-1)[:, 1]

    overall = {"auc": float(roc_auc_score(y, oof)),
               "pr_auc": float(average_precision_score(y, oof)),
               "n": int(len(y)), "n_cases": int(y.sum())}
    overall.update(calibration_stats(y, oof, ev["calibration_bins"]))
    strata = stratified(y, oof, df[spec["ancestry_col"]].to_numpy(), ev["calibration_bins"])
    LOG.info("overall (OOF): AUC=%.3f PR-AUC=%.3f Brier=%.3f ECE=%.3f",
             overall["auc"], overall["pr_auc"], overall["brier"], overall["ece"])
    for g, s in strata.items():
        if "auc" in s:
            LOG.info("  ancestry %s (n=%d): AUC=%.3f cal_slope=%.2f ECE=%.3f",
                     g, s["n"], s["auc"], s["cal_slope"], s["ece"])

    imp = global_importance(model, X, spec["feature_cols"], meta["is_tree"], ev["shap_top_k"])
    imp.to_csv(os.path.join(out, "feature_importance.csv"), index=False)

    enr = carrier_enrichment(df_full, cfg["genomics"]["panel"],
                             [cfg["genomics"]["aggregate_flag"], cfg["genomics"]["extra_aggregate"]],
                             spec["label"])
    enr.to_csv(os.path.join(out, "carrier_enrichment.csv"), index=False)
    top_enr = enr.head(3).to_dict("records")
    LOG.info("carrier enrichment (top): %s",
             [(r["gene"], f"{r['enrichment']}x", f"p={r['p_value']:.1e}") for r in top_enr])

    xc = cross_cohort(cfg, df_full, spec, seed)
    if "note" not in xc:
        for k, v in xc.items():
            LOG.info("cross-cohort %s: AUC=%.3f PR-AUC=%.3f", k, v["auc"], v["pr_auc"])

    # ---- plots ----
    _plot_calibration(y, oof, df[spec["ancestry_col"]].to_numpy(), ev["calibration_bins"],
                      os.path.join(out, "calibration.png"))
    _plot_importance(imp, os.path.join(out, "feature_importance.png"))

    report = {"cancer": cfg["cancer"], "model": meta["model_name"], "population": "matched",
              "overall": overall, "by_ancestry": strata, "cross_cohort": xc,
              "top_features": imp.to_dict("records"), "carrier_enrichment_top": top_enr}
    util.save_json(report, os.path.join(out, "evaluation.json"))
    LOG.info("wrote evaluation.json, feature_importance.csv, carrier_enrichment.csv, PNGs -> %s", out)


def _plot_calibration(y, p, groups, bins, path):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    try:
        fp, mp = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        plt.plot(mp, fp, "o-", label="overall")
    except Exception:
        pass
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() >= 20 and len(np.unique(y[m])) == 2:
            try:
                fp, mp = calibration_curve(y[m], p[m], n_bins=min(bins, 5), strategy="quantile")
                plt.plot(mp, fp, ".-", alpha=0.7, label=f"{g} (n={m.sum()})")
            except Exception:
                pass
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration (overall + by ancestry)"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def _plot_importance(imp, path):
    plt.figure(figsize=(6, max(3, 0.3 * len(imp))))
    d = imp.iloc[::-1]
    plt.barh(d["feature"], d["importance"])
    plt.xlabel("importance"); plt.title("Top features")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


if __name__ == "__main__":
    main()
