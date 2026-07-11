#!/usr/bin/env python3
"""Stage 8 — external validation on held-out cohort(s).

Applies the PRIMARY model (trained on cohort_strategy.train_cohorts, e.g. Cohort I)
to the holdout cohort(s) it was NEVER trained on (e.g. Cohort II), and reports:
  - AUC / PR-AUC / Brier on the holdout
  - overall + ancestry-stratified calibration (slope/intercept, ECE) + a PNG
  - per-gene carrier enrichment on the holdout

Honest external validation requires the holdout cohort to be excluded from
train_cohorts (it is, by config). Skips gracefully if the holdout cohort has not
been preprocessed yet (e.g. Cohort II clinical dir not wired).

  python scripts/external_validate.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, util  # noqa: E402
import evaluate as E  # reuse calibration_stats / stratified / carrier_enrichment / plotting  # noqa: E402

LOG = util.get_logger("external_validate")


def main():
    args = cli.base_parser("External validation on holdout cohort(s)").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    ev = cfg["evaluation"]
    spec = util.load_json(os.path.join(out, "feature_spec.json"))

    strat = cfg.get("cohort_strategy") or {}
    holdout = strat.get("holdout_cohorts") or []
    if not holdout:
        LOG.info("no holdout_cohorts configured; nothing to externally validate")
        return

    df_full = pd.read_csv(os.path.join(out, "dataset_full.csv"))
    sub = df_full[df_full[spec["cohort_col"]].isin(holdout)].copy()
    if sub.empty:
        LOG.warning("holdout cohorts %s have no data (not preprocessed / not wired yet) — "
                    "skipping external validation", holdout)
        return
    y = sub[spec["label"]].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        LOG.warning("holdout %s has a single class (cases=%d) — cannot compute AUC; skipping",
                    holdout, int(y.sum()))
        return

    model = joblib.load(os.path.join(out, "model.pkl"))
    meta = util.load_json(os.path.join(out, "model_metadata.json"))
    p = model.predict_proba(sub[spec["feature_cols"]])[:, 1]

    overall = {"auc": float(roc_auc_score(y, p)),
               "pr_auc": float(average_precision_score(y, p)),
               "n": int(len(y)), "n_cases": int(y.sum())}
    overall.update(E.calibration_stats(y, p, ev["calibration_bins"]))
    strata = E.stratified(y, p, sub[spec["ancestry_col"]].to_numpy(), ev["calibration_bins"])

    tag = "_".join(holdout)
    enr = E.carrier_enrichment(
        sub, cfg["genomics"]["panel"],
        [cfg["genomics"]["aggregate_flag"], cfg["genomics"]["extra_aggregate"]], spec["label"])
    enr.to_csv(os.path.join(out, f"carrier_enrichment_{tag}.csv"), index=False)
    E._plot_calibration(y, p, sub[spec["ancestry_col"]].to_numpy(), ev["calibration_bins"],
                        os.path.join(out, f"calibration_{tag}.png"))

    report = {"trained_on": strat.get("train_cohorts"), "validated_on": holdout,
              "model": meta["model_name"], "overall": overall, "by_ancestry": strata,
              "carrier_enrichment_top": enr.head(5).to_dict("records")}
    util.save_json(report, os.path.join(out, f"external_validation_{tag}.json"))

    LOG.info("external validation [train=%s -> test=%s]: AUC=%.3f PR-AUC=%.3f Brier=%.3f "
             "cal_slope=%.2f ECE=%.3f (n=%d, cases=%d)",
             strat.get("train_cohorts"), holdout, overall["auc"], overall["pr_auc"],
             overall["brier"], overall["cal_slope"], overall["ece"], overall["n"], overall["n_cases"])
    for g, s in strata.items():
        if "auc" in s:
            LOG.info("  [%s] ancestry %s (n=%d): AUC=%.3f cal_slope=%.2f ECE=%.3f",
                     tag, g, s["n"], s["auc"], s["cal_slope"], s["ece"])
    LOG.info("wrote external_validation_%s.json, calibration_%s.png, carrier_enrichment_%s.csv -> %s",
             tag, tag, tag, out)


if __name__ == "__main__":
    main()
