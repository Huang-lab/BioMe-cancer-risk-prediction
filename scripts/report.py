#!/usr/bin/env python3
"""Stage 9 — presentation report.

Consolidates evaluate.py / external_validate.py outputs (metrics, tuning
params, feature importance, calibration figures, matched-vs-full comparison,
cross-cohort + external validation) into a single Markdown file, report.md,
meant to be read from directly or converted to slides -- nothing is
recomputed here, it only reads the JSON/CSV/PNG artifacts those stages wrote.

  python scripts/report.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, util  # noqa: E402

LOG = util.get_logger("report")


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def _by_ancestry_table(strata):
    lines = ["| Ancestry | n | AUC | cal. slope | ECE |", "|---|---|---|---|---|"]
    for g, s in strata.items():
        if "auc" in s:
            lines.append(f"| {g} | {s['n']} | {_fmt(s['auc'])} | {_fmt(s['cal_slope'], 2)} | "
                         f"{_fmt(s['ece'])} |")
        else:
            lines.append(f"| {g} | {s['n']} | - | - | too few / single class, skipped |")
    return "\n".join(lines)


def _features_table(rows, k=15):
    lines = ["| # | feature | importance |", "|---|---|---|"]
    for i, r in enumerate(rows[:k], 1):
        lines.append(f"| {i} | {r['feature']} | {r['importance']:.4f} |")
    return "\n".join(lines)


def main():
    args = cli.base_parser("Presentation report").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)

    spec = util.load_json(os.path.join(out, "feature_spec.json"))
    meta = util.load_json(os.path.join(out, "model_metadata.json"))
    ev_report = util.load_json(os.path.join(out, "evaluation.json"))
    match_meta = util.load_json(os.path.join(out, "match_meta.json"))

    ev_full = None
    full_path = os.path.join(out, "evaluation_full.json")
    if os.path.exists(full_path):
        ev_full = util.load_json(full_path)

    ext_reports = [(os.path.basename(p), util.load_json(p))
                  for p in sorted(glob.glob(os.path.join(out, "external_validation_*.json")))]

    lines = []
    a = lines.append
    cancer_label = cfg.get("label", cfg["cancer"])

    a(f"# {cancer_label} risk model -- results summary\n")
    a(f"Cancer: `{cfg['cancer']}` | Config: `{args.config}` | "
      f"Model selected: **{meta['model_name']}** "
      f"(candidates tried: {', '.join(meta.get('candidates_tried', []))})\n")

    # ---- executive summary ----
    a("## Executive summary\n")
    a("| Population | n | n cases | AUC | PR-AUC | Brier | ECE |")
    a("|---|---|---|---|---|---|---|")
    o = ev_report["overall"]
    a(f"| Matched (primary) | {o['n']} | {o['n_cases']} | {_fmt(o['auc'])} | "
      f"{_fmt(o['pr_auc'])} | {_fmt(o['brier'])} | {_fmt(o['ece'])} |")
    if ev_full:
        of = ev_full["overall"]
        a(f"| Full cohort (class-weighted, comparison) | {of['n']} | {of['n_cases']} | "
          f"{_fmt(of['auc'])} | {_fmt(of['pr_auc'])} | {_fmt(of['brier'])} | {_fmt(of['ece'])} |")
    for _, r in ext_reports:
        holdout = "+".join(r.get("validated_on", []))
        oe = r["overall"]
        a(f"| External validation ({holdout}) | {oe['n']} | {oe['n_cases']} | "
          f"{_fmt(oe['auc'])} | {_fmt(oe['pr_auc'])} | {_fmt(oe['brier'])} | {_fmt(oe['ece'])} |")
    a("")
    a("All numbers above are out-of-fold (matched/full) or applied-to-holdout "
      "(external) — none are in-sample.\n")

    # ---- model & tuning ----
    a("## Model & tuning\n")
    a(f"- Candidates tried: {', '.join(meta.get('candidates_tried', []))} "
      f"(RandomizedSearchCV, {meta['cv'].get('search_iters', '?')} iterations, scored on PR-AUC)")
    a(f"- Selected: **{meta['model_name']}** (CV PR-AUC = {_fmt(meta['metrics']['cv_pr_auc'])})")
    a(f"- CV scheme: {meta['cv']['scheme']} ({meta['cv']['n_splits']} folds"
      + (", grouped on matched-set id so a case and its matched controls never split"
         if meta["population"] == "matched" else "") + ")")
    a(f"- Decision threshold: {_fmt(meta['threshold'])} "
      f"(precision={_fmt(meta['metrics']['precision_at_threshold'])}, "
      f"recall={_fmt(meta['metrics']['recall_at_threshold'])}; "
      f"high-risk tier >= {meta['high_threshold']})")
    a("- Best hyperparameters (winning candidate):")
    for k, v in meta.get("best_params", {}).items():
        a(f"  - `{k}` = {v}")
    a("")

    # ---- features ----
    a("## Features\n")
    a(f"- {len(spec['feature_cols'])} features total "
      f"({len(spec['numeric'])} numeric, {len(spec['categorical'])} categorical) "
      f"-- full list in `feature_spec.json`")
    shap_note = ("SHAP (TreeExplainer, mean |SHAP value|)" if meta["is_tree"] else
                "standardized logistic-regression |coefficient| "
                "(SHAP only runs for tree candidates; the selected model here is not one)")
    a(f"- Importance method: {shap_note}\n")
    a(f"### Top {min(15, len(ev_report['top_features']))} features (matched model)\n")
    a(_features_table(ev_report["top_features"]))
    a("")
    a("![feature importance](feature_importance.png)\n")

    # ---- calibration ----
    a("## Calibration & subgroup fairness (matched, primary)\n")
    a(_by_ancestry_table(ev_report["by_ancestry"]))
    a("")
    a("![calibration](calibration.png)\n")
    if ev_full:
        a("## Calibration (full cohort, comparison)\n")
        a("Read this curve for real-world absolute risk. The matched curve above reflects "
          "the matched sample's artificial case:control ratio, not true population "
          "prevalence -- matching is for confounder balance, not calibration.\n")
        a(_by_ancestry_table(ev_full["by_ancestry"]))
        a("")
        a("![calibration full cohort](calibration_full.png)\n")

    # ---- genomics ----
    carrier_path = os.path.join(out, "carrier_enrichment.csv")
    if os.path.exists(carrier_path):
        enr = pd.read_csv(carrier_path)
        a("## Germline carrier enrichment (case vs. control)\n")
        a("| Gene | Case freq | Control freq | Enrichment | p-value |")
        a("|---|---|---|---|---|")
        for _, r in enr.head(10).iterrows():
            a(f"| {r['gene']} | {r['case_freq']} | {r['control_freq']} | "
              f"{r['enrichment']}x | {r['p_value']:.1e} |")
        a("")

    # ---- cross-cohort / external ----
    xc = ev_report.get("cross_cohort", {})
    if xc and "note" not in xc:
        a("## Cross-cohort validation (pooled dataset, logistic reference model)\n")
        a("_Uses a fresh logistic-regression fit per direction as a stable generalization "
          "check -- not the primary tuned model above._\n")
        a("| Train -> Test | n train | n test | AUC | PR-AUC |")
        a("|---|---|---|---|---|")
        for k, v in xc.items():
            pretty = k.replace("__", " -> ").replace("train_", "").replace("test_", "")
            a(f"| {pretty} | {v['n_train']} | {v['n_test']} | {_fmt(v['auc'])} | {_fmt(v['pr_auc'])} |")
        a("")

    for _, r in ext_reports:
        holdout = "+".join(r.get("validated_on", []))
        trained = "+".join(r.get("trained_on", []))
        a(f"## External validation: trained on {trained} -> tested on {holdout}\n")
        a(_by_ancestry_table(r["by_ancestry"]))
        a("")
        img = f"calibration_{holdout}.png"
        if os.path.exists(os.path.join(out, img)):
            a(f"![calibration {holdout}]({img})\n")

    # ---- matching design ----
    a("## Matching design\n")
    a(f"- Method: propensity score, {cfg['matching']['k_controls']}:1 nearest-neighbour, "
      f"caliper = {_fmt(match_meta.get('caliper', 0))} logit-SD units")
    a(f"- Covariates balanced on: {', '.join(cfg['matching']['ps_covariates'])}")
    a(f"- Matched {match_meta.get('n_cases_matched')}/{match_meta.get('n_cases_total')} cases "
      f"to {match_meta.get('n_controls_matched')} controls; unmatched subjects retained "
      f"in the full-cohort comparison model above")
    a("")

    a("---")
    a("_Generated by `scripts/report.py` from evaluate.py / external_validate.py outputs — "
      "no numbers are recomputed here. Convert to slides with, e.g., `pandoc report.md -o report.pdf`._")

    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    LOG.info("wrote %s", path)


if __name__ == "__main__":
    main()
