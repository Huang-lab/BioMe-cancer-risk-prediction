#!/usr/bin/env python3
"""Stage 2 — propensity-score matching (primary design).

Fits a logistic PS on the configured covariates (age, sex, genetic PCs, cohort),
then greedy nearest-neighbour k:1 caliper matching on the PS logit, without
replacement. Emits match.csv keeping the FULL cohort, with matched flags +
matched-set ids so downstream can build both the matched and full datasets.

  python scripts/match.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, util  # noqa: E402

LOG = util.get_logger("match")


def build_ps_matrix(pheno, covariates, pc_cols):
    """Assemble the PS design matrix from the configured covariates."""
    parts = []
    for cov in covariates:
        if cov == "genetic_pcs":
            parts.append(pheno[[c for c in pc_cols if c in pheno.columns]])
        elif cov == "age_at_index":
            parts.append(pheno[["age_at_index"]])
        elif cov in ("sex", "cohort"):
            parts.append(pd.get_dummies(pheno[cov].astype("string"), prefix=cov, dummy_na=True))
        elif cov in pheno.columns:
            parts.append(pheno[[cov]])
    X = pd.concat(parts, axis=1).apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean(numeric_only=True)).fillna(0.0)


def greedy_match(pheno, logit, k, caliper, seed):
    """Greedy NN k:1 caliper matching on the PS logit, without replacement."""
    rng = np.random.default_rng(seed)
    is_case = pheno["is_case"].to_numpy()
    cohort = pheno["cohort"].to_numpy()
    case_ix = np.where(is_case)[0]
    rng.shuffle(case_ix)

    matched_set = np.full(len(pheno), -1, dtype=int)
    available = {i for i in np.where(~is_case)[0]}
    set_id = 0
    for ci in case_ix:
        # candidate controls from the same cohort, within caliper, still available
        cand = [j for j in available
                if cohort[j] == cohort[ci] and abs(logit[j] - logit[ci]) <= caliper]
        if not cand:
            continue
        cand.sort(key=lambda j: abs(logit[j] - logit[ci]))
        chosen = cand[:k]
        if not chosen:
            continue
        matched_set[ci] = set_id
        for j in chosen:
            matched_set[j] = set_id
            available.discard(j)
        set_id += 1
    return matched_set


def main():
    ap = cli.base_parser("Propensity-score matching")
    args = ap.parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)

    pheno = pd.read_csv(os.path.join(out, "phenotype.csv"))
    meta = util.load_json(os.path.join(out, "phenotype_meta.json"))
    pc_cols = meta.get("pc_cols") or []

    m = cfg["matching"]
    if m["method"] != "propensity":
        LOG.warning("matching.method=%s but this stage implements propensity", m["method"])

    X = build_ps_matrix(pheno, m["ps_covariates"], pc_cols)
    y = pheno["is_case"].astype(int).to_numpy()

    ps_model = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=1000))])
    ps_model.fit(X, y)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    logit = np.log(ps / (1 - ps))
    caliper = m["caliper_sd"] * np.std(logit)

    matched_set = greedy_match(pheno, logit, m["k_controls"], caliper,
                               cfg.get("random_state", 42))

    pheno["propensity"] = ps
    pheno["matched_set"] = matched_set
    pheno["matched"] = matched_set >= 0

    n_case_m = int(pheno[(pheno.is_case) & pheno.matched].shape[0])
    n_ctrl_m = int(pheno[(~pheno.is_case) & pheno.matched].shape[0])
    n_case_tot = int(pheno.is_case.sum())
    LOG.info("matched %d/%d cases to %d controls (k=%d, caliper=%.3f logit-SD units)",
             n_case_m, n_case_tot, n_ctrl_m, m["k_controls"], m["caliper_sd"])
    if m.get("also_emit_unmatched", True):
        LOG.info("full cohort retained for class-weighted comparison (%d subjects)", len(pheno))

    pheno.to_csv(os.path.join(out, "match.csv"), index=False)
    util.save_json({"n_cases_total": n_case_tot, "n_cases_matched": n_case_m,
                    "n_controls_matched": n_ctrl_m, "caliper": float(caliper)},
                   os.path.join(out, "match_meta.json"))
    LOG.info("wrote match.csv -> %s", out)


if __name__ == "__main__":
    main()
