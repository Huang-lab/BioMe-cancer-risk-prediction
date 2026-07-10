#!/usr/bin/env python3
"""Stage 5 — assemble the modeling datasets.

Joins match.csv (label + matched sets + PS + PCs + age/sex/cohort/ancestry) with
features.csv and carriers_wide.csv, then writes:
  - dataset_full.csv      (all subjects; for the class-weighted comparison model)
  - dataset_matched.csv   (matched subjects only; the PRIMARY model)
  - feature_spec.json     (numeric/categorical feature columns; audit + meta excluded)

  python scripts/build_dataset.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, util  # noqa: E402

LOG = util.get_logger("build_dataset")

CATEGORICAL = ["sex", "race", "ethnicity", "marital_status", "religion", "country",
               "smoking_status", "alcohol_use", "country_of_birth", "language_preference",
               "cohort"]
META = ["ehr_id", "sample_id", "group", "is_case", "index_date", "birth_date",
        "ancestry_group", "matched", "matched_set", "propensity"]


def main():
    args = cli.base_parser("Assemble modeling datasets").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)

    match = pd.read_csv(os.path.join(out, "match.csv"))
    features = pd.read_csv(os.path.join(out, "features.csv"))
    carriers = pd.read_csv(os.path.join(out, "carriers_wide.csv"))
    meta = util.load_json(os.path.join(out, "phenotype_meta.json"))
    pc_cols = meta.get("pc_cols") or []

    df = match.merge(features, on="ehr_id", how="left").merge(carriers, on="ehr_id", how="left")

    # audit-only features are kept in the table but excluded from the model
    audit = []
    for key in cfg["features"].get("audit_only", []):
        audit += [c for c in df.columns if c == key or c.startswith(f"{key}_")]

    excluded = set(META) | set(audit)
    feature_cols = [c for c in df.columns if c not in excluded]
    categorical = [c for c in feature_cols if c in CATEGORICAL]
    numeric = [c for c in feature_cols if c not in categorical]

    spec = {
        "label": "is_case",
        "group_col": "matched_set",
        "cohort_col": "cohort",
        "ancestry_col": "ancestry_group",
        "pc_cols": pc_cols,
        "feature_cols": feature_cols,
        "numeric": numeric,
        "categorical": categorical,
        "audit_only": audit,
    }

    df_full = df.copy()
    df_matched = df[df["matched"] == True].copy()  # noqa: E712

    df_full.to_csv(os.path.join(out, "dataset_full.csv"), index=False)
    df_matched.to_csv(os.path.join(out, "dataset_matched.csv"), index=False)
    util.save_json(spec, os.path.join(out, "feature_spec.json"))

    LOG.info("full: %d subjects (%d cases) | matched: %d subjects (%d cases)",
             len(df_full), int(df_full["is_case"].sum()),
             len(df_matched), int(df_matched["is_case"].sum()))
    LOG.info("features: %d (%d numeric, %d categorical); audit-only excluded: %s",
             len(feature_cols), len(numeric), len(categorical), audit)
    LOG.info("wrote dataset_full.csv, dataset_matched.csv, feature_spec.json -> %s", out)


if __name__ == "__main__":
    main()
