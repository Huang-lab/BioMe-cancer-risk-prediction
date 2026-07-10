#!/usr/bin/env python3
"""Stage 4 — genomics carrier join.

Reads each cohort's AlphaMissense-calibrated all_carriers.tsv, selects qualifying
variants by the configured evidence sources, and builds per-gene <gene>_any flags
plus the named aggregates (lynch_any, crc_panel_any). Joins to subjects via the
roster's sample_id<->ehr_id crosswalk. Absence => non-carrier (0).

  python scripts/genomics.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, config as cfgmod, io, util  # noqa: E402

LOG = util.get_logger("genomics")


def qualifying(carriers, evidence_sources, mode):
    cols = [s for s in evidence_sources if s in carriers.columns]
    if not cols:
        return carriers.iloc[0:0]
    mask = carriers[cols].any(axis=1) if mode == "any_of" else carriers[cols].all(axis=1)
    return carriers[mask]


def main():
    args = cli.base_parser("Genomics carrier join").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    g = cfg["genomics"]

    pheno = pd.read_csv(os.path.join(out, "phenotype.csv"))[["ehr_id", "sample_id"]]

    frames = []
    for cohort in cfg["cohorts"]:
        path = cfgmod.resolve_path(cfg, cohort["carrier_flags"])
        if not os.path.exists(path):
            LOG.warning("cohort %s: carrier file %s missing, skipping", cohort["name"], path)
            continue
        frames.append(io.read_carriers(cfg, path))
    if not frames:
        LOG.error("no carrier files found")
        sys.exit(1)
    carriers = pd.concat(frames, ignore_index=True)

    hits = qualifying(carriers, g.get("evidence_sources", ["clinvar", "alphamissense"]),
                      g.get("carrier_evidence", "any_of"))
    LOG.info("qualifying carrier-variants: %d (of %d rows) by evidence %s",
             len(hits), len(carriers), g.get("evidence_sources"))

    # per-gene flags on the sample_id level
    panel = g["panel"]
    wide = pd.DataFrame({"sample_id": pheno["sample_id"].unique()})
    carrier_by_gene = {gene: set(hits[hits["gene"] == gene]["sample_id"]) for gene in panel}
    for gene in panel:
        wide[f"{gene}_any"] = wide["sample_id"].isin(carrier_by_gene[gene]).astype(int)

    # named aggregates
    for agg, members in g.get("aggregate_members", {}).items():
        member_cols = [f"{gene}_any" for gene in members if f"{gene}_any" in wide.columns]
        wide[agg] = wide[member_cols].max(axis=1) if member_cols else 0

    # join to ehr_id; non-carriers -> 0
    merged = pheno.merge(wide, on="sample_id", how="left")
    flag_cols = [c for c in merged.columns if c.endswith("_any")]
    merged[flag_cols] = merged[flag_cols].fillna(0).astype(int)
    merged = merged.drop(columns=["sample_id"])

    merged.to_csv(os.path.join(out, "carriers_wide.csv"), index=False)
    n_carrier = int((merged[g["extra_aggregate"]] > 0).sum()) if g["extra_aggregate"] in merged else 0
    LOG.info("wrote carriers_wide.csv: %d subjects, %d panel-carriers, %d flag columns -> %s",
             len(merged), n_carrier, len(flag_cols), out)


if __name__ == "__main__":
    main()
