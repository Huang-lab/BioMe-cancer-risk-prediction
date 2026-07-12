#!/usr/bin/env python3
"""Stage 0c — parse/clean the RAW clinical files into tidy per-cohort tables.

Reads every configured EHR table (raw exports, mixed delimiters), applies
date/numeric cleaning, and persists canonical-column CSVs under
``<outdir>/interim/<cohort>/<table>.csv`` for the downstream stages.

  python scripts/preprocess.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, config as cfgmod, data, io, util  # noqa: E402

LOG = util.get_logger("preprocess")


def main():
    args = cli.base_parser("Preprocess raw clinical files").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)

    for cohort in cfg["cohorts"]:
        name = cohort["name"]
        ehr_dir = cfgmod.resolve_path(cfg, cohort["ehr_dir"])
        if not os.path.isdir(ehr_dir):
            LOG.warning("cohort %s: ehr_dir %s not found, skipping", name, ehr_dir)
            continue
        id_override = cohort.get("clinical_id_col")
        manifest = io.load_header_manifest(ehr_dir)   # {} if files carry inline headers
        if manifest:
            LOG.info("cohort %s: using Header_File.txt manifest (%d tables, headerless data)",
                     name, len(manifest))
        n_tables = 0
        for table_key in cfg["ehr_tables"]:
            fname = cfg["ehr_tables"][table_key]["file"]
            df = io.read_ehr_table(cfg, ehr_dir, table_key, required=False,
                                   id_override=id_override,
                                   header_cols=manifest.get(fname))
            if df is None:
                LOG.warning("  %s/%s: file absent, skipped", name, table_key)
                continue
            df = data.clean_table(df)
            data.save_tidy(df, out, name, table_key)
            n_tables += 1
        LOG.info("cohort %s: wrote %d tidy tables -> %s", name, n_tables,
                 data.interim_dir(out, name))


if __name__ == "__main__":
    main()
