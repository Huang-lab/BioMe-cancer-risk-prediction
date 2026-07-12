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


def build_row_filters(cfg: dict) -> dict[str, dict[str, set]]:
    """For huge long-format tables, restrict rows to the target analyte/vital/topic
    values. Applied INSIDE the chunked reader so multi-GB files stay tractable.
    Keyed by canonical (post-rename) column names.  Case-insensitive match."""
    fm = cfg.get("feature_maps", {})
    filters: dict[str, dict[str, set]] = {}

    def _flat_lower(mapping):
        vals = set()
        for lst in mapping.values():
            for v in lst:
                vals.add(str(v).strip().lower())
        return vals

    if "labs" in cfg["ehr_tables"] and "lab_analytes" in fm:
        filters["labs"] = {"analyte": _flat_lower(fm["lab_analytes"])}
    if "vitals" in cfg["ehr_tables"] and "vital_signs" in fm:
        filters["vitals"] = {"name": _flat_lower(fm["vital_signs"])}
    hm = cfg["ehr_tables"].get("health_maintenance", {})
    if hm.get("colonoscopy_topics"):
        filters["health_maintenance"] = {
            "topic": {t.strip().lower() for t in hm["colonoscopy_topics"]}}
    return filters


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
        row_filters = build_row_filters(cfg)
        if row_filters:
            LOG.info("cohort %s: applying per-chunk row filters on %s",
                     name, list(row_filters))
        n_tables = 0
        for table_key in cfg["ehr_tables"]:
            fname = cfg["ehr_tables"][table_key]["file"]
            df = io.read_ehr_table(cfg, ehr_dir, table_key, required=False,
                                   id_override=id_override,
                                   header_cols=manifest.get(fname),
                                   row_filter=row_filters.get(table_key))
            if df is None:
                LOG.warning("  %s/%s: file absent, skipped", name, table_key)
                continue
            df = data.clean_table(df)
            data.save_tidy(df, out, name, table_key)
            LOG.info("  %s: %d rows -> %s", table_key, len(df), f"{table_key}.csv")
            n_tables += 1
        LOG.info("cohort %s: wrote %d tidy tables -> %s", name, n_tables,
                 data.interim_dir(out, name))


if __name__ == "__main__":
    main()
