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

# Values ehr_id must NOT contain — catches leading-cohort-column-shift bugs,
# stray inline-header rows, and file-mixup mistakes. Keep in sync with io.py.
COHORT_TAGS = {
    "Regeneron", "REGENERON", "regeneron",
    "Sema4", "SEMA4", "sema4",
    "cohortI", "cohortII", "CohortI", "CohortII",
    "sem_id", "rgnid", "RGNID",
}


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
        file_prefix = cohort.get("file_prefix", "")
        has_cohort_tag = cohort.get("has_cohort_tag", True)   # legacy default = Regen
        manifest = io.load_header_manifest(ehr_dir)   # {} if files carry inline headers
        if manifest:
            LOG.info("cohort %s: using Header_File.txt manifest (%d tables, headerless data%s%s)",
                     name, len(manifest),
                     f", file_prefix={file_prefix!r}" if file_prefix else "",
                     "" if has_cohort_tag else ", NO leading-cohort-column shift")
        # manifest keys are the file names as they appear in Header_File.txt.
        # For Sema4 those already start with 'Sema4_'; for Regen they don't.
        row_filters = build_row_filters(cfg)
        if row_filters:
            LOG.info("cohort %s: applying per-chunk row filters on %s",
                     name, list(row_filters))
        n_tables = 0
        for table_key in cfg["ehr_tables"]:
            spec = cfg["ehr_tables"][table_key]
            manifest_key = file_prefix + spec["file"]        # 'Sema4_Demographics.txt'
            header_cols = manifest.get(manifest_key) or manifest.get(spec["file"])
            # If the cohort has no leading cohort tag, force leading_cols=0 for
            # every table regardless of the shared config (Regen sets it to 1 on 8).
            lc_override = None if has_cohort_tag else 0
            df = io.read_ehr_table(cfg, ehr_dir, table_key, required=False,
                                   id_override=id_override,
                                   header_cols=header_cols,
                                   row_filter=row_filters.get(table_key),
                                   file_prefix=file_prefix,
                                   leading_cols_override=lc_override)
            if df is None:
                LOG.warning("  %s/%s: file absent, skipped", name, table_key)
                continue
            df = data.clean_table(df)
            # Sanity check: catches leading-column-shift, stray inline-header, and
            # file-mixup mistakes BEFORE they silently corrupt downstream joins.
            if len(df) and "ehr_id" in df.columns:
                tag_hit_rate = df["ehr_id"].astype(str).isin(COHORT_TAGS).mean()
                if tag_hit_rate > 0.01:
                    top = df["ehr_id"].astype(str).value_counts().head(3).to_dict()
                    raise RuntimeError(
                        f"[{name}/{table_key}] {tag_hit_rate:.0%} of ehr_id values look "
                        f"like a cohort tag / header keyword {top}. "
                        f"Toggle cohort.has_cohort_tag or ehr_tables.{table_key}.leading_cols.")
            data.save_tidy(df, out, name, table_key)
            LOG.info("  %s: %d rows -> %s", table_key, len(df), f"{table_key}.csv")
            n_tables += 1
        LOG.info("cohort %s: wrote %d tidy tables -> %s", name, n_tables,
                 data.interim_dir(out, name))


if __name__ == "__main__":
    main()
