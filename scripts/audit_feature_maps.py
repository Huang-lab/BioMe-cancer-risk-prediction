#!/usr/bin/env python3
"""Data QA — audit feature_maps value-string guesses against real interim tables.

Every category string in config/*.yaml's feature_maps (vital/lab name strings,
ICD prefixes, medication text, family-history flags, unit assumptions) was
written as a best guess against schema/*.header COLUMN NAMES -- never against
real column VALUES, because this container is never allowed to see real data
(CLAUDE.md). This script reads the interim tables preprocess.py already wrote
and checks the guesses against them:

  - what the real top values are for each column we filter/match on
  - which configured label strings have ZERO matches in the real data
    (probably a wrong guess)
  - what units actually show up for matched vital/lab rows (the vitals_units
    config assumption is a single global guess; this shows the real per-row
    distribution)
  - rough case-rate sanity checks + prefix-collision checks for ICD flags
  - real values for the family-history flag column

Read-only: it never touches dataset/model files and can be re-run as often as
you like before deciding whether anything needs to change.

CAVEAT this script's default mode does NOT catch on its own: preprocess.py
pre-filters vitals/labs rows to config's *already-guessed* name/analyte
values INSIDE the chunked raw-file reader (to keep multi-GB files tractable
-- see build_row_filters() in preprocess.py), so the interim CSVs this script
reads by default can only ever contain rows that already matched a guess.
A vital/lab name that was never listed in feature_maps at all is invisible
here, not just "zero matches" -- it was dropped before interim was written.
Pass --raw to also re-read the RAW files directly (bypassing that filter,
same chunked reader preprocess.py uses) and list real values NOT covered by
any configured label. That pass touches the same multi-GB files preprocess.py
does, so it can take a few minutes on real Order_results/Vitals files --
default mode is instant and fine for everything except "did we miss a whole
vital/lab name."

  python scripts/audit_feature_maps.py --config config/crc.yaml
  python scripts/audit_feature_maps.py --config config/crc.yaml --raw
  python scripts/audit_feature_maps.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, codes, config as cfgmod, data, io, util  # noqa: E402
import preprocess as pp  # sibling stage script; reuse its raw-file plumbing  # noqa: E402

LOG = util.get_logger("audit_feature_maps")


def _emit(lines, msg):
    LOG.info(msg)
    lines.append(msg)


def _labels_of(entry):
    """A feature_maps label-map value is either a flat list (vital_signs) or
    the newer {labels, units} dict form (lab_analytes) -- extract the label
    list either way. Without this, iterating a dict value directly yields its
    KEYS ("labels", "units"), not the analyte names (the exact bug this
    mirrors in preprocess.py's build_row_filters)."""
    return entry.get("labels", []) if isinstance(entry, dict) else entry


def audit_category_column(lines, df, col, unit_col, label_map, table_name, top_n=20):
    """Long-format table filtered on a name/analyte-like column: real top values,
    configured labels with zero real matches, and (if a unit column exists) the
    real unit distribution for each matched label."""
    if df is None or df.empty or col not in df.columns:
        _emit(lines, f"  ({table_name}: no data / no '{col}' column)")
        return
    vals = df[col].astype(str).str.strip()
    _emit(lines, f"  top {top_n} real '{col}' values in {table_name} (n={len(df)} rows):")
    for v, c in vals.value_counts().head(top_n).items():
        _emit(lines, f"    {c:>8}  {v}")

    all_labels = {lab.lower() for labs in label_map.values() for lab in _labels_of(labs)}
    present = set(vals.str.lower().unique())
    missing = sorted(all_labels - present)
    if missing:
        _emit(lines, f"  ! configured {table_name} labels with ZERO real matches "
                     f"(likely wrong guess, or genuinely absent in this cohort): {missing}")
    else:
        _emit(lines, f"  all configured {table_name} labels found at least once.")

    if unit_col and unit_col in df.columns:
        for key, labels in label_map.items():
            labs_l = [s.lower() for s in _labels_of(labels)]
            sub = df[vals.str.lower().isin(labs_l)]
            if sub.empty:
                continue
            uc = sub[unit_col].astype(str).str.strip().value_counts()
            _emit(lines, f"  '{key}' {labels} real '{unit_col}' distribution: {uc.to_dict()}")


def raw_uncovered_values(lines, cfg, cohort_cfg, table_key, col, label_map, top_n=30):
    """Re-read the RAW file directly (bypassing preprocess.py's own value-guess
    row filter) so we can see category strings that were never guessed at all --
    the interim CSVs only ever contain rows that already matched a guess."""
    ehr_dir = cfgmod.resolve_path(cfg, cohort_cfg["ehr_dir"])
    if not os.path.isdir(ehr_dir):
        _emit(lines, f"  (raw ehr_dir not found: {ehr_dir})")
        return
    manifest = io.load_header_manifest(ehr_dir)
    file_prefix = cohort_cfg.get("file_prefix", "")
    spec = cfg["ehr_tables"][table_key]
    header_cols = manifest.get(file_prefix + spec["file"]) or manifest.get(spec["file"])
    cohort_ids = pp.cohort_patient_ids(cfg, cohort_cfg)  # patient scope only, no value filter
    row_filter = {"ehr_id": cohort_ids} if cohort_ids else None
    lc_override = None if cohort_cfg.get("has_cohort_tag", True) else 0

    df = io.read_ehr_table(cfg, ehr_dir, table_key, required=False,
                           id_override=cohort_cfg.get("clinical_id_col"),
                           header_cols=header_cols, row_filter=row_filter,
                           file_prefix=file_prefix, leading_cols_override=lc_override)
    if df is None or df.empty or col not in df.columns:
        _emit(lines, f"  (raw {table_key}: no data / no '{col}' column)")
        return

    vals = df[col].astype(str).str.strip()
    vc = vals.value_counts()
    all_labels = {lab.lower() for labs in label_map.values() for lab in _labels_of(labs)}
    uncovered = vc[~vc.index.str.lower().isin(all_labels)]
    _emit(lines, f"  RAW {table_key} scan (bypasses the config-guess pre-filter): "
                 f"{len(df)} rows, {len(vc)} distinct '{col}' values")
    if uncovered.empty:
        _emit(lines, f"  every real {table_key} '{col}' value is already covered by feature_maps.")
    else:
        _emit(lines, f"  ! top {top_n} REAL {table_key} '{col}' values NOT in any feature_maps "
                     f"label (possible missed signal or unmapped synonym):")
        for v, c in uncovered.head(top_n).items():
            _emit(lines, f"    {c:>8}  {v}")


def audit_static_columns(lines, out, cohort, cfg, top_n=8):
    """For every configured cols: mapping in ehr_tables, report fill-rate + top
    values in the interim CSV, so an empty column (raw name never matched, or
    real column is genuinely empty) shows up loudly instead of silently landing
    in the model as always-NaN."""
    for table_key, spec in cfg["ehr_tables"].items():
        cols_map = spec.get("cols") or {}
        if not cols_map:
            continue
        df = data.load_tidy(out, cohort, table_key)
        if df is None or df.empty:
            _emit(lines, f"  {table_key}: interim CSV missing or empty")
            continue
        n_rows = len(df)
        n_subj = df["ehr_id"].nunique() if "ehr_id" in df.columns else n_rows
        _emit(lines, f"  {table_key}: {n_rows} rows, {n_subj} subjects "
                     f"({list(df.columns)})")
        for canon in cols_map:
            if canon not in df.columns:
                _emit(lines, f"    ! '{canon}' MISSING from interim -- raw column "
                             f"{cols_map[canon]!r} not in the file we read")
                continue
            s = df[canon].astype(str).str.strip().replace({"nan": "", "None": ""})
            nonnull = (s != "").sum()
            fill = nonnull / n_rows if n_rows else 0
            if nonnull == 0:
                _emit(lines, f"    ! '{canon}' (raw {cols_map[canon]!r}) 100% EMPTY "
                             f"-- feature will always be NaN in downstream")
                continue
            top = s[s != ""].value_counts().head(top_n)
            top_str = ", ".join(f"{v!r}={c}" for v, c in top.items())
            _emit(lines, f"    {canon} (raw {cols_map[canon]!r}): fill={fill:.1%} "
                         f"({nonnull}/{n_rows}); top: {top_str}")


def audit_icd(lines, dxw, prefixes_by_key, n_subjects):
    if dxw is None or dxw.empty:
        _emit(lines, "  (no diagnosis rows to check)")
        return
    for key, prefixes in prefixes_by_key.items():
        hit = dxw[dxw["icd_code"].apply(lambda c: codes.matches_any_prefix(c, prefixes))]
        n_pt = hit["ehr_id"].nunique()
        rate = n_pt / n_subjects if n_subjects else 0
        _emit(lines, f"  {key} (prefixes={prefixes}): {n_pt}/{n_subjects} subjects "
                     f"({rate:.1%}) have >=1 matching code, {len(hit)} rows")
    # flag prefix overlaps across keys -- one code could double-count two flags
    normed = [(key, codes.normalize_icd(p)) for key, ps in prefixes_by_key.items() for p in ps]
    for i, (k1, p1) in enumerate(normed):
        for k2, p2 in normed[i + 1:]:
            if k1 != k2 and (p1.startswith(p2) or p2.startswith(p1)):
                _emit(lines, f"  ! prefix overlap: '{k1}' and '{k2}' ({p1!r}/{p2!r}) "
                             f"can both fire on the same code")


def main():
    ap = cli.base_parser("Audit feature_maps guesses against real interim data")
    ap.add_argument("--raw", action="store_true",
                    help="also re-scan the RAW vitals/labs files (bypassing preprocess.py's "
                         "config-guess row filter) for name/analyte values never mapped at "
                         "all -- slower (touches the same multi-GB files preprocess.py does)")
    args = ap.parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    fm = cfg["feature_maps"]
    cohort_cfgs = {c["name"]: c for c in cfg["cohorts"]}

    pheno = pd.read_csv(os.path.join(out, "phenotype.csv"))
    cohorts = pheno["cohort"].unique().tolist()
    lines = [f"# feature_maps audit -- {cfg['cancer']}"]

    for cohort in cohorts:
        LOG.info("\n%s\ncohort %s\n%s", "=" * 70, cohort, "=" * 70)
        lines.append(f"\n## cohort {cohort}\n")
        n_subjects = int((pheno["cohort"] == cohort).sum())
        # interim tables cover every roster case+control BEFORE phenotype.py drops
        # subjects with no index_date -- restrict rate/count denominators below to
        # just the subjects that actually survive into the model, or "N/n_subjects"
        # can exceed 100% (numerator from the larger interim population).
        pheno_ids = set(pheno.loc[pheno["cohort"] == cohort, "ehr_id"].astype(str))

        lines.append("### per-column fill-rate across every configured feature")
        audit_static_columns(lines, out, cohort, cfg)

        lines.append("\n### vitals")
        audit_category_column(lines, data.load_tidy(out, cohort, "vitals"),
                              "name", "unit", fm["vital_signs"], "vitals")
        if args.raw and cohort in cohort_cfgs:
            raw_uncovered_values(lines, cfg, cohort_cfgs[cohort], "vitals", "name", fm["vital_signs"])

        lines.append("\n### labs")
        audit_category_column(lines, data.load_tidy(out, cohort, "labs"),
                              "analyte", "units", fm["lab_analytes"], "labs")
        if args.raw and cohort in cohort_cfgs:
            raw_uncovered_values(lines, cfg, cohort_cfgs[cohort], "labs", "analyte", fm["lab_analytes"])

        dx_tables = [data.load_tidy(out, cohort, t)
                    for t in ("enc_diagnosis", "problem_list", "medical_hx")]
        dx_tables = [d for d in dx_tables if d is not None and "icd_code" in d.columns]
        dxw = pd.concat(dx_tables, ignore_index=True) if dx_tables else None
        if dxw is not None:
            dxw = dxw[dxw["ehr_id"].astype(str).isin(pheno_ids)]
        lines.append("\n### ICD-based flags (comorbidity + symptoms)")
        audit_icd(lines, dxw, {**fm["comorbidity_icd"], **fm["symptom_icd"]}, n_subjects)

        lines.append("\n### medications")
        meds = data.load_tidy(out, cohort, "meds")
        if meds is not None and "name" in meds.columns:
            meds = meds[meds["ehr_id"].astype(str).isin(pheno_ids)]
            _emit(lines, "  top 15 real medication name values:")
            for v, c in meds["name"].astype(str).value_counts().head(15).items():
                _emit(lines, f"    {c:>8}  {v}")
            for key, names in fm["medications"].items():
                pat = "|".join(names)
                hit = meds[meds["name"].astype(str).str.contains(pat, case=False, na=False)]
                _emit(lines, f"  '{key}' {names}: {hit['ehr_id'].nunique()}/{n_subjects} "
                             f"subjects matched")
        else:
            _emit(lines, "  (no meds table / no 'name' column)")

        lines.append("\n### family history flag")
        quest = data.load_tidy(out, cohort, "questionnaire")
        fhcfg = fm["family_history"]
        flag_col = fhcfg.get("questionnaire_flag")
        if quest is not None and flag_col in quest.columns:
            vc = quest[flag_col].astype(str).str.lower().value_counts()
            _emit(lines, f"  '{flag_col}' real value counts: {vc.to_dict()}")
            you_hits = quest[flag_col].astype(str).str.contains(r"\bYou\b", na=False, regex=True).sum()
            if you_hits == 0:
                _emit(lines, "  ! no real value contains whole-word 'You' -- "
                             "family_hx_crc would be always 0")
            else:
                _emit(lines, f"  {you_hits} rows contain whole-word 'You' -> family_hx_crc=1")
        else:
            _emit(lines, f"  ('{flag_col}' not in questionnaire; falls back to "
                         f"Family_History free text matching on {fhcfg.get('crc_terms')})")

    path = os.path.join(out, "feature_audit.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    LOG.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
