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

  python scripts/audit_feature_maps.py --config config/crc.yaml
  python scripts/audit_feature_maps.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, codes, data, util  # noqa: E402

LOG = util.get_logger("audit_feature_maps")


def _emit(lines, msg):
    LOG.info(msg)
    lines.append(msg)


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

    all_labels = {lab.lower() for labs in label_map.values() for lab in labs}
    present = set(vals.str.lower().unique())
    missing = sorted(all_labels - present)
    if missing:
        _emit(lines, f"  ! configured {table_name} labels with ZERO real matches "
                     f"(likely wrong guess, or genuinely absent in this cohort): {missing}")
    else:
        _emit(lines, f"  all configured {table_name} labels found at least once.")

    if unit_col and unit_col in df.columns:
        for key, labels in label_map.items():
            labs_l = [s.lower() for s in labels]
            sub = df[vals.str.lower().isin(labs_l)]
            if sub.empty:
                continue
            uc = sub[unit_col].astype(str).str.strip().value_counts()
            _emit(lines, f"  '{key}' {labels} real '{unit_col}' distribution: {uc.to_dict()}")


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
    args = cli.base_parser("Audit feature_maps guesses against real interim data").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    fm = cfg["feature_maps"]

    pheno = pd.read_csv(os.path.join(out, "phenotype.csv"))
    cohorts = pheno["cohort"].unique().tolist()
    lines = [f"# feature_maps audit -- {cfg['cancer']}"]

    for cohort in cohorts:
        LOG.info("\n%s\ncohort %s\n%s", "=" * 70, cohort, "=" * 70)
        lines.append(f"\n## cohort {cohort}\n")
        n_subjects = int((pheno["cohort"] == cohort).sum())

        lines.append("### vitals")
        audit_category_column(lines, data.load_tidy(out, cohort, "vitals"),
                              "name", "unit", fm["vital_signs"], "vitals")

        lines.append("\n### labs")
        audit_category_column(lines, data.load_tidy(out, cohort, "labs"),
                              "analyte", "units", fm["lab_analytes"], "labs")

        dx_tables = [data.load_tidy(out, cohort, t)
                    for t in ("enc_diagnosis", "problem_list", "medical_hx")]
        dx_tables = [d for d in dx_tables if d is not None and "icd_code" in d.columns]
        dxw = pd.concat(dx_tables, ignore_index=True) if dx_tables else None
        lines.append("\n### ICD-based flags (comorbidity + symptoms)")
        audit_icd(lines, dxw, {**fm["comorbidity_icd"], **fm["symptom_icd"]}, n_subjects)

        lines.append("\n### medications")
        meds = data.load_tidy(out, cohort, "meds")
        if meds is not None and "name" in meds.columns:
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
            configured = {v.lower() for v in fhcfg.get("positive_values", [])}
            if not (configured & set(vc.index)):
                _emit(lines, "  ! none of the configured positive_values appear in the real data "
                             "-- family_hx_crc is likely always 0")
        else:
            _emit(lines, f"  ('{flag_col}' not in questionnaire; falls back to "
                         f"Family_History free text matching on {fhcfg.get('crc_terms')})")

    path = os.path.join(out, "feature_audit.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    LOG.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
