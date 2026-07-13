#!/usr/bin/env python3
"""Stage 1 — phenotyping. The roster is the spine.

Assigns case (CRC)/control from the roster group, derives index_date
(earliest qualifying CRC dx for cases; last encounter for controls), computes
age_at_index, and carries ancestry group + genetic PCs. Writes phenotype.csv.

  python scripts/phenotype.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, codes, config as cfgmod, data, io, util  # noqa: E402

LOG = util.get_logger("phenotype")


def _earliest_case_dx(cfg, out, cohort):
    """min qualifying CRC dx date per ehr_id across enc_diagnosis + problem_list."""
    ph = cfg["phenotype"]
    icd10, icd9 = ph["case_icd10_prefix"], ph["case_icd9_prefix"]
    frames = []
    for tbl in ("enc_diagnosis", "problem_list"):
        df = data.load_tidy(out, cohort, tbl)
        if df is None or "icd_code" not in df or "date" not in df:
            continue
        hit = df[df["icd_code"].apply(lambda c: codes.is_case_code(c, icd10, icd9))]
        frames.append(hit[["ehr_id", "date"]])
    if not frames:
        return pd.Series(dtype="datetime64[ns]")
    allhits = pd.concat(frames, ignore_index=True)
    return allhits.groupby("ehr_id")["date"].min()


def _last_encounter(cfg, out, cohort):
    """max observed date per ehr_id across date-bearing tables (controls' index)."""
    frames = []
    for tbl in ("enc_diagnosis", "problem_list", "vitals", "labs", "meds",
                "surgical_hx", "medical_hx"):
        df = data.load_tidy(out, cohort, tbl)
        if df is not None and "date" in df:
            frames.append(df[["ehr_id", "date"]])
    if not frames:
        return pd.Series(dtype="datetime64[ns]")
    alld = pd.concat(frames, ignore_index=True)
    return alld.groupby("ehr_id")["date"].max()


def main():
    args = cli.base_parser("Phenotyping").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)

    case_labels = set(cfgmod.resolve(cfg["roster"]["case_labels"]))
    control_labels = set(cfgmod.resolve_control_labels(cfg))
    keep = case_labels | control_labels

    all_rows = []
    pc_cols = None
    for cohort in cfg["cohorts"]:
        name = cohort["name"]
        roster_path = cfgmod.resolve_path(cfg, cohort["roster"])
        if not os.path.exists(roster_path):
            LOG.warning("cohort %s: roster %s missing, skipping", name, roster_path)
            continue
        roster, pcs = io.read_roster(cfg, roster_path)
        pc_cols = pcs
        roster["cohort"] = name
        if cfg["roster"].get("drop_other_groups", True):
            roster = roster[roster["group"].isin(keep)].copy()
        roster["is_case"] = roster["group"].isin(case_labels)

        case_idx = _earliest_case_dx(cfg, out, name)
        last_enc = _last_encounter(cfg, out, name)
        roster["index_date"] = roster.apply(
            lambda r: case_idx.get(r["ehr_id"]) if r["is_case"] else last_enc.get(r["ehr_id"]),
            axis=1,
        )

        # sex: prefer the curated roster GENDER; else fall back to Demographics
        if "roster_sex" in roster.columns:
            roster["sex"] = roster["roster_sex"]
        else:
            demo = data.load_tidy(out, name, "demographics")
            if demo is not None and "sex" in demo.columns:
                roster = roster.merge(demo[["ehr_id", "sex"]].drop_duplicates("ehr_id"),
                                      on="ehr_id", how="left")
        if "sex" not in roster.columns:
            roster["sex"] = pd.NA

        # age_at_index: prefer the curated roster Age_at_diagnosis; else derive from
        # Questionnaire.YEAR_OF_BIRTH (no DOB in Demographics).
        roster["age_at_index"] = pd.NA
        if "roster_age" in roster.columns:
            roster["age_at_index"] = pd.to_numeric(roster["roster_age"], errors="coerce")
        need = roster["age_at_index"].isna()
        if need.any():
            quest = data.load_tidy(out, name, "questionnaire")
            if quest is not None and "year_of_birth" in quest.columns:
                yob = quest[["ehr_id", "year_of_birth"]].drop_duplicates("ehr_id")
                roster = roster.merge(yob, on="ehr_id", how="left")
                derived = roster["index_date"].dt.year - roster["year_of_birth"]
                roster["age_at_index"] = roster["age_at_index"].fillna(derived)

        n_before = len(roster)
        roster = roster[roster["index_date"].notna()].copy()
        dropped = n_before - len(roster)
        if dropped:
            LOG.warning("cohort %s: dropped %d subjects with no index_date "
                        "(cases without qualifying dx / controls without encounters)", name, dropped)
        all_rows.append(roster)
        LOG.info("cohort %s: %d cases, %d controls",
                 name, int(roster["is_case"].sum()), int((~roster["is_case"]).sum()))

    if not all_rows:
        LOG.error("no cohorts produced phenotype rows")
        sys.exit(1)

    pheno = pd.concat(all_rows, ignore_index=True)
    keep_cols = (["ehr_id", "sample_id", "cohort", "group", "is_case", "ancestry_group",
                  "index_date", "age_at_index", "sex"] + (pc_cols or []))
    pheno = pheno[[c for c in keep_cols if c in pheno.columns]]

    util.ensure_dir(out)
    pheno.to_csv(os.path.join(out, "phenotype.csv"), index=False)
    util.save_json({"pc_cols": pc_cols, "n": len(pheno),
                    "n_cases": int(pheno["is_case"].sum())},
                   os.path.join(out, "phenotype_meta.json"))
    LOG.info("wrote phenotype.csv: %d subjects (%d cases) -> %s",
             len(pheno), int(pheno["is_case"].sum()), out)


if __name__ == "__main__":
    main()
