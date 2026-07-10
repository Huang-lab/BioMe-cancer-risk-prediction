#!/usr/bin/env python3
"""Generate a small SYNTHETIC dataset matching the config's declared columns.

Mirrors the real BioMe schema (pipe-delimited clinical files, long-format vitals
& labs, Questionnaire with YEAR_OF_BIRTH / FAM_HX_* flags, health-maintenance
history). Writes a local tree under --out that mirrors Minerva, so every stage
runs with --data-root <out>. Fake rows only — regenerable, gitignored, never PHI.

  python scripts/make_synthetic.py --config config/crc.yaml --out tests/synthetic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import config as cfgmod  # noqa: E402
from pipeline import util  # noqa: E402

LOG = util.get_logger("make_synthetic")
ANCESTRY_GROUPS = ["EUR", "AFR", "AMR", "EAS"]
BENIGN_ICD10 = ["I10", "E11", "E785", "M545", "K21", "J45", "R079"]
COMORBID_ICD10 = {"hypertension": "I10", "diabetes": "E119", "ibd": "K509"}
SYMPTOM_ICD10 = {"rectal_bleeding": "K625", "bowel_changes": "R197", "abdominal_pain": "R109"}
DFMT = "%m/%d/%Y"


def _write(df, path, sep, header=True):
    util.ensure_dir(os.path.dirname(path))
    df.to_csv(path, sep=sep, index=False, header=header)


def _local(root, rel):
    return os.path.join(root, cfgmod.resolve(rel))


def _d(ts):
    return ts.strftime(DFMT) if isinstance(ts, pd.Timestamp) else ts


def make_patients(rng, cohort_name, n_cases, n_controls, id_start, cfg):
    case_label = cfgmod.resolve(cfg["roster"]["case_labels"])[0]
    control_label = cfg["roster"]["control_label"]
    n_pc = len(cfgmod.resolve(cfg["roster"]["pc_cols"]))
    rows = []
    for i in range(n_cases + n_controls):
        is_case = i < n_cases
        gid = id_start + i
        index = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 2200)))
        age = int(rng.integers(45, 82))
        rows.append(dict(
            ehr_id=f"RGN{gid:06d}", sample_id=f"SINAI_{gid:05d}", is_case=is_case,
            group=case_label if is_case else control_label,
            ancestry_group=rng.choice(ANCESTRY_GROUPS, p=[0.4, 0.3, 0.2, 0.1]),
            pcs=rng.normal(0, 1, size=n_pc), index_date=index,
            year_of_birth=index.year - age, sex=rng.choice(["M", "F"]), cohort=cohort_name,
        ))
    return pd.DataFrame(rows)


def gen_roster(cfg, patients, path):
    r = cfg["roster"]
    out = pd.DataFrame({
        cfgmod.resolve(r["ehr_id_col"]): patients["ehr_id"],
        cfgmod.resolve(r["sample_id_col"]): patients["sample_id"],
        cfgmod.resolve(r["group_col"]): patients["group"],
        cfgmod.resolve(r["ancestry_group_col"]): patients["ancestry_group"],
    })
    pc_mat = np.vstack(patients["pcs"].to_list())
    for j, name in enumerate(cfgmod.resolve(r["pc_cols"])):
        out[name] = pc_mat[:, j]
    _write(out, path, r.get("sep", "\t"))


def gen_carriers(cfg, patients, cohort_name, path, rng):
    panel = cfg["genomics"]["panel"]
    enriched = panel[:2]
    rows = []
    for _, p in patients.iterrows():
        for gene in panel:
            base = 0.30 if (p["is_case"] and gene in enriched) else 0.03
            if rng.random() < base:
                rows.append(dict(
                    sample_id=p["sample_id"], cohort=cohort_name, gene=gene,
                    chr=f"chr{rng.integers(1, 22)}", pos=int(rng.integers(1e6, 2e8)),
                    ref=rng.choice(list("ACGT")), alt=rng.choice(list("ACGT")),
                    in_ACMG=rng.choice(["yes", "no"], p=[0.4, 0.6]),
                    in_AM=rng.choice(["yes", "no"], p=[0.5, 0.5]),
                    in_ClinVar=rng.choice(["yes", "no"], p=[0.6, 0.4])))
    cols = ["sample_id", "cohort", "gene", "chr", "pos", "ref", "alt",
            "in_ACMG", "in_AM", "in_ClinVar"]
    _write(pd.DataFrame(rows, columns=cols), path, cfg["genomics"]["carrier_file"].get("sep", "\t"))


def _window_dates(rng, index, k):
    return [index - pd.Timedelta(days=int(d)) for d in rng.integers(-180, 1095, size=k)]


def gen_clinical(cfg, patients, ehr_dir, rng):
    tables = cfg["ehr_tables"]

    def raw(table, canon):
        return cfgmod.resolve(tables[table]["cols"][canon])

    def idn(table):
        return cfgmod.resolve(tables[table]["id_col"])

    def path(table):
        return os.path.join(ehr_dir, tables[table]["file"])

    def sep(table):
        return tables[table]["sep"]

    # demographics (one row/patient)
    t = "demographics"
    _write(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "sex"): patients["sex"],
        raw(t, "race"): rng.choice(["White", "Black", "Asian", "Other"], size=len(patients)),
        raw(t, "ethnicity"): rng.choice(["Hispanic", "NotHispanic"], size=len(patients)),
        raw(t, "marital_status"): rng.choice(["Single", "Married", "Divorced", "Widowed"], size=len(patients)),
        raw(t, "religion"): rng.choice(["None", "Catholic", "Jewish", "Other"], size=len(patients)),
        raw(t, "country"): rng.choice(["USA", "DR", "PR", "Other"], size=len(patients)),
    }), path(t), sep(t))

    # social (one row/patient)
    t = "social"
    _write(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "smoking"): rng.choice(["Never", "Former", "Current"], size=len(patients)),
        raw(t, "alcohol"): rng.choice(["Yes", "No"], size=len(patients)),
        raw(t, "years_education"): rng.integers(8, 20, size=len(patients)),
        raw(t, "date"): [_d(d) for d in patients["index_date"] - pd.Timedelta(days=400)],
    }), path(t), sep(t))

    # questionnaire (one row/patient) — YEAR_OF_BIRTH, FAM_HX_COLON_CANCER, pers hx
    t = "questionnaire"
    fam_flag = np.where(patients["is_case"] & (rng.random(len(patients)) < 0.35), "Yes", "No")
    _write(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "year_of_birth"): patients["year_of_birth"],
        raw(t, "country_of_birth"): rng.choice(["USA", "DR", "PR", "MX"], size=len(patients)),
        raw(t, "language_preference"): rng.choice(["English", "Spanish"], size=len(patients)),
        raw(t, "education_grade"): rng.choice(["<HS", "HS", "College", "Grad"], size=len(patients)),
        raw(t, "fam_hx_colon_cancer"): fam_flag,
        raw(t, "pers_hx_ibd"): rng.choice(["Yes", "No"], p=[0.05, 0.95], size=len(patients)),
        raw(t, "pers_hx_diabetes"): rng.choice(["Yes", "No"], p=[0.2, 0.8], size=len(patients)),
        raw(t, "pers_hx_obesity"): rng.choice(["Yes", "No"], p=[0.3, 0.7], size=len(patients)),
        raw(t, "pers_hx_htn"): rng.choice(["Yes", "No"], p=[0.4, 0.6], size=len(patients)),
        raw(t, "smoked_100"): rng.choice(["Yes", "No"], size=len(patients)),
    }), path(t), sep(t))

    # longitudinal collectors
    enc, prob, med, srg, mdh, fam, hm = ([] for _ in range(7))
    vit_rows, lab_rows = [], []
    case_pref = cfg["phenotype"]["case_icd10_prefix"][0]

    for _, p in patients.iterrows():
        eid, idx, is_case = p["ehr_id"], p["index_date"], p["is_case"]

        for d in _window_dates(rng, idx, 6):
            enc.append((eid, rng.choice(BENIGN_ICD10), "ICD-10", "benign", _d(d)))
        if is_case:
            enc.append((eid, case_pref + ".9", "ICD-10", "colorectal cancer", _d(idx)))
        for sym, code in SYMPTOM_ICD10.items():
            if rng.random() < (0.35 if is_case else 0.12):
                enc.append((eid, code, "ICD-10", sym, _d(idx - pd.Timedelta(days=int(rng.integers(182, 730))))))

        for name, code in COMORBID_ICD10.items():
            if rng.random() < 0.25:
                prob.append((eid, code, "ICD-10", name, _d(idx - pd.Timedelta(days=int(rng.integers(200, 1500))))))

        # vitals (LONG): BMI, Weight, BP Systolic, BP Diastolic
        for d in _window_dates(rng, idx, 5):
            bmi = round(float(rng.normal(28, 5)), 1)
            for vname, val in (("BMI", bmi), ("Weight", round(float(rng.normal(80, 15)), 1)),
                               ("BP Systolic", int(rng.normal(130, 15))),
                               ("BP Diastolic", int(rng.normal(80, 10)))):
                vit_rows.append((eid, vname, val, _d(d)))

        # labs (LONG)
        analytes = {"WBC": (6.5, 2), "Hemoglobin": (13.5, 1.5), "Platelet Count": (250, 60),
                    "Creatinine": (0.9, 0.2), "ALT": (25, 10), "AST": (22, 9)}
        for name, (mu, sd) in analytes.items():
            for d in _window_dates(rng, idx, 3):
                lab_rows.append((eid, _d(d), name, round(float(rng.normal(mu, sd)), 1), "u", "N"))
        if rng.random() < (0.5 if is_case else 0.05):
            lab_rows.append((eid, _d(idx - pd.Timedelta(days=90)), "CEA",
                             round(float(rng.normal(3, 2)), 1), "ng/mL", "N"))

        if is_case and rng.random() < 0.3:
            fam.append((eid, "mother", "colorectal cancer", _d(idx - pd.Timedelta(days=2000))))
        for name in ["aspirin", "ibuprofen"]:
            if rng.random() < 0.3:
                med.append((eid, name, _d(idx - pd.Timedelta(days=int(rng.integers(182, 900))))))
        if rng.random() < 0.4:
            srg.append((eid, "colonoscopy", "screening colonoscopy",
                        _d(idx - pd.Timedelta(days=int(rng.integers(200, 1400))))))
        if rng.random() < 0.15:
            srg.append((eid, "polypectomy", "polyp removal",
                        _d(idx - pd.Timedelta(days=int(rng.integers(200, 1400))))))
        mdh.append((eid, "I10", "ICD-10", "hypertension",
                    _d(idx - pd.Timedelta(days=int(rng.integers(200, 1500))))))
        if rng.random() < 0.5:
            hm.append((eid, "Colorectal Cancer Screening",
                       _d(idx - pd.Timedelta(days=int(rng.integers(200, 1600)))), "colonoscopy"))

    def dump(table, rows, canon_order):
        cols = [idn(table)] + [raw(table, c) for c in canon_order]
        _write(pd.DataFrame(rows, columns=cols), path(table), sep(table))

    dump("enc_diagnosis", enc, ["icd_code", "code_type", "text", "date"])
    dump("problem_list", prob, ["icd_code", "code_type", "text", "date"])
    dump("medical_hx", mdh, ["icd_code", "code_type", "text", "date"])
    dump("surgical_hx", srg, ["procedure", "text", "date"])
    dump("family", fam, ["relation", "condition", "date"])
    dump("meds", med, ["name", "date"])
    dump("health_maintenance", hm, ["topic", "date", "type"])

    # vitals (LONG)
    t = "vitals"
    _write(pd.DataFrame(vit_rows, columns=[idn(t), raw(t, "name"), raw(t, "value"), raw(t, "date")]),
           path(t), sep(t))
    # labs (LONG)
    t = "labs"
    _write(pd.DataFrame(lab_rows, columns=[idn(t), raw(t, "date"), raw(t, "analyte"),
                                           raw(t, "value"), raw(t, "units"), raw(t, "flag")]),
           path(t), sep(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="tests/synthetic")
    ap.add_argument("--n-cases", type=int, default=120)
    ap.add_argument("--n-controls", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    rng = np.random.default_rng(args.seed)
    util.ensure_dir(args.out)

    id_start = 1
    for ci, cohort in enumerate(cfg["cohorts"]):
        name = cohort["name"]
        nc = args.n_cases if ci == 0 else int(args.n_cases * 0.7)
        nk = args.n_controls if ci == 0 else int(args.n_controls * 0.7)
        patients = make_patients(rng, name, nc, nk, id_start, cfg)
        id_start += len(patients)
        gen_roster(cfg, patients, _local(args.out, cohort["roster"]))
        gen_carriers(cfg, patients, name, _local(args.out, cohort["carrier_flags"]), rng)
        ehr_dir = _local(args.out, cohort["ehr_dir"])
        util.ensure_dir(ehr_dir)
        gen_clinical(cfg, patients, ehr_dir, rng)
        LOG.info("cohort %s: %d cases + %d controls -> %s", name, nc, nk, ehr_dir)
    LOG.info("synthetic data written under %s", args.out)


if __name__ == "__main__":
    main()
