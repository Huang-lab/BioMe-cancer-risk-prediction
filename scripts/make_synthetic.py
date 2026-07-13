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


DECOY_GROUP = "Other Cancer"   # exercises roster.drop_other_groups (neither case nor control)


def make_patients(rng, cohort_name, n_cases, n_controls, id_start, cfg):
    case_label = cfgmod.resolve(cfg["roster"]["case_labels"])[0]
    # both control age strata (mirrors real roster.control_labels); mixed across
    # rows so the pipeline actually exercises accepting BOTH as controls.
    control_labels = cfgmod.resolve_control_labels(cfg)
    n_pc = len(cfgmod.resolve(cfg["roster"]["pc_cols"]))
    n_decoy = max(5, n_controls // 20)
    rows = []
    for i in range(n_cases + n_controls + n_decoy):
        is_case = i < n_cases
        is_control = n_cases <= i < n_cases + n_controls
        gid = id_start + i
        index = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 2200)))
        if is_control:
            group = control_labels[i % len(control_labels)]
            # age matches whichever control stratum the label encodes
            age = int(rng.integers(20, 50)) if "<50" in group else int(rng.integers(50, 82))
        elif is_case:
            group = case_label
            age = int(rng.integers(45, 82))
        else:
            group = DECOY_GROUP
            age = int(rng.integers(45, 82))
        # clinical files key on sem_id whose VALUES are SINAI-format ids; the roster's
        # SINAI_ID == sem_id == carrier sample_id. MASKED_MRN is a separate integer.
        sinai = f"SINAI_{gid}_AB{int(rng.integers(10_000_000, 99_999_999))}"
        rows.append(dict(
            ehr_id=sinai, sample_id=sinai, masked_mrn=str(700_000 + gid),
            is_case=is_case, group=group, age=age,
            ancestry_group=rng.choice(ANCESTRY_GROUPS, p=[0.4, 0.3, 0.2, 0.1]),
            pcs=rng.normal(0, 1, size=n_pc), index_date=index,
            year_of_birth=index.year - age, sex=rng.choice(["M", "F"]), cohort=cohort_name,
        ))
    return pd.DataFrame(rows)


def gen_roster(cfg, patients, path):
    r = cfg["roster"]
    out = pd.DataFrame({
        cfgmod.resolve(r["ehr_id_col"]): patients["ehr_id"],       # SINAI_ID
        cfgmod.resolve(r["sample_id_col"]): patients["sample_id"],  # SAMPLE_ID alias (== SINAI_ID)
        "MASKED_MRN": patients["masked_mrn"],                       # plain int (PC-join key on real data)
        cfgmod.resolve(r["group_col"]): patients["group"],
        cfgmod.resolve(r["ancestry_group_col"]): patients["ancestry_group"],
    })
    if r.get("age_col"):
        out[cfgmod.resolve(r["age_col"])] = patients["age"]        # Age_at_diagnosis
    if r.get("sex_col"):
        out[cfgmod.resolve(r["sex_col"])] = patients["sex"]        # GENDER
    # PCs baked directly into the synthetic roster (the real prep script attaches them on Minerva)
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


def gen_clinical(cfg, patients, ehr_dir, rng, cohort_spec):
    """cohort_spec: the per-cohort dict from cfg['cohorts'] (has file_prefix,
    has_cohort_tag) — lets synthetic mirror Sema4's differences from Regen."""
    tables = cfg["ehr_tables"]
    file_prefix = cohort_spec.get("file_prefix", "")
    has_cohort_tag = cohort_spec.get("has_cohort_tag", True)

    def raw(table, canon):
        return cfgmod.resolve(tables[table]["cols"][canon])

    def idn(table):
        return cfgmod.resolve(tables[table]["id_col"])

    def path(table):
        return os.path.join(ehr_dir, file_prefix + tables[table]["file"])

    def sep(table):
        return tables[table]["sep"]

    manifest = {}

    def emit(df, table):
        """Write a table HEADERLESS + record its columns in the manifest,
        mirroring the real BRSPD layout. If the table+cohort should have a
        leading cohort tag (Regen: 8 tables; Sema4: none), inject it — this
        reproduces the real bug on synthetic so the leading_cols path is
        exercised end-to-end."""
        cols = [str(c) for c in df.columns]
        lead = int(tables[table].get("leading_cols", 0)) if has_cohort_tag else 0
        out = df.copy()
        tag_val = cohort_spec.get("platform", "Regeneron").split("_")[0].capitalize()
        for i in range(lead):
            out.insert(i, f"__cohort_tag_{i}__", tag_val)
        # manifest only records the DOCUMENTED (post-leading) columns
        manifest[file_prefix + tables[table]["file"]] = cols
        _write(out, path(table), sep(table), header=False)

    # demographics (one row/patient)
    t = "demographics"
    emit(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "sex"): patients["sex"],
        raw(t, "race"): rng.choice(["White", "Black", "Asian", "Other"], size=len(patients)),
        raw(t, "ethnicity"): rng.choice(["Hispanic", "NotHispanic"], size=len(patients)),
        raw(t, "marital_status"): rng.choice(["Single", "Married", "Divorced", "Widowed"], size=len(patients)),
        raw(t, "religion"): rng.choice(["None", "Catholic", "Jewish", "Other"], size=len(patients)),
        raw(t, "country"): rng.choice(["USA", "DR", "PR", "Other"], size=len(patients)),
    }), t)

    # social (one row/patient). Raw TOBACCO_USER/IS_ALCOHOL_USER categories are
    # the CONFIRMED real Epic values (collapsed downstream by features.py's
    # SMOKING_MAP / ALCOHOL_MAP -- keep these strings in sync with that map).
    t = "social"
    smoking_raw = ["Never Smoker", "Former Smoker", "Current Every Day Smoker",
                   "Light Smoker", "Unknown if Ever Smoked"]
    alcohol_raw = ["Yes", "No", "Not Currently", "Not Asked"]
    emit(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "smoking"): rng.choice(smoking_raw, size=len(patients)),
        raw(t, "alcohol"): rng.choice(alcohol_raw, size=len(patients)),
        # YEARS_EDUCATION is EMPTY in the real BioMe Social_History export (only
        # 2.2% filled vs 58.6% for Questionnaire.EDUCATION_HIGHEST_GRADE) --
        # leave it blank here so the audit flags it; features.py no longer
        # reads this column at all (real source: Questionnaire, below).
        raw(t, "years_education"): [""] * len(patients),
        raw(t, "date"): [_d(d) for d in patients["index_date"] - pd.Timedelta(days=400)],
    }), t)

    # questionnaire (one row/patient) — YEAR_OF_BIRTH, FAM_HX_COLON_CANCER, pers hx.
    # PERS_HX_* (except PERS_HX_SMOKING) store WHO has the condition, not Yes/No
    # -- "You" (patient), "Your Mother"/etc. (relative, NOT the patient), or "No".
    t = "questionnaire"
    # FAM_HX_COLON_CANCER is ALSO a "who has it" relation column (confirmed real
    # values: Father/Mother's Parents/Mother/Siblings/You/...), not Yes/No --
    # same encoding as PERS_HX_*. Cases get a higher relative-history rate.
    fam_hx_options = ["No history", "Father", "Mother", "Mother's Parents",
                      "Father's Parents", "Siblings", "You"]
    fam_flag = np.array([
        rng.choice(fam_hx_options[1:], p=[0.28, 0.28, 0.18, 0.13, 0.09, 0.04])
        if (is_case and rng.random() < 0.35) else "No history"
        for is_case in patients["is_case"]
    ])
    who_options = ["No", "You", "Your Mother", "Your Father", "You, Your Mother"]
    who_p = [0.55, 0.20, 0.10, 0.10, 0.05]

    def who_col():
        return rng.choice(who_options, p=who_p, size=len(patients))

    # EDUCATION_HIGHEST_GRADE: confirmed real category text (matches
    # features.EDUCATION_ORDINAL_MAP keys exactly, case-insensitive).
    education_raw = ["Elementary/Primary School (includes grades 1 - 5)",
                     "Middle School/Junior High (includes grades 6 - 8)",
                     "High School/Preparatory School",
                     "Trade School/Vocational School",
                     "University/College", "Other"]
    emit(pd.DataFrame({
        idn(t): patients["ehr_id"],
        raw(t, "year_of_birth"): patients["year_of_birth"],
        raw(t, "country_of_birth"): rng.choice(["USA", "DR", "PR", "MX"], size=len(patients)),
        raw(t, "language_preference"): rng.choice(["English", "Spanish"], size=len(patients)),
        raw(t, "education_grade"): rng.choice(education_raw, size=len(patients)),
        raw(t, "fam_hx_colon_cancer"): fam_flag,
        raw(t, "pers_hx_ibd"): who_col(),
        raw(t, "pers_hx_diabetes"): who_col(),
        raw(t, "pers_hx_obesity"): who_col(),
        raw(t, "pers_hx_htn"): who_col(),
        raw(t, "pers_hx_smoking"): rng.choice(["Yes", "No"], size=len(patients)),
        raw(t, "smoked_100"): rng.choice(["Yes", "No"], size=len(patients)),
    }), t)

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

        # vitals (LONG, real BioMe labels): HEIGHT (inches), WEIGHT/SCALE (oz), BP ("sys/dia")
        height_in = round(float(rng.normal(67, 4)), 1)
        for d in _window_dates(rng, idx, 5):
            wt_oz = int(rng.normal(190, 35) * 16)
            sbp, dbp = int(rng.normal(130, 15)), int(rng.normal(80, 10))
            vit_rows.append((eid, "HEIGHT", height_in, _d(d), "INCHES"))
            vit_rows.append((eid, "WEIGHT/SCALE", wt_oz, _d(d), "oz"))
            vit_rows.append((eid, "BP", f"{sbp}/{dbp}", _d(d), "Blood Pressure"))

        # labs (LONG) — component_name + real reference unit, matching
        # feature_maps.lab_analytes' labels/units allowlists.
        analytes = {"WBC": (6.5, 2, "K/uL"), "HGB": (13.5, 1.5, "g/dL"),
                    "PLATELET COUNT": (250, 60, "K/uL"), "CREATININE": (0.9, 0.2, "mg/dL"),
                    "ALT": (25, 10, "U/L"), "AST": (22, 9, "U/L")}
        for name, (mu, sd, unit) in analytes.items():
            for d in _window_dates(rng, idx, 3):
                lab_rows.append((eid, _d(d), name, round(float(rng.normal(mu, sd)), 1), unit, "N"))
        # one implausible sentinel row per ~50 patients -- exercises the
        # lab_sentinel_values filter (confirmed real Order_results artifact).
        if rng.random() < 0.02:
            lab_rows.append((eid, _d(idx - pd.Timedelta(days=200)), "WBC", 9999999, "K/uL", "N"))
        if rng.random() < (0.5 if is_case else 0.05):
            lab_rows.append((eid, _d(idx - pd.Timedelta(days=90)), "CEA",
                             round(float(rng.normal(3, 2)), 1), "ng/mL", "N"))

        if is_case and rng.random() < 0.3:
            # "Colon Cancer" is the confirmed real Family_History term (neither
            # "colorectal" nor "rectal cancer" has any rows in the real data).
            fam.append((eid, "mother", "Colon Cancer", _d(idx - pd.Timedelta(days=2000))))
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
            # "Colonoscopy" is the confirmed real hm_topic_name value --
            # "Colorectal Cancer Screening" has 0 rows in the real data.
            hm.append((eid, "Colonoscopy",
                       _d(idx - pd.Timedelta(days=int(rng.integers(200, 1600)))), "colonoscopy"))

    def dump(table, rows, canon_order):
        cols = [idn(table)] + [raw(table, c) for c in canon_order]
        emit(pd.DataFrame(rows, columns=cols), table)

    dump("enc_diagnosis", enc, ["icd_code", "code_type", "text", "date"])
    dump("problem_list", prob, ["icd_code", "code_type", "text", "date"])
    dump("medical_hx", mdh, ["icd_code", "code_type", "text", "date"])
    dump("surgical_hx", srg, ["procedure", "text", "date"])
    dump("family", fam, ["relation", "condition", "date"])
    dump("meds", med, ["name", "date"])
    dump("health_maintenance", hm, ["topic", "date", "type"])

    # vitals (LONG)
    t = "vitals"
    emit(pd.DataFrame(vit_rows, columns=[idn(t), raw(t, "name"), raw(t, "value"),
                                         raw(t, "date"), raw(t, "unit")]), t)
    # labs (LONG)
    t = "labs"
    emit(pd.DataFrame(lab_rows, columns=[idn(t), raw(t, "date"), raw(t, "analyte"),
                                         raw(t, "value"), raw(t, "units"), raw(t, "flag")]), t)

    # BRSPD-style manifest: headerless data files + Header_File.txt with the columns
    with open(os.path.join(ehr_dir, "Header_File.txt"), "w") as fh:
        for fname, cols in manifest.items():
            fh.write(f"-- {fname}\n\n{'|'.join(cols)}\n\n")


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
        gen_clinical(cfg, patients, ehr_dir, rng, cohort)
        LOG.info("cohort %s: %d cases + %d controls -> %s", name, nc, nk, ehr_dir)
    LOG.info("synthetic data written under %s", args.out)


if __name__ == "__main__":
    main()
