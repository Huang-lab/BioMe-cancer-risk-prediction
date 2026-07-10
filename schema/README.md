# schema/ — column headers + data dictionary (the ONLY thing we build against here)

This directory is the contract between the code in this repo and the real EHR /
variant data on Minerva. It holds **structure only** — column names — never any
patient rows.

## Contents

`*.header` files: the real BioMe clinical-file headers (pipe-delimited), column
names only. Structure is identical across Regeneron (Cohort I) and Sema4
(Cohort II); the `Sema4_` filename prefix is dropped here. The pipeline resolves
every column it reads from `config/*.yaml`, which was populated from these
headers. `make_synthetic.py` generates fake tables matching these columns so the
pipeline runs end-to-end locally with no PHI.

Key structural facts baked into the config:
- **Pipe-delimited** clinical files; patient key column `sem_id`.
- **Vitals** and **Order_results** are **long format** (one row per
  measurement: `vital_sign_description`/`component_name` + value + date).
- **No DOB** in Demographics — age is derived from `Questionnaire.YEAR_OF_BIRTH`.
- `Questionnaire` carries rich self-report flags (`FAM_HX_COLON_CANCER`,
  `PERS_HX_*`, `EDUCATION_HIGHEST_GRADE`, `SMOKED_GT_100_CIGARETTES_EVER`).

## Still needed on Minerva (tracked as RECONCILE in the configs / PLAN.md)

| Item | Status |
|------|--------|
| `BRSPD_Data_Dictionary_v4.csv` (value codings, units) | ⛔ not in container |
| Roster header — `RegenWXS_HX_Newgroups.tsv` columns (ehr_id, **sample_id crosswalk**, group labels, `Newgroups`, PCs) | ⛔ not provided; `roster:` keys are RECONCILE |
| Cohort II clinical directory path | ⛔ RECONCILE (`src/Sema4` placeholder) |
| Exact `component_name` / `vital_sign_description` value strings | best-guess maps in `feature_maps` — confirm against data |

## How a header is captured on Minerva (no data leaves)

```bash
head -n 1 Demographics.txt > Demographics.header   # column names only, zero rows
```
