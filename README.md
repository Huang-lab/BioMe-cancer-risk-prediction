# cancer-risk-biome

Germline + EHR **colorectal cancer risk prediction** in the **BioMe** biobank
(Cohorts I + II, Regeneron & Sema4 WXS). A config-driven pipeline that replicates
and extends [`mohibul-07/colorectal-cancer-risk-prediction`](https://github.com/mohibul-07/colorectal-cancer-risk-prediction),
plus a Streamlit web app for interactive risk estimation.

> ⚠️ **PHI notice.** BioMe EHR + variant data are IRB-restricted and live only on
> Minerva at `/sc/arion/projects/rg_huangk06/variants_PLP_BioMe/Cancer_risk_prediction`.
> No real data is in this repo or ever should be. All code is developed against
> the column headers in `schema/` and tested on synthetic data. See `CLAUDE.md`.

## What it does (and how it improves on the reference)

- **Two cohorts** (Regeneron + Sema4 WXS): a pooled model **and** cross-cohort
  external validation (train I → test II, and reverse).
- **Propensity-score matching** of cases↔controls (not SMOTE); clean controls are
  taken from the roster `group==control` (no family cancer history).
- **Temporal-leakage control** — features only from `[index−730d, index−182d]`
  (the reference's key lesson).
- **AlphaMissense-calibrated germline carriers** → per-gene + Lynch/panel flags.
- **Ancestry-stratified evaluation & calibration** (BioMe `Newgroups` strata).
- **Web app** mirroring the reference UX: risk gauge, top-5 SHAP factors, model
  performance / feature importance / genomic findings / methods tabs.

## Pipeline (shared scripts, cancer & cohort chosen by config)

`preprocess → phenotype → match → features → genomics → build_dataset → train → evaluate`

| Stage | Script |
|-------|--------|
| Parse raw clinical files (pipe-delimited; long vitals/labs) | `scripts/preprocess.py` |
| Roster-spine case/control + index dates + age (YEAR_OF_BIRTH) | `scripts/phenotype.py` |
| Propensity-score k:1 caliper matching | `scripts/match.py` |
| Temporal-windowed features (labs, vitals, symptoms, trajectory, SES) | `scripts/features.py` |
| Carrier join (`all_carriers.tsv` → per-gene / aggregate flags) | `scripts/genomics.py` |
| Assemble matched + full datasets | `scripts/build_dataset.py` |
| LR/RF/XGBoost, StratifiedGroupKFold, recall-tuned threshold | `scripts/train.py` |
| AUC/PR-AUC, ancestry-stratified calibration, SHAP, enrichment, cross-cohort | `scripts/evaluate.py` |

Shared library in `scripts/pipeline/` (config loader + RECONCILE resolution, IO,
ICD codes, modeling). Everything reads column names/params from `config/*.yaml`.

## Run it

**Locally on synthetic data (no PHI, this container):**
```bash
pip install -r web/requirements.txt          # or use env/environment.yml
python scripts/run.py --config config/crc.yaml --data-root tests/synthetic --with-synthetic
streamlit run web/app.py -- --model-dir tests/synthetic/results/crc
pytest -q
```

**On Minerva (real data at `paths.workdir`):**
```bash
conda env create -f env/environment.yml && conda activate cancer-risk-biome
bsub < lsf/pipeline.bsub          # full pipeline; CONFIG=config/crc.yaml
```
First confirm the remaining `RECONCILE:` names (roster columns, Cohort II dir) —
see `PLAN.md`.

## Layout

```
config/     crc.yaml, breast.yaml (parked stub) — single source of truth
schema/     real clinical-file *.header contracts (column names only)
scripts/    pipeline stages + scripts/pipeline/ shared library
web/        Streamlit app (app.py) + framework-agnostic inference (predict.py)
lsf/        Minerva LSF submit scripts
env/        conda environment.yml
tests/      unit tests (+ gitignored tests/synthetic/ generated data)
results/    outputs (gitignored — derived from PHI)
PLAN.md     design + remaining open questions
```

## Status

CRC pipeline + web app are implemented and pass an end-to-end synthetic run and
unit tests. Breast (`config/breast.yaml`) is a parked stub for a later phase.
Remaining open questions (roster crosswalk, Cohort II path, data dictionary
value codings) are tracked in `PLAN.md`.
