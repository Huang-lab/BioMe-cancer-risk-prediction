# PLAN.md — cancer-risk-biome (CRC), implemented design + open questions

Status: the CRC pipeline and web app are **implemented** and pass an end-to-end
run on synthetic data plus unit tests. This file records the design as built and
the questions still blocking a real Minerva run.

## Design as built

- **Source model:** replicates + extends `mohibul-07/colorectal-cancer-risk-prediction`
  (All of Us CRC model; XGBoost + SHAP; temporal-leakage fix; threshold ~0.24).
- **Cohorts:** Regeneron WXS (I) + Sema4 WXS (II). Pooled model with a `cohort`
  covariate **and** cross-cohort validation (train I→test II, and reverse).
- **Phenotype:** roster is the spine — patient ids + case(CRC)/control labels +
  ancestry `Newgroups`. Clean controls = `group==control`. Index date = earliest
  qualifying CRC dx (cases) / last encounter (controls). Age from
  `Questionnaire.YEAR_OF_BIRTH`.
- **Matching:** propensity-score k:1 nearest-neighbour caliper on age, sex,
  genetic PCs, cohort. Full cohort also emitted (class-weighted) for comparison.
- **Temporal window:** features only from `[index−730d, index−182d]`; post-index
  rows counted + logged.
- **Features:** demographics, labs (long `Order_results`), vitals (long, BMI/BP),
  engineered flags, symptoms (ICD), comorbidities, meds, screening
  (`HEALTH_MAINTENANCE_HISTORY`), trajectory slopes, SES (`YEARS_EDUCATION`,
  marital/religion/language/birthplace), family hx (`FAM_HX_COLON_CANCER`).
  **CEA excluded** from the primary model (leakage/indication bias), reported as
  audit-only.
- **Genomics:** AlphaMissense-calibrated `all_carriers.tsv`
  (`in_ClinVar`/`in_AM`/`in_ACMG`), per-gene `*_any` + `lynch_any`/`crc_panel_any`.
- **Model:** LR / RandomForest / XGBoost via RandomizedSearchCV, StratifiedGroupKFold
  keyed on matched-set id, recall-tuned threshold (precision floor). Matched model
  is primary; full-cohort model is the comparison. Exports `.pkl` + metadata.
- **Evaluation:** OOF AUC/PR-AUC/Brier; **ancestry-stratified AUC + calibration**
  (slope/intercept/ECE); SHAP/importance; per-gene carrier enrichment (Fisher);
  cross-cohort validation.
- **Web:** Streamlit (`web/app.py`) + framework-agnostic `web/predict.py` (mirrors
  the reference's predict contract for an easy FastAPI swap later).

## Cohort I — resolved against real Minerva data

Clinical files key on `sem_id` (SINAI-format values); roster `SINAI_ID` == `sem_id`
(EHR key) == carrier `sample_id`; `MASKED_MRN` (int) joins the separate
`GSA_GDA_PCA_V2.txt` PC file. Config wired: `case_labels=[Colon/Rectum]`,
`control_label="Control (age≥50)"`, `ancestry=genetically_determined`,
`pc_cols=[PC1..PC4]`; `scripts/prep_roster_cohortI.py` builds the augmented roster.

## Open questions (CRC)

1. **Cohort II clinical directory** (`ehr_dir` for Sema4) — cohort II still stubbed.
2. **Exact value strings** for `component_name` (labs) and `vital_sign_description`
   (vitals) — `feature_maps` has best-guesses; confirm against the data.

(The BRSPD data dictionary is not used by the pipeline.)

## Notes / risks carried forward

- Propensity matching balances confounders, not prevalence — that's why the
  full-cohort class-weighted model is kept for comparison.
- No batch correction across platforms; cross-cohort AUC is the empirical
  batch-effect check (carriers already come from one harmonized AM-calibration run).
- Charlson is a simplified comorbidity-count proxy pending a proper code map.
- Decision thresholds are data-driven (recall-tuned); on the strong-signal
  synthetic data they land very low — expect more meaningful values on real data.
