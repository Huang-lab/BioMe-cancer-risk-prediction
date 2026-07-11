# CRC risk prediction in BioMe — 20-minute presentation kit

Methods narrative + a slide-by-slide map to the exact output files the pipeline
writes to `…/Cancer_risk_prediction/CRC/`. Fill the numbers/plots from your real
Minerva run; the synthetic figures show the format.

---

## Suggested 20-min flow (~14 slides, ~80s each)

| # | Slide | Content | Output file to show |
|---|-------|---------|---------------------|
| 1 | Title | CRC risk from germline + EHR in BioMe (Cohorts I & II) | — |
| 2 | Motivation | Early CRC risk stratification; germline (Lynch/APC) + routine EHR; who to screen earlier | — |
| 3 | Prior work | Extends `mohibul-07` All-of-Us CRC model; **key lesson: temporal leakage** (their AUC 0.90→0.80 once labs were taken pre-diagnosis) | — |
| 4 | Data | BioMe, 2 WXS cohorts: **Cohort I = Regeneron**, **Cohort II = Sema4**; roster is the spine (SINAI_ID ↔ EHR, `Group`, ancestry) | `phenotype.csv`, log counts |
| 5 | Phenotype | Cases = `Group` **Colon/Rectum**; controls = **Control (age≥50)**, family-cancer-free; index date = dx (cases) / last encounter (controls) | phenotype log line (N cases/controls) |
| 6 | Design | **Propensity-score matching** (age, sex, genetic PCs, cohort), k:1 caliper; clean case/control balance | `match_meta.json` |
| 7 | Anti-leakage | Features only from **[index−730d, index−182d]**; post-index rows dropped + counted | features log ("post-index rows excluded") |
| 8 | Features | Demographics, labs (WBC/Hgb/platelets/creatinine/ALT/AST), vitals/BMI, symptoms, comorbidities, meds, screening, trajectory slopes, SES; **CEA excluded (leakage)** | `feature_spec.json` |
| 9 | Genomics | AlphaMissense-calibrated `all_carriers.tsv` → per-gene + **Lynch/panel** flags | `carriers_wide.csv` |
| 10 | Model | LR / RandomForest / **XGBoost**, RandomizedSearchCV, **StratifiedGroupKFold** (matched sets intact), recall-tuned threshold | `model_metadata.json` |
| 11 | Internal results (Cohort I) | Cross-validated **AUC / PR-AUC**, recall @ threshold | `evaluation.json` → `overall` |
| 12 | **Ancestry-stratified calibration** | Per-`genetically_determined` AUC + calibration (slope/ECE) — the fairness story BioMe enables | `calibration.png`, `evaluation.json → by_ancestry` |
| 13 | Explainability | Top SHAP features (expect age, family hx, Lynch/APC carriers, anemia signals) | `feature_importance.png` / `.csv` |
| 14 | Genomic enrichment | Per-gene case-vs-control carrier frequency + enrichment + Fisher p (expect Lynch/APC strongly enriched) | `carrier_enrichment.csv` |
| 15 | **External validation (Cohort II)** | Train on I → test on II: AUC/PR-AUC + calibration, overall + by ancestry — the generalization headline | `external_validation_cohortII.json`, `calibration_cohortII.png` |
| 16 | Limitations / next | Non-nested CV (mild optimism → Cohort II is the unbiased number); EHR sparsity; screening-ascertainment; breast next | — |

(Collapse 11–14 if you need fewer slides; 12 and 15 are the two "money" slides.)

---

## Methods paragraph (drop-in for a slide or the abstract)

> We built a colorectal-cancer risk model in the BioMe biobank using germline
> whole-exome sequencing and longitudinal EHR across two independent cohorts
> (Regeneron, Sema4). Cases were BioMe `Group` **Colon/Rectum**; controls were
> genetically-screened, family-cancer-free individuals aged ≥50. Controls were
> matched to cases by propensity score (logistic on age, sex, genetic principal
> components, and cohort; k:1 nearest-neighbour caliper). To prevent temporal
> leakage, every feature was extracted from a fixed window ending **6 months
> before** the index date and reaching back **2 years** (post-index records were
> excluded and counted). Features spanned demographics, socioeconomic survey
> items, routine labs and vitals, symptoms, comorbidities, medications,
> colonoscopy history, and per-gene pathogenic/likely-pathogenic germline carrier
> status (AlphaMissense-calibrated), including a Lynch-syndrome aggregate.
> Logistic regression, random forest, and XGBoost were compared by
> cross-validated PR-AUC using StratifiedGroupKFold (matched sets kept intact);
> the decision threshold was tuned for recall subject to a precision floor. The
> model was developed and cross-validated on Cohort I and **externally validated
> on the held-out Cohort II**. Discrimination (AUC, PR-AUC), calibration
> (slope, ECE), and SHAP explanations were reported overall and **stratified by
> genetically-determined ancestry**; per-gene carrier enrichment was tested with
> Fisher's exact test.

---

## What each output file contains

- `evaluation.json` — internal (Cohort I, out-of-fold): `overall` {auc, pr_auc, brier, ece, cal_slope}, `by_ancestry` {per-group same metrics}, `cross_cohort`, `top_features`, `carrier_enrichment_top`.
- `external_validation_cohortII.json` — same metrics on the held-out Cohort II (`trained_on`, `validated_on`, `overall`, `by_ancestry`).
- `calibration.png` / `calibration_cohortII.png` — reliability curves, overall + per ancestry.
- `feature_importance.png` / `.csv` — global SHAP (or |coef|) ranking.
- `carrier_enrichment.csv` / `carrier_enrichment_cohortII.csv` — gene, case/control freq, enrichment ratio, Fisher p.
- `model_metadata.json` — chosen model, feature list, threshold, CV metrics, n/cases.

> Numbers on synthetic data are meaningless placeholders (strong injected signal →
> AUC ~0.90); your real BioMe run produces the values to present.

---

## Reproduce (Minerva)

```bash
module load anaconda3
conda env create -f env/environment.yml && conda activate cancer-risk-biome
bsub < lsf/pipeline.bsub          # runs prep_roster_cohortI/II + full pipeline
#   edit -P (allocation) / -q (queue) in lsf/pipeline.bsub for your account first
```
Outputs land in `/sc/arion/projects/rg_huangk06/variants_PLP_BioMe/Cancer_risk_prediction/CRC/`.
Interactive alternative: run the three `python scripts/…` lines from `lsf/pipeline.bsub` directly.
