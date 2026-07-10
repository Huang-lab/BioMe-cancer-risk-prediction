# PLAN.md — cancer-risk-biome (for review; **STOP — no code until approved**)

Scope: one **shared, config-driven** pipeline for CRC and breast cancer risk in
BioMe (Cohort I, Regeneron WXS). Everything below is derived from
`config/crc.yaml` + `config/breast.yaml`, which are the single source of truth.

> ⛔ **Blocking gap up front:** only the two YAMLs were uploaded. `schema/` has
> **no headers and no `BRSPD_Data_Dictionary_v4.csv`**. Because of that, **every
> `RECONCILE:` column is unconfirmed** and several other schema-dependent choices
> are guesses. See §4. I can write the pipeline against the header contract once
> those files land; I did not want to invent column names.

---

## 1. Shared pipeline stages & script ownership

All scripts take `--config config/{crc,breast}.yaml` and nothing cancer-specific
is hardcoded. Cancer selection = which YAML you pass.

| # | Stage | Script (owner) | Key config keys consumed | Output (gitignored) |
|---|-------|----------------|--------------------------|---------------------|
| 0 | Config load + validation | `scripts/config.py` | whole file | (in-memory) |
| 0b| Synthetic data for local testing | `scripts/make_synthetic.py` | `ehr_tables`, `schema/*.header` | `tests/synthetic/*.txt` |
| 1 | Phenotyping (cases/controls, index_date) | `scripts/phenotype.py` | `phenotype`, `controls`, `paths.roster` | case/control table + index dates |
| 2 | Risk-set matching | `scripts/match.py` | `matching` | matched sets (+ unmatched full cohort) |
| 3 | Temporal windowing + feature build | `scripts/features.py` (+ `scripts/temporal.py` helper) | `temporal_window`, `features`, `ehr_tables` | feature matrix |
| 4 | Genomics carrier join | `scripts/genomics.py` | `genomics`, `paths.carrier_flags` | carrier flags merged in |
| 5 | Assemble modeling dataset | `scripts/build_dataset.py` | `matching.also_emit_unmatched` | matched + full-cohort tables |
| 6 | Train / CV / search | `scripts/train.py` | `model` | fitted models (`*.pkl`) |
| 7 | Evaluation | `scripts/evaluate.py` | `evaluation` | metrics, calibration, SHAP, enrichment |
| — | Orchestration | `scripts/run.py` + `lsf/*.bsub` | `paths` | run logs |

Shared helpers: `scripts/io.py` (read EHR tables via `ehr_tables` + confirmed
headers), `scripts/codes.py` (ICD9/ICD10 prefix matching). One loader reads
either YAML because the key structure is identical across the two files (breast
adds `ob_hx` + `features.breast_specific.reproductive`; the loader treats those
as optional).

**LSF (`lsf/`):** one submit script per stage (or a chained job), `module load`
+ `conda activate` at the top, resources parameterized. `env/` holds the conda
spec (`environment.yml`) Rita builds on Minerva. This container never submits.

---

## 2. Phenotyping, matching, temporal leakage (from the YAMLs)

### Phenotyping (`phenotype.py`)
- **Cases:** ICD match on `case_icd10_prefix` / `case_icd9_prefix`
  (CRC = C18/C19/C20, 153/154; breast = C50, 174/175). `index_date =
  earliest_qualifying_dx` per patient.
- **Controls:** `controls.definition`. Default `any_cancer_free` = patient has
  **no** ICD in the broad malignancy net (`cancer_icd10_prefix: ["C"]` + the
  ICD9 140–209 list). Alt `target_cancer_free` = free of the target cancer only.
- **Universe:** roster (`RegenWXS_HX_Newgroups.250109.tsv`, all WXS patients) ∩
  EHR. Only genotyped patients are eligible.
- **breast `restrict_sex: null`** — keep both sexes, `sex` as covariate (male
  C50 is rare but retained).

### Risk-set matching (`match.py`)
- `method: risk_set`, `k_controls: 4`. A control is eligible for a case iff it is
  **cancer-free and still under observation at that case's `index_date`**;
  matched controls **inherit the case's index_date** (this is what makes the
  temporal window well-defined for controls — see below).
- `match_vars: [age_at_index, sex, ancestry_pcs, followup_len]`,
  `caliper_sd: 0.2`, ancestry via `n_pcs: 4` (fall back to self-reported
  race/ethnicity if PCs missing). `also_emit_unmatched: true` → also emit the
  full cohort with class weights for a matched-vs-unmatched comparison.

### Temporal-leakage window (`temporal.py`)
- Features drawn **only** from `[index_date − max_lookback_days(730),
  index_date − min_lead_days(182)]` — i.e. a 6-month pre-index blackout, 2-year
  lookback. `drop_on_or_after_index: true`; `log_dropped_post_index: true` (the
  count is expected to be large and must be reported — this was the key lesson
  from the source repo). Controls use the matched case's index_date, so the same
  window logic applies uniformly.

---

## 3. Where the ClinVar carrier flags join in

- Stage **4 (`genomics.py`)**, after the feature matrix exists and before dataset
  assembly. Join `paths.carrier_flags`
  (`genomic/{crc,breast}_clinvar_PLP_carriers.tsv`, which **Rita generates
  separately**) on the patient key (`patient_id_col`, RECONCILE).
- Per-gene one-hot over `genomics.panel`, plus aggregates `aggregate_flag`
  (`lynch_any` / `hboc_any`) and `extra_aggregate` (`crc_panel_any` /
  `breast_panel_any`).
- **Non-carrier = 0**: absence from the carrier file is treated as non-carrier.
  This assumes the carrier file lists carriers only and covers all genotyped
  patients — needs confirmation (§4, Q-G1).

---

## 4. `RECONCILE:` columns — open questions

### 4a. Explicitly tagged `RECONCILE:` (both YAMLs) — **NONE confirmable, schema/ is empty**

| ID | Config key | Placeholder | Used for | Status |
|----|-----------|-------------|----------|--------|
| R1 | `phenotype.patient_id_col` | `masked_mrn` | **join key across ALL tables + roster + carrier file** | ❓ cannot confirm — no headers |
| R2 | `phenotype.icd_col` | `icd_code` | case/control ICD matching | ❓ cannot confirm — no headers |
| R3 | `phenotype.dx_date_col` | `diagnosis_date` | `index_date` (in `enc_diagnosis` and/or `problem_list`) | ❓ cannot confirm — no headers |

R1 is the highest-stakes: if the patient key differs in name/format across EHR
tables, roster, and the carrier file, every join breaks. R3 also needs to know
**which table** carries a reliable diagnosis date (encounter dx vs problem list).

### 4b. Additional columns the code needs but the YAMLs don't name — also unconfirmable without schema/

These aren't tagged `RECONCILE:` but are just as blocking; flagging so they're not overlooked:

- **ICD version discriminator** — is ICD9 vs ICD10 flagged by a column, or inferred? (affects R2 matching)
- **`sex`, DOB/`age_at_index`** source columns (Demographics)
- **ancestry PCs** source (a PC file? Demographics?) and **`ancestry_group`** categorical (needed by `evaluation.stratify_by`, not defined anywhere)
- **`followup_len`** derivation (first/last encounter dates → which columns)
- **`bmi`** — a BMI column vs height/weight in Vitals; value + measurement-date columns
- **`smoking_status` / `alcohol_use`** coding in Social_History
- **`family_hx_{crc,breast}`** representation in Family_History
- **Labs** (WBC, hemoglobin, CEA) — analyte identifier (LOINC? test name?), result-value + result-date + units columns in Order_results
- **Procedures** (colonoscopy, polypectomy, breast biopsy) — CPT? free text? in Surgical/Medical_History
- **Medications** (aspirin, nsaid, HRT, OCP) — RxNorm? name-string matching?
- **breast reproductive** (parity, age_at_menarche, age_at_first_birth) columns in OB_HISTORY
- **carrier file schema** — Q-G1 below

---

## 5. Evaluation plan (incl. ancestry-stratified calibration)

- **Metrics** (`evaluation.metrics`): AUC, PR-AUC, calibration — reported
  **OVERALL and per `ancestry_group`** (`stratify_by`). This per-ancestry view is
  the main intended improvement over the source repo.
- **Ancestry-stratified calibration:** calibration curve + calibration
  slope/intercept + Brier + ECE computed **within each ancestry group**, not just
  overall — a model can be well-calibrated in aggregate yet miscalibrated in
  smaller ancestry strata. Report side-by-side with the overall curve.
- **SHAP** (`shap: treeexplainer`) for the tree models; global importance + a
  carrier-vs-noncarrier contrast.
- **Carrier enrichment** (`carrier_enrichment: true`): per-gene case-vs-control
  frequency, enrichment ratio, and a p-value (Fisher/χ²), for `panel` + the
  aggregate flags.
- **CV must be group-aware** (see Q-M2): matched sets kept intact across folds.

---

## 6. Things I think are wrong or risky in the configs

Ordered roughly by impact. None changed — flagging for your call.

**High**
- **Q-M1 — `followup_len` as a match variable + risk-set.** Follow-up length is
  partly a *post-index* quantity for cases (follow-up often ends at diagnosis).
  Matching on it can leak outcome information and fight the risk-set alignment,
  which already handles time. Recommend dropping `followup_len` from `match_vars`
  (keep as a covariate/QC only), or defining it strictly pre-index.
- **Q-M2 — CV leakage across matched sets.** `cv_stratified: true` alone will
  split a case and its 4 controls into different folds. Need
  `StratifiedGroupKFold` keyed on matched-set id. Please confirm I should add
  group-aware CV.
- **Q-P1 — control ICD net asymmetry (ICD9 vs ICD10).** ICD10 uses `"C"` (all
  malignancies, **including** C43 melanoma + C44 non-melanoma skin), but the ICD9
  list **omits 173** (non-melanoma skin). So an NMSC patient is control-eligible
  in the ICD9 era but excluded in the ICD10 era. Decide: exclude NMSC in both
  (add C44 handling / keep 173 out consistently) or include it in both.
- **Q-G2 — aggregate gene subsets aren't defined.** `lynch_any` / `hboc_any` are
  named but the YAML doesn't say which panel genes belong to each aggregate.
  `lynch_any` should be MLH1/MSH2/MSH6/PMS2/EPCAM (not APC/MUTYH/etc.);
  `hboc_any` classically BRCA1/BRCA2 (±PALB2). This mapping must live in config —
  propose adding a `genomics.aggregate_members:` block.

**Medium**
- **Q-P2 — in-situ neoplasms unhandled.** DCIS (D05) / colon in-situ (D01) are
  neither cases (case codes are C-only) nor excluded from controls (net is
  C-prefix / ICD9 malignancy list). A DCIS patient could land in the control
  group. Decide whether in-situ should be excluded from controls (recommended)
  and/or treated as cases.
- **Q-F1 — CEA as a pre-index CRC feature is a leakage risk.** CEA is largely
  ordered during diagnostic workup; even with the 182-day lead, its presence may
  proxy for impending diagnosis. Keep but audit, or gate behind availability.
- **Q-M3 — matching distance is underspecified.** `caliper_sd: 0.2` over a mix of
  continuous age, categorical sex, a 4-dim PC vector, and followup_len needs an
  explicit metric. Propose: exact-match on `sex`, caliper on age, Mahalanobis on
  the 4 PCs; document it in config.
- **Q-G1 — carrier file contract + non-carrier assumption.** Need the carrier
  TSV columns (patient key, gene, classification?) and confirmation that absence
  = non-carrier across *all* genotyped patients (else missing ≠ 0). Also **MUTYH**
  is recessive (biallelic → MAP); a monoallelic MUTYH flag shouldn't be scored
  like a dominant high-penetrance hit — does the carrier file encode zygosity?
- **Q-E1 — `ancestry_group` undefined.** `evaluation.stratify_by: ancestry_group`
  but no column/derivation is specified anywhere (PCs → cluster? self-reported?).
  Needs a definition to stratify calibration.

**Low / notes**
- **Q-M4 — `n_pcs: 4`** may under-capture structure in a cohort as diverse as
  BioMe; consider 10.
- **Q-D1 — `imbalance: scale_pos_weight`** maps cleanly to XGBoost only; RF and
  logistic need `class_weight`. Loader will translate per-model (implementation
  note, not a blocker).
- **Q-D2 — `threshold_tuning: recall`** is underdetermined — need a target
  (e.g. recall floor, or max recall s.t. precision ≥ x). Please specify.
- **Q-F2 — problem-list-derived features** (`charlson_index`, comorbidity flags,
  reproductive history) are often undated/carried-forward; temporal windowing is
  weak for them. Propose treating a documented subset as static with explicit
  missingness handling.

---

## 7. What I will do once you approve (still no code until then)

1. You (or I, if you paste them) drop the headers + `BRSPD_Data_Dictionary_v4.csv`
   into `schema/`; I resolve R1–R3 and §4b against them, updating the YAMLs
   (removing `RECONCILE:` tags only when confirmed).
2. Decisions on Q-M1/M2/M3, Q-P1/P2, Q-G1/G2, Q-E1, Q-D2 folded into config.
3. Implement stages 0→7 as the shared scripts above + LSF submit scripts +
   `env/environment.yml`, with `make_synthetic.py` driving local tests against
   the confirmed headers.

**Awaiting your review of this plan and the open questions before writing any pipeline code.**
