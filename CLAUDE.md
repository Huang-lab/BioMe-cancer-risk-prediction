# CLAUDE.md — project constraints for `cancer-risk-biome`

Cancer risk prediction from BioMe EHR + germline variants (BioMe Cohort I,
Regeneron WXS). One shared pipeline, two cancers (colorectal, breast),
selected entirely by config.

These rules are hard. Do not relax them without an explicit instruction that
names the rule being changed.

---

## 1. This container has NO real data — and never will

- BioMe EHR extracts and germline variants are **IRB-restricted PHI**. They
  live **only on Minerva**, under
  `/sc/arion/projects/rg_huangk06/variants_PLP_BioMe/Cancer_risk_prediction`.
- **Real data must never enter this repo or this container.** No `*.txt` EHR
  extracts, no `*.tsv` roster/carrier files, no patient rows, ever.
- `.gitignore` is the backstop (`data/`, `*.txt`, `*.tsv`, `results/*`,
  `*.pkl`). Treat it as a safety net, not permission — do not commit PHI even
  if a path slips past the ignore rules.
- If you ever find real data in the working tree, **stop and flag it** — do
  not commit, do not push.

## 2. Build against `schema/` only

- Develop entirely against **column headers + the data dictionary** in
  `schema/` (`*.header` files, `BRSPD_Data_Dictionary_v4.csv`). That is the
  single source of column names and types available in this container.
- For local testing, **generate a small synthetic table** that matches the
  schema headers exactly (correct column names, plausible dtypes, fake rows).
  Synthetic data is regenerable and stays gitignored (`tests/synthetic/`).
- Never invent a column name. If a needed column is not in `schema/`, it is an
  **open question** — surface it, do not guess it into the code.

## 3. The user runs everything on Minerva

- Rita runs the real pipeline herself on Minerva (**LSF** scheduler, **conda**
  environments, the **module** system). This container does not submit jobs,
  does not have the data, and does not reproduce results.
- Deliverables here are **code + configs + LSF submit scripts + a conda env
  spec** that she runs on Minerva. Write for that target: paths from config,
  no assumptions about a local dataset.

## 4. Everything configurable lives in `config/*.yaml`

- Every path, ICD code set, gene panel, matching parameter, temporal window,
  and model hyperparameter **must come from `config/crc.yaml` or
  `config/breast.yaml`**. The YAMLs are the single source of truth.
- **Never hardcode** a path, code list, column name, or parameter in a script.
  If it varies between the two cancers — or might ever change — it belongs in
  the config, read at runtime.

## 5. One shared pipeline; only the config differs

- The CRC and breast models run **the same scripts**. `crc.yaml` and
  `breast.yaml` share an identical key structure so a single loader reads
  either one.
- Cancer-specific behavior (phenotype codes, gene panel, breast-only
  reproductive features from `OB_HISTORY`) is expressed **as config, not as
  branching code**. Avoid `if cancer == "breast"` in the pipeline; drive it
  from the config keys instead.

---

## Reconciliation discipline

Any column tagged `RECONCILE:` in a YAML is a **placeholder name that must be
confirmed against `schema/` before it is used on Minerva.** Until confirmed it
stays an open question (tracked in `PLAN.md`). Do not silently resolve a
`RECONCILE:` tag by guessing the real column name.

## Status note

`schema/` now holds the **real clinical-file headers** (`*.header`, column names
only) Rita provided, so most `RECONCILE:` column names in the configs are
resolved. Still open (kept as `RECONCILE:` and tracked in `PLAN.md`): the
**roster columns** (incl. the sample_id↔ehr_id crosswalk), the **Cohort II
clinical directory**, `BRSPD_Data_Dictionary_v4.csv` (value codings/units), and
a few categorical value strings. The pipeline runs end-to-end on synthetic data
(`scripts/run.py --with-synthetic`); on Minerva, confirm the remaining
`RECONCILE:` names, then run for real.
