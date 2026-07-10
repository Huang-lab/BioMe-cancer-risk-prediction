# schema/ — column headers + data dictionary (the ONLY thing we build against here)

This directory is the contract between the code in this repo and the real EHR /
variant data that lives on Minerva. It holds **structure only** — column names,
types, and definitions — never any patient rows.

## Expected contents (SOURCE OF TRUTH for column names)

| File                          | What it is                                                        | Status |
|-------------------------------|------------------------------------------------------------------|--------|
| `BRSPD_Data_Dictionary_v4.csv`| Master data dictionary: every EHR column, type, description       | ⛔ NOT YET UPLOADED |
| `Demographics.header`         | Header row of `Demographics.txt`                                  | ⛔ NOT YET UPLOADED |
| `Problem_List.header`         | Header row of `Problem_List.txt`                                  | ⛔ NOT YET UPLOADED |
| `Encounter_Diagnosis.header`  | Header row of `Encounter_Diagnosis.txt`                           | ⛔ NOT YET UPLOADED |
| `Vitals.header`               | Header row of `Vitals.txt`                                        | ⛔ NOT YET UPLOADED |
| `Order_results.header`        | Header row of `Order_results.txt`                                 | ⛔ NOT YET UPLOADED |
| `Social_History.header`       | Header row of `Social_History.txt`                               | ⛔ NOT YET UPLOADED |
| `Family_History.header`       | Header row of `Family_History.txt`                               | ⛔ NOT YET UPLOADED |
| `Medical_History.header`      | Header row of `Medical_History.txt`                              | ⛔ NOT YET UPLOADED |
| `Surgical_History.header`     | Header row of `Surgical_History.txt`                             | ⛔ NOT YET UPLOADED |
| `Medications.header`          | Header row of `Medications.txt`                                  | ⛔ NOT YET UPLOADED |
| `OB_HISTORY.header`           | Header row of `OB_HISTORY.txt` (breast only)                     | ⛔ NOT YET UPLOADED |

> **These files were not part of the initial upload.** Only `crc.yaml` and
> `breast.yaml` were provided. Until the headers + data dictionary land here,
> **no `RECONCILE:` column can be confirmed** — see the open-questions section
> of `PLAN.md`, where all of them are listed as blocked.

## How a header file is captured on Minerva (no data leaves)

```bash
# ONLY the first line — the column names, zero patient rows:
head -n 1 Demographics.txt > Demographics.header
```

Header files contain column names only. Confirm each one is row-free before it
enters this repo. The data dictionary describes columns, not patients, so it is
also safe to commit.
