#!/usr/bin/env python3
"""Post-build_dataset CHECKPOINT — assert the concrete real-data feature-source
fixes actually landed correctly, without spending time on train/evaluate.

Run this right after `build_dataset.py` (e.g. as the last step of a fast
features-onward rerun) to catch a silently-wrong fix before burning compute on
the model or drafting slides on broken numbers.

  python scripts/sanity_check.py --out <outdir> --cohort cohortI --population matched

Exits 0 if every check passes, 1 otherwise (so it can gate a job script).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import data, util  # noqa: E402

LOG = util.get_logger("sanity_check")


def _dead(s: pd.Series) -> bool:
    """A feature carries zero modeling signal if it's entirely null or has a
    single distinct non-null value (constant -- nothing for a model to learn)."""
    nn = s.dropna()
    return nn.empty or nn.nunique() <= 1


def check_creatinine_median(s):
    nn = pd.to_numeric(s, errors="coerce").dropna()
    if nn.empty:
        return False, "no values"
    med = nn.median()
    return 0.5 <= med <= 1.5, f"median={med:.3f} (expect ~0.9)"


def check_wbc_max(s):
    nn = pd.to_numeric(s, errors="coerce").dropna()
    if nn.empty:
        return False, "no values"
    mx = nn.max()
    return mx < 100, f"max={mx:.1f} (sentinel 9999999 must be gone)"


def check_bmi_max(s):
    nn = pd.to_numeric(s, errors="coerce").dropna()
    if nn.empty:
        return False, "no values"
    mx = nn.max()
    return mx < 70, f"max={mx:.1f} (height/weight outliers must be clipped)"


def check_nonzero(s):
    nn = pd.to_numeric(s, errors="coerce").dropna()
    n = int((nn > 0).sum())
    return n > 0, f"{n} nonzero"


def check_nunique(expected):
    def _f(s):
        nn = s.dropna()
        u = nn.nunique()
        vals = sorted(str(v) for v in nn.unique())[:6]
        return u == expected, f"nunique={u} (expect {expected}) values~{vals}"
    return _f


CHECKS = [
    ("creatinine_last", check_creatinine_median),
    ("wbc_last", check_wbc_max),
    ("bmi_last", check_bmi_max),
    ("pers_hx_ibd", check_nonzero),
    ("smoking_status", check_nunique(4)),
    ("years_education", check_nunique(4)),
    ("alcohol_use", check_nunique(2)),
]


def main():
    ap = argparse.ArgumentParser(description="Post-build_dataset sanity check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default=None, help="restrict to one cohort (else all)")
    ap.add_argument("--population", choices=["matched", "full"], default="matched")
    args = ap.parse_args()

    path = os.path.join(args.out, f"dataset_{args.population}.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run build_dataset.py first")
    df = pd.read_csv(path)
    if args.cohort:
        df = df[df["cohort"] == args.cohort]
    LOG.info("checking %s (population=%s, cohort=%s): %d subjects",
             path, args.population, args.cohort or "all", len(df))

    results = []
    for feat, check in CHECKS:
        if feat not in df.columns:
            results.append((feat, False, "COLUMN MISSING"))
            continue
        ok, detail = check(df[feat])
        results.append((feat, ok, detail))

    # generic dead-feature sweep across EVERY modeling feature, not just the
    # ones with an explicit check above -- this is what catches the NEXT
    # silent empty-column bug, not just the ones already found this round.
    spec_path = os.path.join(args.out, "feature_spec.json")
    dead = []
    if os.path.exists(spec_path):
        spec = util.load_json(spec_path)
        for feat in spec.get("feature_cols", []):
            if feat == "cohort" and args.cohort:
                continue    # trivially constant once --cohort filters to one cohort
            if feat in df.columns and _dead(df[feat]):
                dead.append(feat)

    # confirm the questionnaire interim table actually carries
    # FAM_HX_COLON_CANCER (it sits far out in the raw file -- explicitly
    # checking it landed guards against a silent parse truncation).
    quest_ok = None
    if args.cohort:
        q = data.load_tidy(args.out, args.cohort, "questionnaire")
        quest_ok = q is not None and "fam_hx_colon_cancer" in q.columns

    print("\n=== SANITY CHECK ===")
    all_pass = True
    for feat, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"  [{status}] {feat}: {detail}")

    print(f"\n  questionnaire.fam_hx_colon_cancer present in interim: "
          f"{'YES' if quest_ok else ('NO' if quest_ok is False else 'unknown (pass --cohort)')}")
    if quest_ok is False:
        all_pass = False

    if dead:
        print(f"\n  ! DEAD features (entirely null or constant -- carry zero signal): {dead}")
        all_pass = False
    else:
        print("\n  no dead features among feature_cols.")

    print(f"\n=== SUMMARY: "
          f"{'ALL CHECKS PASSED' if all_pass else 'FAILURES -- DO NOT PROCEED TO TRAIN/SLIDES'} ===\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
