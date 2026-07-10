"""Cleaning + interim-artifact helpers shared across stages.

The pipeline persists intermediate frames as CSVs under ``<outdir>/interim`` so
each stage can run as its own LSF job. Canonical column names (from io.py) are
used throughout; raw/placeholder names never appear past preprocessing.
"""
from __future__ import annotations

import os

import pandas as pd

from . import util

DATE_KEYS = {"date", "birth_date", "last_done", "next_due", "measure_date", "outcome_date"}
NUMERIC_KEYS = {"value", "bmi", "height", "weight", "sbp", "dbp", "years_education",
                "parity", "age_at_menarche", "age_at_first_birth", "year_of_birth"}


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in DATE_KEYS or col.endswith("_date") or col == "date":
            df[col] = util.parse_dates(df[col])
        elif col in NUMERIC_KEYS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def interim_dir(out_root: str, cohort: str) -> str:
    return os.path.join(out_root, "interim", cohort)


def save_tidy(df: pd.DataFrame, out_root: str, cohort: str, table: str) -> str:
    d = util.ensure_dir(interim_dir(out_root, cohort))
    path = os.path.join(d, f"{table}.csv")
    df.to_csv(path, index=False)
    return path


def load_tidy(out_root: str, cohort: str, table: str) -> pd.DataFrame | None:
    path = os.path.join(interim_dir(out_root, cohort), f"{table}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return clean_table(df)
