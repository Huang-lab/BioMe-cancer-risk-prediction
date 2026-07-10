"""Small shared utilities: logging, dates, IO for JSON/parquet-free artifacts."""
from __future__ import annotations

import json
import logging
import os

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column of mixed-format date strings to datetime (coerce failures to NaT)."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)
