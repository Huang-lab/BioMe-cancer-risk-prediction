"""Readers for the RAW clinical files, roster, and carrier file.

Each reader resolves column names from config (RECONCILE tags stripped), selects
only the needed columns, and renames them to stable *canonical keys* so the rest
of the pipeline never sees a raw/placeholder name. The canonical patient key is
always ``ehr_id``.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from . import config as cfgmod


def _read_delimited(path: str, sep: str, header: bool = True) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        header=0 if header else None,
        keep_default_na=True,
        na_values=["", "NA", "NaN", "null", "."],
        engine="python",
    )


def read_ehr_table(cfg: dict, ehr_dir: str, table_key: str,
                   required: bool = False, id_override: str = None) -> Optional[pd.DataFrame]:
    """Read one raw clinical table and return it with canonical column names.

    ``id_override`` (e.g. a cohort's clinical_id_col) replaces the table id column.
    Returns None if the file is absent and ``required`` is False.
    """
    spec = cfg["ehr_tables"][table_key]
    path = os.path.join(ehr_dir, spec["file"])
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{table_key}: missing required file {path}")
        return None

    sep = spec.get("sep", "\t")
    has_header = spec.get("header", True)

    if not has_header:
        # positional file (headerless, delimited)
        raw = _read_delimited(path, sep, header=False)
        out = pd.DataFrame()
        out["ehr_id"] = raw.iloc[:, spec["id_col_idx"]]
        for canon, idx in spec.get("cols_idx", {}).items():
            out[canon] = raw.iloc[:, idx]
        return out

    raw = _read_delimited(path, sep, header=True)
    raw.columns = [c.strip() for c in raw.columns]  # some headers carry trailing spaces
    id_raw = cfgmod.resolve(id_override or spec["id_col"])
    rename = {id_raw: "ehr_id"}
    for canon, raw_name in spec.get("cols", {}).items():
        rename[cfgmod.resolve(raw_name)] = canon
    missing = [c for c in rename if c not in raw.columns]
    if missing:
        raise KeyError(
            f"{table_key} ({path}): columns {missing} not found. "
            f"Available: {list(raw.columns)[:20]}. Check the RECONCILE names in config."
        )
    return raw[list(rename)].rename(columns=rename)


def read_roster(cfg: dict, path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read the roster (the spine). Returns (df, pc_cols) with canonical names:
    ehr_id, sample_id, group, ancestry_group, pc1..pcK.
    """
    r = cfg["roster"]
    raw = _read_delimited(path, r.get("sep", "\t"), header=True)

    rename = {
        cfgmod.resolve(r["ehr_id_col"]): "ehr_id",
        cfgmod.resolve(r["sample_id_col"]): "sample_id",
        cfgmod.resolve(r["group_col"]): "group",
        cfgmod.resolve(r["ancestry_group_col"]): "ancestry_group",
    }
    pc_src = cfgmod.resolve(r["pc_cols"])
    pc_canon = [f"pc{i+1}" for i in range(len(pc_src))]
    for src, canon in zip(pc_src, pc_canon):
        rename[src] = canon

    # optional curated columns in the roster (used preferentially by phenotype)
    for opt_key, canon in (("age_col", "roster_age"), ("sex_col", "roster_sex")):
        col = r.get(opt_key)
        if col:
            col = cfgmod.resolve(col)
            if col in raw.columns:
                rename[col] = canon

    missing = [c for c in rename if c not in raw.columns]
    if missing:
        raise KeyError(
            f"roster ({path}): columns {missing} not found. Available: {list(raw.columns)[:20]}."
        )
    df = raw[list(rename)].rename(columns=rename)
    return df, pc_canon


def read_carriers(cfg: dict, path: str) -> pd.DataFrame:
    """Read one cohort's all_carriers.tsv with canonical columns:
    sample_id, gene, and one boolean column per evidence source (clinvar/alphamissense/acmg).
    """
    cf = cfg["genomics"]["carrier_file"]
    raw = _read_delimited(path, cf.get("sep", "\t"), header=True)
    pos = str(cf.get("positive_value", "yes")).lower()

    out = pd.DataFrame()
    out["sample_id"] = raw[cf["sample_id_col"]]
    out["gene"] = raw[cf["gene_col"]]
    for source, col in cf["evidence_cols"].items():
        if col in raw.columns:
            out[source] = raw[col].astype(str).str.lower().eq(pos)
        else:
            out[source] = False
    return out
