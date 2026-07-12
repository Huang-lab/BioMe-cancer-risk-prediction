"""Readers for the RAW clinical files, roster, and carrier file.

Each reader resolves column names from config (RECONCILE tags stripped), selects
only the needed columns, and renames them to stable *canonical keys* so the rest
of the pipeline never sees a raw/placeholder name. The canonical patient key is
always ``ehr_id``.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

import pandas as pd

from . import config as cfgmod
from . import util

LOG = util.get_logger("io")
HEADER_MANIFEST = "Header_File.txt"


CHUNK_ROWS = 1_000_000   # ~1M rows per chunk keeps peak memory flat on multi-GB files


def _read_delimited(path: str, sep: str, header: bool = True, usecols=None,
                    postprocess=None) -> pd.DataFrame:
    """Read a pipe/tab file with the fast C parser and QUOTE_NONE (literal quotes
    are fine, e.g. 5'2" in SIG fields). For files > 512 MB, stream in chunks and
    apply ``postprocess`` per-chunk (rename/filter/drop) BEFORE concatenating —
    this keeps peak memory flat regardless of file size."""
    kw = dict(
        sep=sep, dtype=str,
        header=0 if header else None, usecols=usecols,
        keep_default_na=True, na_values=["", "NA", "NaN", "null", "."],
        engine="c", quoting=csv.QUOTE_NONE, on_bad_lines="skip",
    )
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if size < 512 * 1024 * 1024:
        raw = pd.read_csv(path, **kw)
        return postprocess(raw) if postprocess is not None else raw
    LOG.info("reading %s (%.1f GB) in %d-row chunks", path, size / 1e9, CHUNK_ROWS)
    parts, read = [], 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_ROWS, **kw)):
        read += len(chunk)
        if postprocess is not None:
            chunk = postprocess(chunk)
        parts.append(chunk)
        if (i + 1) % 10 == 0:
            kept = sum(len(p) for p in parts)
            LOG.info("  %s: %d chunks (%d read, %d kept)",
                     os.path.basename(path), i + 1, read, kept)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_header_manifest(ehr_dir: str, sep: str = "|",
                         filename: str = HEADER_MANIFEST) -> dict[str, list[str]]:
    """Parse a BRSPD Header_File.txt manifest -> {data_filename: [column names]}.

    Format (sections):
        -- Demographics.txt
        rgnid|gender|ethnic_group_c|...
    Returns {} if no manifest is present (files then assumed to carry inline headers).
    """
    path = os.path.join(ehr_dir, filename)
    if not os.path.exists(path):
        return {}
    manifest: dict[str, list[str]] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("--"):
                current = s.lstrip("-").strip()          # "Demographics.txt"
            elif current and sep in s:
                manifest[current] = [c.strip() for c in s.split(sep)]
                current = None
    return manifest


def read_ehr_table(cfg: dict, ehr_dir: str, table_key: str,
                   required: bool = False, id_override: str = None,
                   header_cols: list[str] = None,
                   row_filter: dict[str, set] = None) -> Optional[pd.DataFrame]:
    """Read one raw clinical table and return it with canonical column names.

    If ``header_cols`` is given (from a Header_File.txt manifest), the data file is
    read HEADERLESS and those names are assigned (column 0 = patient id). Otherwise
    the file is assumed to carry an inline header. Missing optional columns are
    warned about and skipped (real files carry extra columns and some vary); only
    the patient id column is required. Returns None if the file is absent.
    """
    spec = cfg["ehr_tables"][table_key]
    path = os.path.join(ehr_dir, spec["file"])
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{table_key}: missing required file {path}")
        return None

    sep = spec.get("sep", "\t")

    if header_cols is not None:
        # headerless data + manifest names; read ONLY the columns we need and FILTER
        # inside each chunk so multi-GB files stay tractable.
        #
        # `leading_cols`: some raw files carry N extra UNDOCUMENTED leading columns
        # NOT present in Header_File.txt (BioMe files have a leading `cohort` tag
        # in 8/12 tables). Without this offset, manifest names would be applied at
        # position 0 and the patient id would be captured from the wrong column
        # (silently zeroing every join). With `lead=N`, real data begins at column
        # N and manifest names[0] is the patient id.
        lead = int(spec.get("leading_cols", 0))
        names_full = [f"__lead{i}__" for i in range(lead)] + [c.strip() for c in header_cols]
        wanted = {cfgmod.resolve(v) for v in spec.get("cols", {}).values()}
        keep = sorted({lead} | {i for i, c in enumerate(names_full) if c in wanted})
        usenames = [names_full[i] if i < len(names_full) else f"col{i}" for i in keep]
        rename = {}
        for canon, raw_name in spec.get("cols", {}).items():
            rename[cfgmod.resolve(raw_name)] = canon
        # rename is applied per-chunk so row_filter (by canonical col) works cheaply.

        def _prep(chunk: pd.DataFrame) -> pd.DataFrame:
            chunk.columns = usenames[:chunk.shape[1]]
            id_raw_local = chunk.columns[0]
            present = {id_raw_local: "ehr_id"}
            for k, v in rename.items():
                if k in chunk.columns:
                    present[k] = v
            chunk = chunk[list(present)].rename(columns=present)
            if row_filter:
                for col, allowed in row_filter.items():
                    if col in chunk.columns:
                        chunk = chunk[
                            chunk[col].astype(str).str.strip().str.lower().isin(allowed)]
            return chunk

        return _read_delimited(path, sep, header=False, usecols=keep, postprocess=_prep)
    elif not spec.get("header", True):
        # legacy positional (headerless, explicit indices)
        raw = _read_delimited(path, sep, header=False)
        out = pd.DataFrame({"ehr_id": raw.iloc[:, spec["id_col_idx"]]})
        for canon, idx in spec.get("cols_idx", {}).items():
            out[canon] = raw.iloc[:, idx]
        return out
    else:
        raw = _read_delimited(path, sep, header=True)
        raw.columns = [c.strip() for c in raw.columns]
        id_raw = cfgmod.resolve(id_override or spec["id_col"])

    if id_raw not in raw.columns:
        raise KeyError(f"{table_key} ({path}): id column {id_raw!r} not found. "
                       f"Available: {list(raw.columns)[:15]}")
    rename = {id_raw: "ehr_id"}
    for canon, raw_name in spec.get("cols", {}).items():
        rename[cfgmod.resolve(raw_name)] = canon
    present = {k: v for k, v in rename.items() if k in raw.columns}
    missing = [k for k in rename if k not in raw.columns]
    if missing:
        LOG.warning("%s: columns not found, skipped: %s", table_key, missing)
    return raw[list(present)].rename(columns=present)


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
