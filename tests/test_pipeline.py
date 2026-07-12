"""Unit tests for the shared pipeline logic (no real data; synthetic/in-memory).

Run: pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from pipeline import codes, config as cfgmod  # noqa: E402


# --- RECONCILE resolution --------------------------------------------------
def test_resolve_scalar_and_list():
    assert cfgmod.resolve("RECONCILE:masked_mrn") == "masked_mrn"
    assert cfgmod.resolve("plain") == "plain"
    assert cfgmod.resolve(["RECONCILE:CRC", "control"]) == ["CRC", "control"]
    assert cfgmod.resolve("RECONCILE:[PC1, PC2, PC3]") == ["PC1", "PC2", "PC3"]


def test_collect_reconcile_finds_tags():
    cfg = {"a": "RECONCILE:x", "b": {"c": "ok", "d": ["RECONCILE:y"]}}
    paths = {p for p, _ in cfgmod.collect_reconcile(cfg)}
    assert "a" in paths and "b.d[0]" in paths and "b.c" not in paths


# --- ICD matching ----------------------------------------------------------
def test_icd_prefix_matching():
    assert codes.matches_any_prefix("C18.9", ["C18", "C19", "C20"])
    assert codes.matches_any_prefix("153.0", ["153", "154"])
    assert not codes.matches_any_prefix("C50", ["C18", "C19", "C20"])


def test_is_case_code_either_system():
    assert codes.is_case_code("C20", ["C18", "C19", "C20"], ["153", "154"])
    assert codes.is_case_code("1541", ["C18"], ["153", "154"])
    assert not codes.is_case_code("E11", ["C18"], ["153"])


# --- temporal window boundary (the leakage guard) --------------------------
def _windowed(df, win):
    m = df.merge(win, on="ehr_id", how="inner")
    return m[(m["date"] >= m["lo"]) & (m["date"] <= m["hi"])]


def test_temporal_window_excludes_post_index_and_far_past():
    index = pd.Timestamp("2020-01-01")
    win = pd.DataFrame({"ehr_id": ["p"], "lo": [index - pd.Timedelta(days=730)],
                        "hi": [index - pd.Timedelta(days=182)]})
    df = pd.DataFrame({
        "ehr_id": ["p"] * 4,
        "date": [index,                                   # on index -> excluded
                 index - pd.Timedelta(days=90),           # inside blackout -> excluded
                 index - pd.Timedelta(days=400),          # in window -> kept
                 index - pd.Timedelta(days=900)],         # before lookback -> excluded
    })
    kept = _windowed(df, win)
    assert len(kept) == 1
    assert kept.iloc[0]["date"] == index - pd.Timedelta(days=400)


# --- carrier evidence selection -------------------------------------------
def test_carrier_qualifying_any_of():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import genomics
    carriers = pd.DataFrame({
        "sample_id": ["a", "b", "c"], "gene": ["MLH1", "MLH1", "APC"],
        "clinvar": [True, False, False], "alphamissense": [False, False, True],
        "acmg": [False, False, False],
    })
    q = genomics.qualifying(carriers, ["clinvar", "alphamissense"], "any_of")
    assert set(q["sample_id"]) == {"a", "c"}           # b has no positive evidence
    q_strict = genomics.qualifying(carriers, ["acmg"], "any_of")
    assert len(q_strict) == 0


# --- matched-set CV integrity ---------------------------------------------
def test_stratified_group_kfold_keeps_sets_together():
    from sklearn.model_selection import StratifiedGroupKFold
    n_sets = 40
    groups = np.repeat(np.arange(n_sets), 5)              # 5 members per matched set
    y = np.tile([1, 0, 0, 0, 0], n_sets)                 # 1 case + 4 controls per set
    X = np.zeros((len(y), 1))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))   # no set split across folds


# --- headerless clinical files + Header_File.txt manifest -----------------
def test_header_manifest_and_headerless_read(tmp_path):
    from pipeline import io
    ehr = tmp_path
    (ehr / "Header_File.txt").write_text(
        "-- Demographics.txt\n\nrgnid|gender|self_reported_race|zip_first_3_char\n\n")
    # headerless data (first row is a patient), extra trailing column not in config
    (ehr / "Demographics.txt").write_text(
        "SINAI_1_ABC|F|European American|100\nSINAI_2_DEF|M|Black|212\n")
    cfg = {"ehr_tables": {"demographics": {
        "file": "Demographics.txt", "sep": "|",
        "cols": {"sex": "gender", "race": "self_reported_race"}}}}
    manifest = io.load_header_manifest(str(ehr))
    assert manifest["Demographics.txt"][0] == "rgnid"
    df = io.read_ehr_table(cfg, str(ehr), "demographics", header_cols=manifest["Demographics.txt"])
    assert list(df.columns) == ["ehr_id", "sex", "race"]      # id at pos0, extra col dropped
    assert df.iloc[0]["ehr_id"] == "SINAI_1_ABC"
    assert df.iloc[1]["sex"] == "M"


def test_leading_cols_offsets_id_position(tmp_path):
    """Real BioMe files carry an undocumented leading `cohort` tag NOT in the
    manifest; leading_cols must shift the id column to the correct position so the
    patient key isn't misread as 'Regeneron'/'Sema4'."""
    from pipeline import io
    ehr = tmp_path
    (ehr / "Header_File.txt").write_text(
        "-- Social_History.txt\n\nrgnid|TOBACCO_USER|YEARS_EDUCATION\n\n")
    # data has an UNDOCUMENTED leading cohort col BEFORE the manifest names
    (ehr / "Social_History.txt").write_text(
        "Regeneron|SINAI_1_AB|Never|16\nRegeneron|SINAI_2_CD|Current|12\n")
    cfg = {"ehr_tables": {"social": {
        "file": "Social_History.txt", "sep": "|", "leading_cols": 1,
        "cols": {"smoking": "TOBACCO_USER", "years_education": "YEARS_EDUCATION"}}}}
    manifest = io.load_header_manifest(str(ehr))
    df = io.read_ehr_table(cfg, str(ehr), "social",
                           header_cols=manifest["Social_History.txt"])
    assert df.iloc[0]["ehr_id"] == "SINAI_1_AB"           # NOT "Regeneron"
    assert df.iloc[0]["smoking"] == "Never"
    assert df.iloc[1]["years_education"] == "12"


def test_headerless_read_tolerates_literal_quotes(tmp_path):
    """EHR text fields can contain a literal " (e.g. a height 5'2\") — must not break parsing."""
    from pipeline import io
    (tmp_path / "Medications.txt").write_text(
        'SINAI_1_AB|take 1 tab, 5\'2" person|01/01/2020\nSINAI_2_CD|aspirin|02/02/2020\n')
    cfg = {"ehr_tables": {"meds": {"file": "Medications.txt", "sep": "|",
                                   "cols": {"name": "DESCRIPTION", "date": "ordering_date"}}}}
    df = io.read_ehr_table(cfg, str(tmp_path), "meds",
                           header_cols=["rgnid", "DESCRIPTION", "ordering_date"])
    assert len(df) == 2
    assert '5\'2"' in df.iloc[0]["name"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
