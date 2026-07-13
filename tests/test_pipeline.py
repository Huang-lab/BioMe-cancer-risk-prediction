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


def test_leading_cols_override_disables_shift(tmp_path):
    """Sema4 has NO leading cohort tag while Regen has one; the same shared config
    must not shift Sema4. leading_cols_override=0 forces no shift regardless of
    spec.leading_cols. If it doesn't work, ehr_id captures the (non-existent) tag."""
    from pipeline import io
    ehr = tmp_path
    (ehr / "Header_File.txt").write_text(
        "-- Social_History.txt\n\nsem_id|TOBACCO_USER|YEARS_EDUCATION\n\n")
    # Sema4-shaped file: NO leading cohort tag; id at position 0
    (ehr / "Social_History.txt").write_text("SINAI_1_AB|Never|16\nSINAI_2_CD|Current|12\n")
    cfg = {"ehr_tables": {"social": {
        "file": "Social_History.txt", "sep": "|", "leading_cols": 1,   # shared Regen value
        "cols": {"smoking": "TOBACCO_USER", "years_education": "YEARS_EDUCATION"}}}}
    manifest = io.load_header_manifest(str(ehr))
    df = io.read_ehr_table(cfg, str(ehr), "social",
                           header_cols=manifest["Social_History.txt"],
                           leading_cols_override=0)
    assert df.iloc[0]["ehr_id"] == "SINAI_1_AB"
    assert df.iloc[0]["smoking"] == "Never"


def test_file_prefix_lets_cohort_ii_find_sema4_files(tmp_path):
    """Sema4 files use a 'Sema4_' prefix on disk; the manifest section headers
    also use that prefix. file_prefix should let read_ehr_table find them."""
    from pipeline import io
    ehr = tmp_path
    (ehr / "Sema4_Demographics.txt").write_text("SINAI_1|F|White\n")
    cfg = {"ehr_tables": {"demographics": {
        "file": "Demographics.txt", "sep": "|",
        "cols": {"sex": "gender", "race": "self_reported_race"}}}}
    df = io.read_ehr_table(cfg, str(ehr), "demographics", file_prefix="Sema4_",
                           header_cols=["sem_id", "gender", "self_reported_race"])
    assert df is not None and df.iloc[0]["ehr_id"] == "SINAI_1"


def test_leaked_inline_header_row_dropped(tmp_path):
    """Sema4_Demographics.txt has an inline header row while other Sema4 files
    are headerless. The reader assigns manifest names positionally, then must
    drop the row where ehr_id equals a manifest id keyword."""
    from pipeline import io
    ehr = tmp_path
    (ehr / "Header_File.txt").write_text(
        "-- Demographics.txt\n\nsem_id|gender|self_reported_race\n\n")
    # Row 0 IS the header (Sema4 quirk); rows 1+ are data
    (ehr / "Demographics.txt").write_text(
        "sem_id|gender|self_reported_race\nSINAI_1|F|White\nSINAI_2|M|Black\n")
    cfg = {"ehr_tables": {"demographics": {
        "file": "Demographics.txt", "sep": "|",
        "cols": {"sex": "gender", "race": "self_reported_race"}}}}
    manifest = io.load_header_manifest(str(ehr))
    df = io.read_ehr_table(cfg, str(ehr), "demographics",
                           header_cols=manifest["Demographics.txt"])
    assert len(df) == 2
    assert list(df["ehr_id"]) == ["SINAI_1", "SINAI_2"]     # 'sem_id' row dropped


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


# --- static-feature source fallback (the empty-column guard) ---------------
def test_coalesce_static_falls_back_when_first_source_empty():
    """`education` must come from Questionnaire.EDUCATION_HIGHEST_GRADE when
    Social_History.YEARS_EDUCATION is empty in the real export — the coalesce
    fallback picks the first source that actually has a value per patient."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import features
    index = pd.Index(["p1", "p2", "p3"], name="ehr_id")
    tables = {
        # social YEARS_EDUCATION is entirely blank (mirrors real BioMe)
        "social": pd.DataFrame({"ehr_id": ["p1", "p2", "p3"],
                                "years_education": ["", "", ""]}),
        # questionnaire carries the real education signal (+ p3 missing to prove
        # per-patient coalesce, not all-or-nothing)
        "questionnaire": pd.DataFrame({"ehr_id": ["p1", "p2"],
                                       "education_grade": ["College", "HS"]}),
    }
    sources = [["questionnaire", "education_grade"], ["social", "years_education"]]
    col = features.coalesce_static(index, tables, sources)
    assert col.loc["p1"] == "College" and col.loc["p2"] == "HS"
    assert pd.isna(col.loc["p3"])          # blank social + absent questionnaire -> NaN, not ""

    # reverse priority: a populated first source is NOT overwritten by a later one
    sources_rev = [["social", "years_education"], ["questionnaire", "education_grade"]]
    col2 = features.coalesce_static(index, tables, sources_rev)
    assert col2.loc["p1"] == "College"     # social blank -> still falls through


# --- lab sentinel value + unit allowlist (Fix 1 / Fix 2) -------------------
def test_long_agg_drops_sentinel_value():
    """9999999 is a real data-entry-error sentinel found in Order_results --
    one such row must not blow out last/mean for that patient."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import features
    win_df = pd.DataFrame({
        "ehr_id": ["p1", "p1", "p1"],
        "analyte": ["WBC", "WBC", "WBC"],
        "value": ["6.5", "9999999", "7.0"],
        "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
    })
    agg = features._long_agg(win_df, "analyte", ["WBC"], sentinel_values=[9999999])
    last, mean, _ = agg
    assert last.loc["p1"] == 7.0
    assert mean.loc["p1"] == pytest.approx(6.75)   # (6.5+7.0)/2, sentinel excluded


def test_unit_allowlist_drops_cross_unit_contamination():
    """A WBC value recorded under the wrong reference unit (same component_name,
    different real unit) must be excluded before aggregation."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from features import _apply_unit_allowlist, _long_agg
    win_lab = pd.DataFrame({
        "ehr_id": ["p1", "p1"],
        "analyte": ["WBC", "WBC"],
        "value": ["6.5", "999.0"],
        "units": ["K/uL", "mg/dL"],       # second row: wrong unit for WBC
        "date": pd.to_datetime(["2019-01-01", "2019-02-01"]),
    })
    filtered = _apply_unit_allowlist(win_lab, ["K/uL", "x10E3/uL"])
    last, _, _ = _long_agg(filtered, "analyte", ["WBC"])
    assert last.loc["p1"] == 6.5


# --- vitals physiological clipping (Fix 3) ----------------------------------
def test_vitals_clips_implausible_height_before_bmi():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import features
    win_vit = pd.DataFrame({
        "ehr_id": ["p1", "p2"],
        "name": ["HEIGHT", "HEIGHT"],
        "value": ["67", "5"],            # p2: implausible (5 inches -- data-entry error)
        "date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
        "unit": ["INCHES", "INCHES"],
    })
    cfg = {"feature_maps": {"vital_signs": {"height": ["HEIGHT"], "weight": [], "bp": []},
                            "vitals_units": {"height": "in", "weight": "oz"}}}
    feats = pd.DataFrame(index=pd.Index(["p1", "p2"], name="ehr_id"))
    th = {"height_valid_in": [48, 84], "weight_valid_lb": [30, 500], "bmi_valid": [10, 70]}
    win = pd.DataFrame({"ehr_id": ["p1", "p2"],
                        "lo": pd.to_datetime(["2018-01-01", "2018-01-01"]),
                        "hi": pd.to_datetime(["2020-01-01", "2020-01-01"])})
    features.vitals_features(cfg, {"vitals": win_vit}, win, feats, th)
    assert feats.loc["p1", "height_last"] == 67
    assert pd.isna(feats.loc["p2", "height_last"])   # clipped to NaN, not left at 5


def test_build_row_filters_handles_dict_shaped_lab_analytes():
    """preprocess.py's chunked-reader row filter must extract `labels` from the
    new {labels, units} dict form -- otherwise it iterates the dict's KEYS
    ("labels", "units") instead of the analyte names and silently filters out
    every lab row before features.py ever sees them."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import preprocess
    cfg = {
        "ehr_tables": {"labs": {}},
        "feature_maps": {"lab_analytes": {
            "wbc": {"labels": ["WBC", "WHITE BLOOD CELL"], "units": ["K/uL"]},
            "cea": ["CEA", "CARCINOEMBRYONIC AG"],   # flat-list backward compat
        }},
    }
    filters = preprocess.build_row_filters(cfg)
    assert filters["labs"]["analyte"] == {"wbc", "white blood cell", "cea", "carcinoembryonic ag"}


def test_lab_spec_backward_compat_flat_list():
    from features import _lab_spec
    labels, units = _lab_spec(["WBC", "WHITE BLOOD CELL"])
    assert labels == ["WBC", "WHITE BLOOD CELL"] and units == []
    labels, units = _lab_spec({"labels": ["WBC"], "units": ["K/uL"]})
    assert labels == ["WBC"] and units == ["K/uL"]


# --- PERS_HX_* "who has it" binarization (Fix 4) ----------------------------
def test_pers_hx_you_regex_matches_whole_word_only():
    s = pd.Series(["You", "Your Mother", "You, Your Mother", "No", pd.NA])
    hit = s.astype(str).str.contains(r"\bYou\b", na=False, regex=True)
    assert list(hit) == [True, False, True, False, False]


# --- smoking / alcohol collapse maps (Fix 5 / Fix 6) ------------------------
def test_smoking_map_collapses_to_four_buckets():
    from features import SMOKING_MAP
    raw = pd.Series(["Never Smoker", "Former Smoker", "Current Every Day Smoker",
                     "Light Smoker", "Unknown if Ever Smoked", "Never Assessed"])
    collapsed = raw.str.strip().str.lower().map(SMOKING_MAP).fillna("unknown")
    assert set(collapsed) == {"never", "former", "current", "unknown"}


def test_alcohol_map_not_asked_is_nan_not_zero():
    from features import ALCOHOL_MAP
    raw = pd.Series(["Yes", "No", "Not Currently", "Not Asked"])
    mapped = raw.str.strip().str.lower().map(ALCOHOL_MAP)
    assert mapped.tolist()[:3] == [1, 0, 0]
    assert pd.isna(mapped.iloc[3])          # "Not Asked" -> NaN, NOT 0


# --- education ordinal encoding (Fix 7) -------------------------------------
def test_education_ordinal_map_four_distinct_grades():
    from features import EDUCATION_ORDINAL_MAP
    raw = pd.Series(["Elementary/Primary School (includes grades 1 - 5)",
                     "Middle School/Junior High (includes grades 6 - 8)",
                     "High School/Preparatory School",
                     "Trade School/Vocational School",
                     "University/College", "Other"])
    ordinal = (raw.str.strip().str.lower().str.split(",").str[0].str.strip()
              .map(EDUCATION_ORDINAL_MAP))
    assert set(ordinal.dropna()) == {1, 2, 3, 4}


# --- control_labels resolution (Fix 11) -------------------------------------
def test_resolve_control_labels_list_form():
    cfg = {"roster": {"control_labels": ["Control (age≥50)", "Control (age<50)"]}}
    assert cfgmod.resolve_control_labels(cfg) == ["Control (age≥50)", "Control (age<50)"]


def test_resolve_control_labels_backward_compat_singular():
    cfg = {"roster": {"control_label": "control"}}
    assert cfgmod.resolve_control_labels(cfg) == ["control"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
