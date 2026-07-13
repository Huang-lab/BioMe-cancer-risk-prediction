#!/usr/bin/env python3
"""Stage 3 — temporal-windowed feature engineering.

Features are drawn ONLY from the window [index - max_lookback, index - min_lead]
(the leakage-control lesson from the reference). Emits features.csv keyed by
ehr_id, and logs the count of post-index rows dropped.

  python scripts/features.py --config config/crc.yaml --data-root tests/synthetic
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import cli, codes, data, util  # noqa: E402

LOG = util.get_logger("features")

# --- confirmed real-value collapses (from BioMe raw-value audit) -----------
# PERS_HX_* (except PERS_HX_SMOKING) store WHO has the condition ("You",
# "Your Mother", ...); \bYou\b (whole word) means the PATIENT has it -- "Your
# Mother" alone must NOT match.
PERS_HX_YOU_COLS = ["pers_hx_ibd", "pers_hx_diabetes", "pers_hx_obesity", "pers_hx_htn"]

SMOKING_MAP = {
    "never": "never",
    "never smoker": "never",
    "never smoker ": "never",
    "never assessed": "never",
    "passive smoke exposure - never smoker": "never",
    "former": "former",
    "former smoker": "former",
    "every day": "current",
    "current every day smoker": "current",
    "some days": "current",
    "current some day smoker": "current",
    "light smoker": "current",
    "light tobacco smoker": "current",
    "heavy smoker": "current",
    "heavy tobacco smoker": "current",
    "smoker, current status unknown": "unknown",
    "unknown if ever smoked": "unknown",
    "unknown": "unknown",
}

ALCOHOL_MAP = {
    "yes": 1, "no": 0, "never": 0,
    "not currently": 0, "not asked": None,
}

# EDUCATION_HIGHEST_GRADE category -> ordinal grade (1-4); anything not listed
# (including blank/unmapped) stays NaN rather than guessing a value.
EDUCATION_ORDINAL_MAP = {
    "elementary/primary school (includes grades 1 - 5)": 1,
    "middle school/junior high (includes grades 6 - 8)": 2,
    "high school/preparatory school": 3,
    "trade school/vocational school": 3,
    "university/college": 4,
    "other": 3,
}

# pandas < 2.2 forwards an unrecognized `include_groups` kwarg straight into the
# applied function instead of consuming it at the `.apply()` level (the pinned
# Minerva env is pandas==2.1.4) -- only pass it where the installed pandas honors it.
_PD_SUPPORTS_INCLUDE_GROUPS = tuple(int(p) for p in pd.__version__.split(".")[:2]) >= (2, 2)


def coalesce_static(index, tables, sources):
    """Build one static (one-row-per-patient) feature from an ORDERED list of
    ``[table, column]`` sources. The first source with a non-empty value for a
    patient wins; later sources only fill patients still missing/blank. This is
    the fallback that lets an empty column in its expected table (e.g. the real
    ``Social_History.YEARS_EDUCATION``) defer to another (``Questionnaire``)
    instead of silently landing in the model as always-NaN.
    """
    col = pd.Series(pd.NA, index=index, dtype=object)
    for table, raw_col in sources:
        df = tables.get(table)
        if df is None or raw_col not in df.columns:
            continue
        mapped = df.drop_duplicates("ehr_id").set_index("ehr_id")[raw_col].reindex(index)
        need = col.isna() | (col.astype(str).str.strip() == "")
        col = col.where(~need, mapped)
    # a blank string from the LAST source tried is still "no data" -- normalize
    # to NaN so callers never have to special-case "" vs missing.
    col = col.where(~(col.astype(str).str.strip() == ""), pd.NA)
    return col


def _windowed(df, win):
    """Rows whose date falls in each subject's [lo, hi] window."""
    if df is None or "date" not in df.columns:
        return None
    m = df.merge(win[["ehr_id", "lo", "hi"]], on="ehr_id", how="inner")
    return m[(m["date"] >= m["lo"]) & (m["date"] <= m["hi"])]


def _slope_per_year(g):
    """Least-squares slope of value vs time (per year); NaN if <2 points."""
    g = g.dropna(subset=["date", "value"])
    if len(g) < 2:
        return np.nan
    x = (g["date"] - g["date"].min()).dt.days.to_numpy() / 365.25
    y = g["value"].to_numpy()
    if np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _long_agg(win_df, name_col, labels, sentinel_values=()):
    """From a long table, return (last, mean, slope) numeric-value Series for rows
    whose name_col EXACTLY matches any label (case-insensitive). None if no rows.
    Coerces value to numeric here (tidy tables keep raw strings so BP survives).
    `sentinel_values` are literal data-entry-error / placeholder codes (e.g. the
    real Order_results 9999999 sentinel) dropped before aggregation -- otherwise
    one sentinel row can blow out last/mean/slope for that patient."""
    if win_df is None or win_df.empty or name_col not in win_df.columns:
        return None
    labels_l = [s.lower() for s in labels]
    sub = win_df[win_df[name_col].astype(str).str.strip().str.lower().isin(labels_l)].copy()
    if sub.empty:
        return None
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    if sentinel_values:
        sub = sub[~sub["value"].isin(sentinel_values)]
    sub = sub.dropna(subset=["value"]).sort_values("date")
    if sub.empty:
        return None
    kwargs = {"include_groups": False} if _PD_SUPPORTS_INCLUDE_GROUPS else {}
    return (sub.groupby("ehr_id")["value"].last(),
            sub.groupby("ehr_id")["value"].mean(),
            sub.groupby("ehr_id").apply(_slope_per_year, **kwargs))


def _lab_spec(spec):
    """Support both the old flat-list config (`['WBC', ...]`) and the newer
    `{labels: [...], units: [...]}` dict form. `units`, when non-empty, is a
    reference-unit ALLOWLIST -- rows whose real units value isn't in it are
    dropped before aggregating (catches cross-unit contamination under the
    same component_name that a label-only match can't see)."""
    if isinstance(spec, dict):
        return spec.get("labels", []), spec.get("units", [])
    return spec, []


def _apply_unit_allowlist(sub_win, unit_allowlist):
    """Drop rows whose `units` value isn't in the allowlist (case-insensitive).
    No-op if there's no allowlist configured or no `units` column to check."""
    if not unit_allowlist or sub_win is None or sub_win.empty or "units" not in sub_win.columns:
        return sub_win
    allow_l = {u.strip().lower() for u in unit_allowlist}
    return sub_win[sub_win["units"].astype(str).str.strip().str.lower().isin(allow_l)]


def lab_features(cfg, out, cohort_tables, win, feats):
    amap = cfg["feature_maps"]["lab_analytes"]
    sentinels = cfg.get("thresholds", {}).get("lab_sentinel_values", [])
    win_lab = _windowed(cohort_tables.get("labs"), win)
    for key, raw_spec in amap.items():
        labels, unit_allowlist = _lab_spec(raw_spec)
        feats[f"{key}_last"] = np.nan
        feats[f"{key}_mean"] = np.nan
        sub_win = _apply_unit_allowlist(win_lab, unit_allowlist)
        agg = _long_agg(sub_win, "analyte", labels, sentinel_values=sentinels)
        if agg is None:
            continue
        last, mean, slope = agg
        feats[f"{key}_last"] = feats.index.map(last)
        feats[f"{key}_mean"] = feats.index.map(mean)
        if key == "hemoglobin":
            feats["hemoglobin_slope"] = feats.index.map(slope)


def _to_lb(series, unit):
    unit = (unit or "lb").lower()
    if unit == "oz":
        return series / 16.0
    if unit == "kg":
        return series * 2.20462
    return series  # lb


def vitals_features(cfg, cohort_tables, win, feats, th):
    """Real BioMe vitals are long-format with HEIGHT (in), WEIGHT/SCALE, and a
    combined BP ('sys/dia'); BMI is COMPUTED (703*lb/in^2), BP is SPLIT."""
    fm = cfg["feature_maps"]
    vmap = fm["vital_signs"]
    units = fm.get("vitals_units", {})
    win_vit = _windowed(cohort_tables.get("vitals"), win)
    for c in ("bmi_last", "sbp_last", "dbp_last", "bmi_slope",
              "weight_loss_velocity", "height_last", "weight_last"):
        feats[c] = np.nan
    if win_vit is None or win_vit.empty:
        return

    h = _long_agg(win_vit, "name", vmap.get("height", []))
    w = _long_agg(win_vit, "name", vmap.get("weight", []))
    if h is not None:
        feats["height_last"] = feats.index.map(h[0])
    if w is not None:
        feats["weight_last"] = feats.index.map(w[0])
        feats["weight_loss_velocity"] = feats.index.map(_to_lb(w[2], units.get("weight")))

    ht_in = pd.to_numeric(feats["height_last"], errors="coerce")
    if units.get("height") == "cm":
        ht_in = ht_in / 2.54
    wt_lb = _to_lb(pd.to_numeric(feats["weight_last"], errors="coerce"), units.get("weight"))

    # physiological-range clipping BEFORE computing BMI -- implausible raw
    # values (data-entry errors, unit mixups) must not propagate into a
    # garbage BMI. Bounds are unit-normalized (inches/lb) so they apply
    # regardless of the cohort's configured raw vitals unit.
    ht_lo, ht_hi = th.get("height_valid_in", [-np.inf, np.inf])
    wt_lo, wt_hi = th.get("weight_valid_lb", [-np.inf, np.inf])
    ht_in = ht_in.where(ht_in.between(ht_lo, ht_hi))
    wt_lb = wt_lb.where(wt_lb.between(wt_lo, wt_hi))
    feats["height_last"] = ht_in     # now standardized to inches (was raw config unit)
    feats["weight_last"] = wt_lb     # now standardized to lb (was raw config unit)

    with np.errstate(divide="ignore", invalid="ignore"):
        feats["bmi_last"] = 703.0 * wt_lb / (ht_in ** 2)
        feats["bmi_slope"] = 703.0 * pd.to_numeric(feats["weight_loss_velocity"],
                                                   errors="coerce") / (ht_in ** 2)
    bmi_lo, bmi_hi = th.get("bmi_valid", [-np.inf, np.inf])
    feats["bmi_last"] = feats["bmi_last"].where(feats["bmi_last"].between(bmi_lo, bmi_hi))

    # blood pressure: value like "120/80" -> split; last in window
    bp_labels = [s.lower() for s in vmap.get("bp", [])]
    bp = win_vit[win_vit["name"].astype(str).str.strip().str.lower().isin(bp_labels)].copy()
    if not bp.empty:
        bp = bp.sort_values("date")
        last_bp = bp.groupby("ehr_id")["value"].last().astype(str)
        feats["sbp_last"] = pd.to_numeric(
            feats.index.map(last_bp.str.extract(r"(\d+)\s*/\s*\d+")[0]), errors="coerce")
        feats["dbp_last"] = pd.to_numeric(
            feats.index.map(last_bp.str.extract(r"\d+\s*/\s*(\d+)")[0]), errors="coerce")


def icd_flag(win_df, prefixes):
    """Series ehr_id -> 1 if any windowed ICD matches a prefix."""
    if win_df is None or win_df.empty or "icd_code" not in win_df.columns:
        return None
    hit = win_df[win_df["icd_code"].apply(lambda c: codes.matches_any_prefix(c, prefixes))]
    return hit.groupby("ehr_id").size().clip(upper=1)


def dx_windowed(cfg, cohort_tables, win):
    frames = [cohort_tables.get(t) for t in ("enc_diagnosis", "problem_list", "medical_hx")]
    frames = [_windowed(f, win) for f in frames if f is not None]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    # medical_hx uses 'condition' text not icd; keep only ICD-bearing
    icd_frames = [f[["ehr_id", "icd_code", "date"]] for f in frames if "icd_code" in f.columns]
    return pd.concat(icd_frames, ignore_index=True) if icd_frames else None


def build_cohort(cfg, out, cohort, pheno_c, tw, th):
    """Feature matrix for one cohort's subjects."""
    tables = {t: data.load_tidy(out, cohort, t) for t in cfg["ehr_tables"]}

    win = pheno_c[["ehr_id", "index_date"]].copy()
    win["index_date"] = pd.to_datetime(win["index_date"])
    win["lo"] = win["index_date"] - pd.Timedelta(days=tw["max_lookback_days"])
    win["hi"] = win["index_date"] - pd.Timedelta(days=tw["min_lead_days"])

    feats = pd.DataFrame(index=pheno_c.set_index("ehr_id").index)

    # labs + vitals + trajectory
    lab_features(cfg, out, tables, win, feats)
    vitals_features(cfg, tables, win, feats, th)

    # engineered flags
    feats["obese"] = (feats["bmi_last"] >= th["obese_bmi"]).astype("Int64")
    feats["high_platelets"] = (feats["platelet_count_last"] > th["platelet_high"]).astype("Int64")
    feats["low_platelets"] = (feats["platelet_count_last"] < th["platelet_low"]).astype("Int64")
    feats["high_creatinine"] = (feats["creatinine_last"] > th["creatinine_high"]).astype("Int64")

    fm = cfg["feature_maps"]
    dxw = dx_windowed(cfg, tables, win)
    htn_dx = icd_flag(dxw, fm["comorbidity_icd"]["hypertension_dx"])
    feats["hypertension"] = (
        (feats["sbp_last"] >= th["hypertension_sbp"]) | (feats["dbp_last"] >= th["hypertension_dbp"])
    ).astype("Int64")
    if htn_dx is not None:
        feats["hypertension"] = feats["hypertension"].fillna(0)
        feats.loc[feats.index.isin(htn_dx.index), "hypertension"] = 1

    # comorbidity + symptom ICD flags
    for key in ("ibd", "diabetes"):
        s = icd_flag(dxw, fm["comorbidity_icd"][key])
        feats[key] = feats.index.map(s).fillna(0).astype("Int64") if s is not None else 0
    for key, prefixes in fm["symptom_icd"].items():
        s = icd_flag(dxw, prefixes)
        feats[key] = feats.index.map(s).fillna(0).astype("Int64") if s is not None else 0

    # charlson proxy: count of distinct comorbidity categories present
    charlson_cats = ["diabetes", "ibd", "hypertension"]
    feats["charlson_index"] = feats[[c for c in charlson_cats if c in feats]].fillna(0).sum(axis=1)

    # medications
    meds = _windowed(tables.get("meds"), win)
    for key, names in fm["medications"].items():
        if meds is None or meds.empty:
            feats[key] = 0
            continue
        pat = "|".join(names)
        hit = meds[meds["name"].astype(str).str.contains(pat, case=False, na=False)]
        feats[key] = feats.index.isin(hit["ehr_id"]).astype(int)

    # procedures: colonoscopy (health-maintenance history, in-window) + polypectomy (surgical)
    hm = tables.get("health_maintenance")
    feats["prior_colonoscopy"] = 0
    if hm is not None and "topic" in hm.columns:
        topics = [t.lower() for t in cfg["ehr_tables"]["health_maintenance"].get(
            "colonoscopy_topics", ["colorectal cancer screening", "colonoscopy"])]
        hm_topic = hm[hm["topic"].astype(str).str.lower().isin(topics)]
        w = _windowed(hm_topic, win)
        if w is not None:
            feats["prior_colonoscopy"] = feats.index.isin(w["ehr_id"]).astype(int)
    srg = _windowed(tables.get("surgical_hx"), win)
    feats["prior_polypectomy"] = 0
    if srg is not None and "procedure" in srg.columns:
        pat = "|".join(fm["procedures"]["prior_polypectomy"])
        poly = srg[srg["procedure"].astype(str).str.contains(pat, case=False, na=False)]
        feats["prior_polypectomy"] = feats.index.isin(poly["ehr_id"]).astype(int)

    # family history of CRC (not temporally windowed — a historical fact).
    # Prefer the questionnaire survey flag; fall back to Family_History free text.
    feats["family_hx_crc"] = 0
    fhcfg = fm["family_history"]
    quest = tables.get("questionnaire")
    flag_col = fhcfg.get("questionnaire_flag")
    if quest is not None and flag_col and flag_col in quest.columns:
        pos = [str(v).lower() for v in fhcfg.get("positive_values", ["yes"])]
        q = quest.drop_duplicates("ehr_id")
        hit = q[q[flag_col].astype(str).str.lower().isin(pos)]
        feats["family_hx_crc"] = feats.index.isin(hit["ehr_id"]).astype(int)
    else:
        fam = tables.get("family")
        if fam is not None and "condition" in fam.columns:
            pat = "|".join(fhcfg["crc_terms"])
            fh = fam[fam["condition"].astype(str).str.contains(pat, case=False, na=False)]
            feats["family_hx_crc"] = feats.index.isin(fh["ehr_id"]).astype(int)

    # static (one-row-per-patient) features — config-driven sourcing with
    # fallback. `static_features` maps each modeling feature to an ORDERED list
    # of [interim_table, canonical_column] sources; the first source with a
    # non-empty value for a patient wins. This is why `education` reads
    # Questionnaire before Social_History: Social_History.YEARS_EDUCATION is
    # empty in the real BioMe export, so it falls back / forward to
    # Questionnaire.EDUCATION_HIGHEST_GRADE instead of dying as always-NaN.
    static_map = cfg.get("static_features")
    if static_map:
        for feat_name, sources in static_map.items():
            feats[feat_name] = coalesce_static(feats.index, tables, sources)
    else:
        # legacy hardcoded sourcing (kept so configs without static_features work)
        def merge_static(table, cols):
            df = tables.get(table)
            if df is None:
                return
            avail = [c for c in cols if c in df.columns]
            if not avail:
                return
            d = df.drop_duplicates("ehr_id").set_index("ehr_id")[avail]
            for c in avail:
                feats[c] = feats.index.map(d[c])

        merge_static("demographics", ["race", "ethnicity", "marital_status", "religion", "country"])
        merge_static("social", ["smoking", "alcohol", "years_education"])
        merge_static("questionnaire", ["country_of_birth", "language_preference"])
        feats.rename(columns={"smoking": "smoking_status", "alcohol": "alcohol_use"}, inplace=True)

    # PERS_HX_* "who has it" columns -> binary "does the PATIENT have it".
    # PERS_HX_SMOKING is the one column that's genuinely Yes/No already.
    for col in PERS_HX_YOU_COLS:
        if col in feats.columns:
            feats[col] = feats[col].astype(str).str.contains(
                r"\bYou\b", na=False, regex=True).astype(int)
    if "pers_hx_smoking" in feats.columns:
        feats["pers_hx_smoking"] = (feats["pers_hx_smoking"].astype(str).str.strip()
                                    .str.lower().isin(["yes", "y"]).astype(int))

    # smoking_status: collapse the raw TOBACCO_USER categories (confirmed real
    # values, incl. a trailing-space variant) into never/former/current/unknown.
    if "smoking_status" in feats.columns:
        feats["smoking_status"] = (feats["smoking_status"]
            .astype(str).str.strip().str.lower()
            .map(SMOKING_MAP).fillna("unknown"))

    # alcohol_use: collapse IS_ALCOHOL_USER to binary; "not asked" -> NaN (not
    # a no-response default) rather than guessing.
    if "alcohol_use" in feats.columns:
        feats["alcohol_use"] = (feats["alcohol_use"]
            .astype(str).str.strip().str.lower()
            .map(ALCOHOL_MAP))

    # years_education: ordinal-encode EDUCATION_HIGHEST_GRADE, then drop the
    # raw category column -- the ordinal value is the real modeling feature.
    if "education_grade" in feats.columns:
        feats["years_education"] = (feats["education_grade"]
            .astype(str).str.strip().str.lower()
            .str.split(",").str[0].str.strip()
            .map(EDUCATION_ORDINAL_MAP))
        feats.drop(columns=["education_grade"], inplace=True)

    feats = feats.reset_index().rename(columns={"index": "ehr_id"})
    return feats, win


def main():
    args = cli.base_parser("Temporal-windowed feature engineering").parse_args()
    cfg = cli.load(args)
    out = cli.out_root(cfg)
    tw = cfg["temporal_window"]
    th = cfg["thresholds"]

    pheno = pd.read_csv(os.path.join(out, "phenotype.csv"))
    pheno["index_date"] = pd.to_datetime(pheno["index_date"])

    all_feats, dropped_total = [], 0
    for cohort in pheno["cohort"].unique():
        pheno_c = pheno[pheno["cohort"] == cohort].copy()
        feats, win = build_cohort(cfg, out, cohort, pheno_c, tw, th)

        # leakage audit: count rows on/after index across date-bearing tables
        if tw.get("log_dropped_post_index", True):
            dropped = 0
            for t in cfg["ehr_tables"]:
                df = data.load_tidy(out, cohort, t)
                if df is not None and "date" in df.columns:
                    m = df.merge(pheno_c[["ehr_id", "index_date"]], on="ehr_id", how="inner")
                    dropped += int((m["date"] >= m["index_date"]).sum())
            dropped_total += dropped
            LOG.info("cohort %s: %d post-index rows excluded by temporal window", cohort, dropped)

        all_feats.append(feats)

    features = pd.concat(all_feats, ignore_index=True)
    features.to_csv(os.path.join(out, "features.csv"), index=False)
    LOG.info("wrote features.csv: %d subjects x %d features (post-index rows dropped: %d) -> %s",
             len(features), features.shape[1] - 1, dropped_total, out)


if __name__ == "__main__":
    main()
