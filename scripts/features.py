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


def _long_agg(win_df, name_col, labels):
    """From a long table, return (last, mean, slope) value Series for rows whose
    name_col matches any label (case-insensitive). None if no matching rows."""
    if win_df is None or win_df.empty or name_col not in win_df.columns:
        return None
    labels_l = [s.lower() for s in labels]
    sub = win_df[win_df[name_col].astype(str).str.lower().isin(labels_l)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("date")
    return (sub.groupby("ehr_id")["value"].last(),
            sub.groupby("ehr_id")["value"].mean(),
            sub.groupby("ehr_id").apply(_slope_per_year, include_groups=False))


def lab_features(cfg, out, cohort_tables, win, feats):
    amap = cfg["feature_maps"]["lab_analytes"]
    win_lab = _windowed(cohort_tables.get("labs"), win)
    for key, labels in amap.items():
        feats[f"{key}_last"] = np.nan
        feats[f"{key}_mean"] = np.nan
        agg = _long_agg(win_lab, "analyte", labels)
        if agg is None:
            continue
        last, mean, slope = agg
        feats[f"{key}_last"] = feats.index.map(last)
        feats[f"{key}_mean"] = feats.index.map(mean)
        if key == "hemoglobin":
            feats["hemoglobin_slope"] = feats.index.map(slope)


def vitals_features(cfg, cohort_tables, win, feats, th):
    vmap = cfg["feature_maps"]["vital_signs"]
    win_vit = _windowed(cohort_tables.get("vitals"), win)  # LONG: name/value/date
    for c in ("bmi_last", "sbp_last", "dbp_last", "bmi_slope", "weight_loss_velocity"):
        feats[c] = np.nan
    for vital, labels in vmap.items():
        agg = _long_agg(win_vit, "name", labels)
        if agg is None:
            continue
        last, _mean, slope = agg
        if vital == "bmi":
            feats["bmi_last"] = feats.index.map(last)
            feats["bmi_slope"] = feats.index.map(slope)
        elif vital == "sbp":
            feats["sbp_last"] = feats.index.map(last)
        elif vital == "dbp":
            feats["dbp_last"] = feats.index.map(last)
        elif vital == "weight":
            feats["weight_loss_velocity"] = feats.index.map(slope)


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

    # static demographics / social / questionnaire (one row per patient, not windowed)
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
