"""BioMe CRC risk-prediction web app (Streamlit).

Mirrors the reference repo's UX — animated-style risk gauge, top-5 SHAP factors,
and tabs for model performance / feature importance / genomic findings / methods —
but served as a single Python app that loads the pickled Pipeline directly.

Run locally against the synthetic model:
  streamlit run web/app.py -- --model-dir tests/synthetic/results/crc

PHI-free: the form takes manually-entered values only; nothing here reads patient data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as P  # noqa: E402

LEVEL_COLOR = {"LOW": "#1a9850", "MODERATE": "#f39c12", "HIGH": "#d73027"}
LEVEL_MSG = {
    "LOW": "Below screening threshold — routine, age-appropriate monitoring.",
    "MODERATE": "Elevated — consider colonoscopy referral / earlier screening.",
    "HIGH": "High — urgent colonoscopy and genetic counseling recommended.",
}


def resolve_model_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("BIOME_MODEL_DIR", "results/crc"))
    args, _ = ap.parse_known_args()
    for d in (args.model_dir, "tests/synthetic/results/crc", "results/crc"):
        if os.path.exists(os.path.join(d, "model.pkl")):
            return d
    return args.model_dir


def section_of(feat):
    if feat.endswith("_any"):
        return "Germline variants (ClinVar / AlphaMissense)"
    if feat in ("family_hx_crc",):
        return "Family history"
    if feat in ("rectal_bleeding", "bowel_changes", "abdominal_pain"):
        return "Symptoms"
    if feat in ("aspirin", "nsaid"):
        return "Medications"
    if feat in ("prior_colonoscopy", "prior_polypectomy"):
        return "Screening / procedures"
    if feat in ("ibd", "diabetes", "charlson_index"):
        return "Comorbidities"
    if any(k in feat for k in ("wbc", "hemoglobin", "platelet", "creatinine", "alt", "ast",
                               "bmi", "sbp", "dbp", "obese", "hypertension")):
        return "Labs & vitals"
    if feat in ("years_education", "marital_status", "religion", "language_preference",
                "country_of_birth"):
        return "Socioeconomic"
    return "Demographics"


AUTO_FILL = ("cohort",)  # filled from defaults; PCs (pc*) also auto


def build_form(meta):
    values = dict(meta.get("defaults", {}))
    feats = [f for f in meta["feature_cols"] if not f.startswith("pc") and f not in AUTO_FILL]
    sections = {}
    for f in feats:
        sections.setdefault(section_of(f), []).append(f)

    order = ["Demographics", "Labs & vitals", "Symptoms", "Family history",
             "Germline variants (ClinVar / AlphaMissense)", "Screening / procedures",
             "Comorbidities", "Medications", "Socioeconomic"]
    cats = set(meta.get("categorical", []))
    levels = meta.get("categorical_levels", {})

    for sec in order:
        if sec not in sections:
            continue
        with st.expander(sec, expanded=sec in ("Demographics", "Labs & vitals")):
            cols = st.columns(2)
            for i, f in enumerate(sorted(sections[sec])):
                c = cols[i % 2]
                if f in cats:
                    opts = levels.get(f, [])
                    default = values.get(f)
                    idx = opts.index(default) if default in opts else 0
                    values[f] = c.selectbox(f, opts, index=idx) if opts else c.text_input(f, str(default or ""))
                elif f.endswith("_any") or f in ("obese", "hypertension", "high_platelets",
                                                 "low_platelets", "high_creatinine", "ibd",
                                                 "diabetes", "aspirin", "nsaid",
                                                 "prior_colonoscopy", "prior_polypectomy",
                                                 "rectal_bleeding", "bowel_changes",
                                                 "abdominal_pain", "family_hx_crc"):
                    values[f] = 1 if c.checkbox(f.replace("_", " "),
                                                value=bool(values.get(f, 0))) else 0
                else:
                    values[f] = c.number_input(f, value=float(values.get(f) or 0.0))
    return values


def show_result(res):
    color = LEVEL_COLOR[res["risk_level"]]
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:64px;font-weight:800;color:{color}'>{res['risk_score']}%</div>"
            f"<div style='font-size:22px;font-weight:700;color:{color}'>{res['risk_level']} RISK</div>"
            f"</div>", unsafe_allow_html=True)
        st.progress(min(1.0, res["probability"]))
        st.caption(f"Decision threshold (moderate) = {res['threshold']:.3f} | high = {res['high_threshold']:.2f}")
    with b:
        st.info(LEVEL_MSG[res["risk_level"]])
        st.markdown("**Top contributing factors**")
        for fct in res["top_factors"]:
            arrow = "↑" if fct["impact"] > 0 else "↓"
            fc = "#d73027" if fct["impact"] > 0 else "#1a9850"
            width = min(100, abs(fct["impact"]) / (abs(res["top_factors"][0]["impact"]) + 1e-9) * 100)
            st.markdown(
                f"<div style='margin:2px 0'>{arrow} <b>{fct['feature']}</b> "
                f"<span style='color:{fc}'>({fct['direction']})</span>"
                f"<div style='background:{fc};height:8px;width:{width}%;border-radius:4px'></div></div>",
                unsafe_allow_html=True)
    st.caption("⚠️ Research tool on a de-identified cohort — not for clinical decision-making.")


def tab_performance(model_dir, meta):
    st.subheader("Model performance")
    m = meta["metrics"]
    c = st.columns(4)
    c[0].metric("AUC", f"{m['auc']:.3f}")
    c[1].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c[2].metric("Recall @ thr", f"{m.get('recall_at_threshold', float('nan')):.2f}")
    c[3].metric("N (cases)", f"{m['n']} ({m['n_cases']})")
    ev_path = os.path.join(model_dir, "evaluation.json")
    if os.path.exists(ev_path):
        ev = json.load(open(ev_path))
        st.markdown("**Ancestry-stratified (out-of-fold)**")
        rows = [{"ancestry": g, **{k: v for k, v in s.items() if k in
                 ("n", "auc", "brier", "cal_slope", "ece")}}
                for g, s in ev.get("by_ancestry", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
        xc = ev.get("cross_cohort", {})
        if isinstance(xc, dict) and "note" not in xc:
            st.markdown("**Cross-cohort validation**")
            st.dataframe(pd.DataFrame([{"direction": k, **v} for k, v in xc.items()]).round(3),
                         width="stretch")
    cal = os.path.join(model_dir, "calibration.png")
    if os.path.exists(cal):
        st.image(cal, caption="Calibration — overall and by ancestry", width="stretch")


def tab_importance(model_dir):
    st.subheader("Feature importance")
    png = os.path.join(model_dir, "feature_importance.png")
    csv = os.path.join(model_dir, "feature_importance.csv")
    if os.path.exists(png):
        st.image(png, width="stretch")
    if os.path.exists(csv):
        st.dataframe(pd.read_csv(csv).round(4), width="stretch")


def tab_genomics(model_dir):
    st.subheader("Genomic findings — germline carrier enrichment")
    csv = os.path.join(model_dir, "carrier_enrichment.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        st.dataframe(df.round(4), width="stretch")
        st.caption("Enrichment = case carrier frequency / control carrier frequency (Fisher exact p).")
    else:
        st.info("Run scripts/evaluate.py to generate carrier enrichment.")


def tab_methods(meta):
    st.subheader("Methods")
    st.markdown(f"""
- **Cohort:** BioMe (Regeneron + Sema4 WXS), case–control. Cases from the roster
  CRC group; **clean controls** (`group==control`, no family cancer history).
- **Matching:** propensity-score k:1 nearest-neighbour on age, sex, genetic PCs, cohort.
- **Temporal-leakage control:** features only from **[index−730d, index−182d]**
  (6-month pre-index blackout) — the key lesson from the reference model.
- **Model:** {meta['model_name']} selected from LR / RandomForest / XGBoost by
  cross-validated PR-AUC (StratifiedGroupKFold on matched sets). Threshold tuned for
  recall subject to a precision floor.
- **Germline:** AlphaMissense-calibrated `all_carriers.tsv`; per-gene + Lynch/panel aggregates.
- **Evaluation:** AUC/PR-AUC, **ancestry-stratified calibration**, SHAP, carrier
  enrichment, and cross-cohort validation.
- Threshold (moderate) = **{meta['threshold']:.3f}**, high = **{meta['high_threshold']:.2f}**.
""")


def main():
    st.set_page_config(page_title="BioMe CRC Risk", page_icon="🧬", layout="wide")
    model_dir = resolve_model_dir()
    if not os.path.exists(os.path.join(model_dir, "model.pkl")):
        st.error(f"No model found in {model_dir}. Run the pipeline first "
                 f"(scripts/run.py) or pass --model-dir.")
        st.stop()
    model, meta = P.load_artifacts(model_dir)

    st.title("🧬 BioMe Colorectal Cancer Risk")
    st.caption(f"{meta['label']} · model: {meta['model_name']} · population: {meta['population']} "
               f"· AUC {meta['metrics']['auc']:.3f}")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Risk assessment", "Model performance", "Feature importance",
         "Genomic findings", "Methods"])
    with t1:
        st.subheader("Patient inputs")
        with st.form("risk"):
            values = build_form(meta)
            submitted = st.form_submit_button("Estimate risk", type="primary")
        if submitted:
            show_result(P.predict_risk(values, model, meta))
    with t2:
        tab_performance(model_dir, meta)
    with t3:
        tab_importance(model_dir)
    with t4:
        tab_genomics(model_dir)
    with t5:
        tab_methods(meta)


if __name__ == "__main__":
    main()
