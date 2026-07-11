#!/usr/bin/env python3
"""Cohort-II (Sema4) roster prep (real-data only): attach GSA/GDA PCs + a SAMPLE_ID
alias, producing the augmented roster the config points at. Same PC file as cohort I
(keyed on MASKED_MRN). Run once on Minerva:  python scripts/prep_roster_cohortII.py

Assumes the Sema4 roster shares the cohort-I column convention (SINAI_ID, MASKED_MRN,
Group, genetically_determined). If a column name differs, adjust below."""
import os
import pandas as pd

WORKDIR = "/sc/arion/projects/rg_huangk06/variants_PLP_BioMe"
ROSTER  = f"{WORKDIR}/metadata/Sema4WXS_HX_Newgroups.250109.tsv"
PCFILE  = f"{WORKDIR}/GSA_GDA_PCA_V2.txt"
OUT     = f"{WORKDIR}/Cancer_risk_prediction/derived/Sema4WXS_HX_Newgroups.withPC.tsv"
PCS     = ["PC1", "PC2", "PC3", "PC4"]   # first 4 of 20
PC_KEY  = "ID2"                          # PC col == roster MASKED_MRN (ID1==ID2)

roster = pd.read_csv(ROSTER, sep="\t", dtype=str)
pc = pd.read_csv(PCFILE, sep=r"\s+", dtype=str, engine="python").rename(columns={PC_KEY: "MASKED_MRN"})
merged = roster.merge(pc[["MASKED_MRN"] + PCS], on="MASKED_MRN", how="left")
merged["SAMPLE_ID"] = merged["SINAI_ID"]   # alias -> avoids ehr_id/sample_id collision

os.makedirs(os.path.dirname(OUT), exist_ok=True)
merged.to_csv(OUT, sep="\t", index=False)
n, have = len(merged), merged["PC1"].notna().sum()
print(f"wrote {OUT}\ntotal: {n} | with PCs: {have} ({have/n:.1%}) | missing: {n-have}")
print(merged["Group"].value_counts(dropna=False).to_string())
