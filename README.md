# cancer-risk-biome

Germline + EHR cancer risk prediction in the **BioMe** biobank (Cohort I,
Regeneron WXS). One shared, config-driven pipeline covering two cancers:

- **Colorectal (CRC)** — `config/crc.yaml`
- **Breast** — `config/breast.yaml`

The two models run the *same* scripts; only the config differs.

> ⚠️ **PHI notice.** BioMe EHR + variant data are IRB-restricted and live only
> on Minerva at
> `/sc/arion/projects/rg_huangk06/variants_PLP_BioMe/Cancer_risk_prediction`.
> No real data is in this repo or ever should be. See `CLAUDE.md`.

## Layout

```
cancer-risk-biome/
├── CLAUDE.md        # hard project constraints (read this first)
├── PLAN.md          # pipeline design + open questions (under review)
├── config/          # crc.yaml, breast.yaml — single source of truth
├── schema/          # column headers + data dictionary (build against these only)
├── scripts/         # shared pipeline (empty — nothing written until PLAN.md is approved)
├── tests/           # unit tests + generated synthetic data (gitignored)
├── lsf/             # LSF submit scripts for Minerva
├── env/             # conda environment spec
└── results/{crc,breast}/   # outputs (gitignored — derived from PHI)
```

## Where things run

- **This container:** scaffolding, code, configs, synthetic-data testing. No PHI.
- **Minerva (Rita runs it):** the real pipeline — LSF scheduler, conda, modules.

## Status

Scaffold + planning only. **No pipeline code is written yet** — see `PLAN.md`
and await approval before any script is implemented.
