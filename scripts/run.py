#!/usr/bin/env python3
"""Orchestrator — run the full pipeline end to end.

Runs each stage as a subprocess in order. On Minerva, submit via lsf/pipeline.bsub;
locally, add --with-synthetic to generate a synthetic dataset first.

  # local smoke test on synthetic data:
  python scripts/run.py --config config/crc.yaml --data-root tests/synthetic --with-synthetic

  # Minerva (real data at paths.workdir):
  python scripts/run.py --config config/crc.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["preprocess", "phenotype", "match", "features",
          "genomics", "build_dataset", "train", "evaluate", "external_validate", "report"]
# stages runnable via --stages but not part of the default end-to-end run
EXTRA_STAGES = ["audit_feature_maps"]


def run(script, extra):
    cmd = [sys.executable, os.path.join(HERE, script)] + extra
    print(f"\n=== {script} ===\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run the full CRC pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--with-synthetic", action="store_true",
                    help="generate synthetic data into --data-root first (local testing)")
    ap.add_argument("--population", choices=["matched", "full", "both"], default="both")
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset to run instead of the full pipeline "
                         "(e.g. 'preprocess,phenotype,audit_feature_maps' for a fast "
                         "checkpoint before the expensive train). Order is preserved.")
    args = ap.parse_args()

    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in stages if s not in STAGES + EXTRA_STAGES]
        if unknown:
            sys.exit(f"unknown stage(s): {unknown}. Known: {STAGES + EXTRA_STAGES}")
    else:
        stages = STAGES

    common = ["--config", args.config]
    if args.data_root:
        common += ["--data-root", args.data_root]

    if args.with_synthetic:
        if not args.data_root:
            sys.exit("--with-synthetic requires --data-root")
        run("make_synthetic.py", ["--config", args.config, "--out", args.data_root])

    for stage in stages:
        extra = list(common)
        if stage == "train":
            extra += ["--population", args.population]
        run(f"{stage}.py", extra)

    print(f"\n{'Checkpoint' if args.stages else 'Pipeline'} complete"
          f" ({', '.join(stages)}).", flush=True)


if __name__ == "__main__":
    main()
