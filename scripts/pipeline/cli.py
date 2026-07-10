"""Shared CLI plumbing for stage scripts.

Every stage takes ``--config`` and an optional ``--data-root`` that overrides
paths.workdir (used for local synthetic runs so inputs AND outputs stay local).
"""
from __future__ import annotations

import argparse
import os

from . import config as cfgmod


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="config/crc.yaml or config/breast.yaml")
    ap.add_argument("--data-root", default=None,
                    help="override paths.workdir (e.g. tests/synthetic for local runs)")
    return ap


def load(args) -> dict:
    cfg = cfgmod.load_config(args.config)
    if args.data_root:
        cfg["paths"]["workdir"] = os.path.abspath(args.data_root)
    return cfg


def out_root(cfg: dict) -> str:
    return cfgmod.resolve_path(cfg, cfg["paths"]["outdir"])
