"""Config loading, path resolution, and RECONCILE handling.

A value tagged ``RECONCILE:<name>`` is a PLACEHOLDER column/label name that must
be confirmed against schema/ on Minerva before the real run. In this container we
develop against synthetic data, so we RESOLVE a RECONCILE tag to its bare name
(strip the prefix) and the synthetic generator emits columns under that bare name.
``collect_reconcile`` walks the whole config so the pipeline can report exactly
which names are still unconfirmed.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

RECONCILE_PREFIX = "RECONCILE:"


def load_config(path: str) -> dict:
    """Load a cancer config YAML into a plain dict."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "cancer" not in cfg:
        raise ValueError(f"{path}: not a valid cancer config (missing 'cancer' key)")
    return cfg


def is_reconcile(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(RECONCILE_PREFIX)


def resolve(value: Any) -> Any:
    """Return the working value for a (possibly RECONCILE-tagged) config value.

    - "RECONCILE:masked_mrn"            -> "masked_mrn"
    - "RECONCILE:[PC1, PC2, PC3]"       -> ["PC1", "PC2", "PC3"]
    - ["RECONCILE:CRC", "control"]      -> ["CRC", "control"]
    - anything else                     -> unchanged
    """
    if isinstance(value, list):
        return [resolve(v) for v in value]
    if not is_reconcile(value):
        return value
    bare = value[len(RECONCILE_PREFIX):].strip()
    if bare.startswith("[") and bare.endswith("]"):
        inner = bare[1:-1].strip()
        return [x.strip() for x in inner.split(",")] if inner else []
    return bare


def collect_reconcile(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Walk the config and return (dotted_path, tagged_value) for every RECONCILE tag."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += collect_reconcile(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += collect_reconcile(v, f"{path}[{i}]")
    elif is_reconcile(node):
        found.append((path, node))
    return found


def resolve_path(cfg: dict, rel: str) -> str:
    """Resolve a path relative to paths.workdir (absolute paths pass through)."""
    rel = resolve(rel)
    if os.path.isabs(rel):
        return rel
    workdir = cfg["paths"]["workdir"]
    return os.path.join(workdir, rel)


def resolve_control_labels(cfg: dict) -> list[str]:
    """Return the roster group label(s) meaning "control", regardless of
    whether the config uses the newer `control_labels` (list -- e.g. separate
    age>=50 / age<50 strata) or the older singular `control_label`."""
    r = cfg["roster"]
    if "control_labels" in r:
        return list(resolve(r["control_labels"]))
    return [resolve(r["control_label"])]


def cohort_names(cfg: dict) -> list[str]:
    return [c["name"] for c in cfg["cohorts"]]


def get_cohort(cfg: dict, name: str) -> dict:
    for c in cfg["cohorts"]:
        if c["name"] == name:
            return c
    raise KeyError(f"cohort {name!r} not in config")
