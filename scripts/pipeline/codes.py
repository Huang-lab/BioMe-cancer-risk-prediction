"""ICD code helpers. Prefix sets come from config, never hardcoded here."""
from __future__ import annotations

import re


def normalize_icd(code: str) -> str:
    """Uppercase, strip dots/whitespace: 'C18.9' -> 'C189', ' 153.0 ' -> '1530'."""
    if code is None:
        return ""
    return re.sub(r"[.\s]", "", str(code)).upper()


def matches_any_prefix(code: str, prefixes: list[str]) -> bool:
    norm = normalize_icd(code)
    if not norm:
        return False
    return any(norm.startswith(normalize_icd(p)) for p in prefixes)


def is_case_code(code: str, icd10_prefixes: list[str], icd9_prefixes: list[str]) -> bool:
    """True if an ICD code matches the case definition (either coding system).

    ICD-9 vs ICD-10 is inferred by prefix membership rather than a version column,
    which is safe here because the two prefix sets do not overlap for these cancers.
    """
    return matches_any_prefix(code, icd10_prefixes) or matches_any_prefix(code, icd9_prefixes)
