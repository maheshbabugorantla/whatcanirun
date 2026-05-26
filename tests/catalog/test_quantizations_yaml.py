"""Integrity tests for seeds/quantizations.yaml.

Counts and uniqueness only; the schema itself is exercised by
`test_seed_schemas.py` and the loader by `test_loaders.py`.
"""

from __future__ import annotations

from pathlib import Path

from whatcanirun.catalog.loaders import load_quantizations

_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUANTS_PATH = _REPO_ROOT / "seeds" / "quantizations.yaml"


def test_has_exactly_10_rows() -> None:
    rows = load_quantizations(_QUANTS_PATH)
    assert len(rows) == 10


def test_slugs_are_unique() -> None:
    slugs = [r.slug for r in load_quantizations(_QUANTS_PATH)]
    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
    assert duplicates == []


def test_stable_formats_include_fp16_bf16_fp8_int8_int4() -> None:
    rows = load_quantizations(_QUANTS_PATH)
    stable = {r.slug for r in rows if not r.experimental}
    # The book §5.1.1 baseline set we must ship as stable:
    assert {"fp16", "bf16", "fp8", "int8", "int4"}.issubset(stable)


def test_experimental_set_matches_blackwell_microscaling_variants() -> None:
    rows = load_quantizations(_QUANTS_PATH)
    experimental = {r.slug for r in rows if r.experimental}
    assert experimental == {"nvfp4", "mxfp4", "mxfp8"}
