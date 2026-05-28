"""Slice E: `seeds/aa_slug_mapping.yaml` loader and resolution.

The mapping is curated — NEVER fuzzy-matched — per CLAUDE.md
invariant. Substring matching on AA's 525-row response would map
`llama-3-1-405b` to `hermes-4-llama-3-1-405b` (a Nous fine-tune,
NOT vanilla Llama). Verified live; this loader exists to make the
mapping explicit and queryable.

Shape:
  - `cp_slug` (str, required) — the ComputePrices slug we already
    track (joins our catalog)
  - `aa_slugs` (list, required, may be empty) — zero or more
    variants. Empty list means AA doesn't track this model under
    any known slug; the row stays for documentation + audit (the
    Llama-3.3-70B case from spec § Verification status).
  - `aa_slugs[].aa_slug` (str, required) — the slug as it appears
    in AA's response
  - `aa_slugs[].reasoning_effort` (low | medium | high | null) —
    explicit pairing for reasoning models that ship multiple rows
    per base model; null for non-reasoning models
  - `investigation_note` (str, optional) — free-form context for
    documented absences (Llama-3.3-70B), AA's per-model quirks, etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whatcanirun.catalog.loaders import SeedLoadError, load_aa_slug_mapping
from whatcanirun.pricing.aa_slug_mapping import (
    AaSlugMappingRow,
    AaSlugVariant,
    resolve_aa_slug,
)

# ----------------------------------------------------------- shape parsing


def test_loads_single_non_reasoning_mapping(tmp_path: Path) -> None:
    yaml = """\
- cp_slug: deepseek-v3
  aa_slugs:
    - aa_slug: deepseek-v3
      reasoning_effort: null
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    rows = load_aa_slug_mapping(f)
    assert len(rows) == 1
    row = rows[0]
    assert row.cp_slug == "deepseek-v3"
    assert len(row.aa_slugs) == 1
    assert row.aa_slugs[0].aa_slug == "deepseek-v3"
    assert row.aa_slugs[0].reasoning_effort is None
    assert row.investigation_note is None


def test_loads_reasoning_model_with_multiple_effort_levels(tmp_path: Path) -> None:
    """One CP slug may map to several AA rows when the model ships
    `-low`/`-medium`/`-high` reasoning variants. Test the full set
    parses + the (cp, aa, effort) triples are queryable."""
    yaml = """\
- cp_slug: gpt-oss-120b
  aa_slugs:
    - aa_slug: gpt-oss-120b-low
      reasoning_effort: low
    - aa_slug: gpt-oss-120b-medium
      reasoning_effort: medium
    - aa_slug: gpt-oss-120b-high
      reasoning_effort: high
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    rows = load_aa_slug_mapping(f)
    assert len(rows) == 1
    row = rows[0]
    assert len(row.aa_slugs) == 3
    triples = {(row.cp_slug, v.aa_slug, v.reasoning_effort) for v in row.aa_slugs}
    assert triples == {
        ("gpt-oss-120b", "gpt-oss-120b-low", "low"),
        ("gpt-oss-120b", "gpt-oss-120b-medium", "medium"),
        ("gpt-oss-120b", "gpt-oss-120b-high", "high"),
    }


def test_loads_explicit_absence_with_investigation_note(tmp_path: Path) -> None:
    """Per spec § Acceptance criteria: Llama-3.3-70B has no AA match
    today. We document the absence rather than silently leaving the
    slug out — empty `aa_slugs` + `investigation_note` makes the
    decision auditable, and M07 can route to Tier 3/4 without
    asking AA."""
    yaml = """\
- cp_slug: llama-3-3-70b
  aa_slugs: []
  investigation_note: "AA tracks this as `llama-3-3-instruct-70b` under a different vendor prefix; see Slice F."
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    rows = load_aa_slug_mapping(f)
    assert len(rows) == 1
    assert rows[0].aa_slugs == []
    assert rows[0].investigation_note is not None
    assert "llama-3-3-instruct-70b" in rows[0].investigation_note


# ----------------------------------------------------------- validation


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    """Curated supplements use `extra="forbid"` so typos in the
    YAML (e.g. `aa_slug` at the top level instead of inside
    `aa_slugs`) fail loudly. Mirrors the M01 supplement loader
    convention."""
    yaml = """\
- cp_slug: x
  aa_slugs: []
  aa_slug: typo-at-top-level
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError):
        load_aa_slug_mapping(f)


def test_invalid_reasoning_effort_value_rejected(tmp_path: Path) -> None:
    """`reasoning_effort` is a closed enum (low | medium | high |
    null). `extreme` or `medium-high` should fail validation, not
    silently route through to a Pydantic-coerced string."""
    yaml = """\
- cp_slug: x
  aa_slugs:
    - aa_slug: x-extreme
      reasoning_effort: extreme
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError):
        load_aa_slug_mapping(f)


def test_duplicate_cp_slug_rejected(tmp_path: Path) -> None:
    """Two rows sharing the same `cp_slug` is a typo / merge-conflict
    footgun — without detection, the second would silently shadow
    the first in any dict-keyed lookup. Catch at load time."""
    yaml = """\
- cp_slug: dup
  aa_slugs:
    - aa_slug: dup-a
      reasoning_effort: null
- cp_slug: dup
  aa_slugs:
    - aa_slug: dup-b
      reasoning_effort: null
"""
    f = tmp_path / "m.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError, match="duplicate"):
        load_aa_slug_mapping(f)


# -------------------------------------------------------- resolve_aa_slug


def test_resolve_returns_matching_variant_by_effort() -> None:
    """`resolve_aa_slug(rows, cp_slug, effort)` — the M07-facing
    helper. Returns the explicit (aa_slug, effort) pair that matches
    OR None when no variant pairs to that effort. Effort-as-None is
    a legitimate request (non-reasoning models)."""
    rows = [
        AaSlugMappingRow(
            cp_slug="gpt-oss-120b",
            aa_slugs=[
                AaSlugVariant(aa_slug="gpt-oss-120b-low", reasoning_effort="low"),
                AaSlugVariant(aa_slug="gpt-oss-120b-medium", reasoning_effort="medium"),
            ],
        ),
        AaSlugMappingRow(
            cp_slug="deepseek-v3",
            aa_slugs=[AaSlugVariant(aa_slug="deepseek-v3", reasoning_effort=None)],
        ),
    ]
    assert resolve_aa_slug(rows, "gpt-oss-120b", "low") == "gpt-oss-120b-low"
    assert resolve_aa_slug(rows, "gpt-oss-120b", "medium") == "gpt-oss-120b-medium"
    # High variant not registered → None (M07 falls through to Tier 3/4).
    assert resolve_aa_slug(rows, "gpt-oss-120b", "high") is None
    # Non-reasoning query against non-reasoning model.
    assert resolve_aa_slug(rows, "deepseek-v3", None) == "deepseek-v3"


def test_resolve_returns_none_when_cp_slug_absent() -> None:
    """A CP slug not in the mapping returns None — caller (M07)
    decides whether that's a route-to-Tier-3 case or an error.
    Better than raising because "AA doesn't track this" is the
    common case, not the exception."""
    rows = [
        AaSlugMappingRow(
            cp_slug="known",
            aa_slugs=[AaSlugVariant(aa_slug="known", reasoning_effort=None)],
        ),
    ]
    assert resolve_aa_slug(rows, "unknown-cp-slug", None) is None


def test_resolve_returns_none_for_explicit_absence() -> None:
    """A row with `aa_slugs=[]` (the Llama-3.3-70B case) resolves
    to None for any effort. This is the "documented absence" path
    distinct from "slug not in mapping at all", but both return
    None so M07 doesn't need to branch on the two cases — the
    investigation_note exists for human auditors, not the routing
    code."""
    rows = [
        AaSlugMappingRow(
            cp_slug="llama-3-3-70b",
            aa_slugs=[],
            investigation_note="AA tracks as llama-3-3-instruct-70b instead.",
        ),
    ]
    assert resolve_aa_slug(rows, "llama-3-3-70b", None) is None
    assert resolve_aa_slug(rows, "llama-3-3-70b", "low") is None
