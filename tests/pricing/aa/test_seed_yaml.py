"""Slice F: smoke-tests for the populated `seeds/aa_slug_mapping.yaml`.

These verify the seed file (1) loads without error, (2) covers every
slug in `seeds/tracked_models.yaml`, and (3) every non-empty mapping
resolves to a real row in the live AA fixture. Catches the failure
mode where someone edits tracked_models.yaml to add a new slug but
forgets to add the corresponding AA mapping — without this test,
M07's Tier-2 routing would silently fall through to Tier 3 for the
new model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whatcanirun.catalog.loaders import (
    load_aa_slug_mapping,
    load_tracked_models,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEEDS = _REPO_ROOT / "seeds"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def test_seed_yaml_loads_without_error() -> None:
    """The populated `seeds/aa_slug_mapping.yaml` must parse via the
    loader — no orphan typos, no schema drift between the file and
    `AaSlugMappingRow`."""
    rows = load_aa_slug_mapping(_SEEDS / "aa_slug_mapping.yaml")
    assert len(rows) > 0


def test_every_tracked_model_has_an_aa_mapping_row() -> None:
    """`seeds/tracked_models.yaml` and `seeds/aa_slug_mapping.yaml`
    must stay in lockstep. Every CP slug we sync HF metadata for
    needs at least an explicit mapping row — even if `aa_slugs` is
    empty (documented absence). Without this gate, adding a model
    to tracked_models would silently leave M07's Tier-2 path with
    nothing to look up."""
    tracked = load_tracked_models(_SEEDS / "tracked_models.yaml")
    mapping = load_aa_slug_mapping(_SEEDS / "aa_slug_mapping.yaml")
    mapped_cp_slugs = {row.cp_slug for row in mapping}
    missing = sorted({t.slug for t in tracked} - mapped_cp_slugs)
    assert not missing, (
        f"tracked_models.yaml has {missing!r} but aa_slug_mapping.yaml "
        f"doesn't. Add a mapping row (empty `aa_slugs` is fine — "
        f"document the absence in `investigation_note`)."
    )


def test_every_aa_slug_exists_in_the_live_fixture() -> None:
    """Each populated `aa_slug` in the mapping must correspond to a
    real row in AA's live response. A typo'd mapping would route
    `resolve_aa_slug` to a string AA's projection list doesn't
    contain, and M07 would silently get None at lookup time —
    classified as "AA doesn't track this" when really it's "we
    typo'd the slug".

    Loads the captured 2026-05-27 fixture and asserts every mapped
    `aa_slug` is in the set. New AA slugs added after the capture
    won't trigger this (they don't exist in the fixture yet, so
    they're allowed); the test is for regression on what was true
    at capture time."""
    fixture = json.loads((_FIXTURES / "aa_models_2026-05-27.json").read_text())
    live_slugs = {row["slug"] for row in fixture["data"]}

    mapping = load_aa_slug_mapping(_SEEDS / "aa_slug_mapping.yaml")
    typos = []
    for row in mapping:
        for variant in row.aa_slugs:
            if variant.aa_slug not in live_slugs:
                typos.append((row.cp_slug, variant.aa_slug))
    assert not typos, (
        f"aa_slug_mapping.yaml references slugs that aren't in the "
        f"2026-05-27 live fixture: {typos!r}. Either re-capture the "
        f"fixture or fix the typo."
    )


@pytest.mark.parametrize(
    ("cp_slug", "expected_aa_slug"),
    [
        # Llama 3.3 70B — the spec § Verification status documented
        # this as absent. Slice F's investigation found it under the
        # Meta-prefixed slug `llama-3-3-instruct-70b`. Pin the
        # resolved mapping so a future seed edit can't silently
        # revert to "documented absence" without an explicit reason.
        ("llama-3-3-70b", "llama-3-3-instruct-70b"),
        ("deepseek-v3", "deepseek-v3"),
        # Note: AA spells Mixtral's MoE family with `mistral-` prefix.
        ("mixtral-8x22b", "mistral-8x22b-instruct"),
    ],
)
def test_known_cp_slugs_resolve_to_their_documented_aa_slugs(
    cp_slug: str, expected_aa_slug: str
) -> None:
    """Lock the (cp, aa) resolutions discovered during Slice F so a
    drive-by edit to the YAML can't quietly re-introduce the
    Llama-3.3-70B-style mystery."""
    from whatcanirun.pricing.aa_slug_mapping import resolve_aa_slug

    mapping = load_aa_slug_mapping(_SEEDS / "aa_slug_mapping.yaml")
    assert resolve_aa_slug(mapping, cp_slug, None) == expected_aa_slug
