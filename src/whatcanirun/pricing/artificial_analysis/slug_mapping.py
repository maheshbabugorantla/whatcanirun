"""Pydantic schema + resolution helpers for `seeds/aa_slug_mapping.yaml`.

The mapping is curated, NEVER fuzzy-matched (CLAUDE.md invariant):
substring matching `llama-3-1-405b` against AA's 525-row response
would match `hermes-4-llama-3-1-405b` — a Nous Research fine-tune
that is NOT vanilla Llama. Every CP slug → AA slug pairing must be
explicit, with `reasoning_effort` pinned to the right variant for
models that ship multiple effort levels.

Per CLAUDE.md "supplements use `extra="forbid"`" — YAML typos in
this file fail loudly rather than silently dropping unknown keys.

The resolution helper `resolve_aa_slug(rows, cp_slug, effort)` is
the M07-facing entry point. Returns None for both the "CP slug not
in mapping" case AND the "documented absence" case (Llama-3.3-70B
style empty `aa_slugs`) — M07 doesn't need to branch on which is
which; the `investigation_note` exists for human auditors.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from whatcanirun.pricing.artificial_analysis.projections import ReasoningEffort


class AaSlugVariant(BaseModel):
    """One AA slug paired to a specific reasoning effort.

    Models that ship multiple effort levels (e.g. `gpt-oss-120b-low`,
    `-medium`, `-high`) produce one variant per level. Non-reasoning
    models have a single variant with `reasoning_effort=None`.
    """

    model_config = ConfigDict(extra="forbid")

    aa_slug: str
    reasoning_effort: ReasoningEffort | None = None


class AaSlugMappingRow(BaseModel):
    """One row of `seeds/aa_slug_mapping.yaml` — a CP slug paired to
    zero or more AA variants.

    Empty `aa_slugs` is legitimate: it documents an absence (AA
    doesn't track this model under any known slug today). Combined
    with an `investigation_note` it gives auditors the context for
    why M07 routes this slug straight to Tier 3/4 instead of asking
    AA. See spec/M04 § Verification status (Llama-3.3-70B case).
    """

    model_config = ConfigDict(extra="forbid")

    cp_slug: str
    aa_slugs: list[AaSlugVariant] = Field(default_factory=list)
    investigation_note: str | None = None


def resolve_aa_slug(
    rows: Sequence[AaSlugMappingRow],
    cp_slug: str,
    reasoning_effort: ReasoningEffort | None,
) -> str | None:
    """Return the AA slug matching `(cp_slug, reasoning_effort)` or
    None when no variant pairs to that combination.

    "No match" covers three cases — all returning None so the M07
    routing code stays single-branch:
      1. `cp_slug` not present in the mapping at all
      2. `cp_slug` present but `aa_slugs` is empty (documented
         absence — Llama-3.3-70B style)
      3. `cp_slug` present, variants exist, but none pair to the
         requested `reasoning_effort` (e.g. asking for `high` when
         only `low`/`medium` are registered)
    """
    for row in rows:
        if row.cp_slug != cp_slug:
            continue
        for variant in row.aa_slugs:
            if variant.reasoning_effort == reasoning_effort:
                return variant.aa_slug
        return None
    return None
