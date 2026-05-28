"""Pydantic projection of an Artificial Analysis `/api/v2/data/llms/models`
row (one entry of the response's `data[]` array).

Per ADR-015 (Raw + Projection): the full upstream row lives in `raw`;
the typed fields below are the subset M07's Tier-2 anchor consumes
today. `evaluations` and `pricing` are deliberately `dict[str, float
| None]` — AA's Intelligence Index ships new evaluation keys every
few releases (live capture 2026-05-27 has 15 keys including
`aime_25`, `lcr`, `terminalbench_hard`, `tau2`, `ifbench`, `hle` —
none of which were in AA's published docs at spec-writing time).
Narrow-typing these would break ingest on every Intelligence Index
revision, exactly the trust-contract failure mode ADR-015 prevents.

`reasoning_effort` is intentionally None-by-default on the
projection's own auto-detection: many non-reasoning model slugs end
in `-medium` (`mistral-medium`, `magistral-medium`, `devstral-medium`)
and would be mis-classified by a regex on the suffix. The
`aa_slug_mapping.yaml` curated loader (M04 Slice E) supplies the
explicit pairing and passes the resolved value into
`AaModelRow.project(..., reasoning_effort=...)`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

ReasoningEffort = Literal["low", "medium", "high"]


class AaModelRow(BaseModel):
    """One row of AA's `data[]` array."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    # Identifiers — `id` (AA UUID) is the stable primary join key;
    # `slug` may change between releases. Both are required at the
    # boundary so downstream code never sees a None primary key.
    id: str
    slug: str
    name: str
    model_creator: dict[str, str]
    release_date: date | None = None

    # Reasoning effort dimension. The projection's own auto-detection
    # is intentionally None — slug-mapping loader stamps the resolved
    # value via `project(..., reasoning_effort=...)` to avoid
    # mis-classifying base-model slugs that end in `-medium`.
    reasoning_effort: ReasoningEffort | None = None

    # Throughput / latency — projected, used by M07 Tier-2 anchor.
    median_output_tokens_per_second: float | None = None
    median_time_to_first_token_seconds: float | None = None
    median_time_to_first_answer_token: float | None = None

    # Pricing — sub-keys evolve (cache, batch, tiered variants).
    pricing: dict[str, float | None] = Field(default_factory=dict)

    # Evaluations — schema explicitly evolving. NEVER narrow-typed.
    evaluations: dict[str, float | None] = Field(default_factory=dict)

    # Full upstream row, verbatim.
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def project(
        cls,
        payload: dict[str, Any],
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> Self:
        """Project an AA `data[]` row into the typed model.

        `raw` is set from a shallow copy of `payload` so future-added
        top-level fields survive even though they aren't typed.
        `reasoning_effort` is an explicit kwarg owned by the
        slug-mapping loader — the projection never tries to infer it
        from the slug suffix because base-model names like
        `mistral-medium` would be mis-classified.
        """
        projected = {**payload, "raw": dict(payload)}
        if reasoning_effort is not None:
            projected["reasoning_effort"] = reasoning_effort
        return cls.model_validate(projected)
