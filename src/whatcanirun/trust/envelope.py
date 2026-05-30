"""`TrustEnvelope` + `Source` Pydantic shapes — the canonical
metadata wrapper every numerical tool output carries.

Per spec/SHARED.md § Trust contract: every number the MCP server
returns rides on a trust envelope with:
  - sources (each upstream that contributed)
  - confidence + per-domain confidence_breakdown (weakest-link)
  - assumptions (what was held fixed)
  - caveats (what we DO NOT model)
  - freshness (per-source last-updated)
  - verify_links (URLs the user can audit upstream)

M08 builds partial envelopes when constructing CostCells from
catalog + pricing + throughput data. M09's MCP tool layer
enriches them when wrapping responses (adding `workload_assumption`
when synthesizing derived counts, etc.).

`extra="forbid"` follows the convention for OWNED output types
(FitResult, TpsEstimate, WorkloadProfile, CostCell): a typo in
construction fails loudly rather than silently dropping a field
the LLM client would surface verbatim.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Per spec/SHARED.md the closed set of source names. Adding a
# new upstream (e.g. AWS Pricing API in v2) requires extending
# this Literal AND updating M09's trust-envelope assembly.
SourceName = Literal[
    "computeprices",
    "huggingface",
    "artificial_analysis",
    "own_measured_benchmark",
    "public_benchmark_anchor",
    "bandwidth_heuristic",
    "datasheet_yaml",
]

# Per spec/SHARED.md the closed set of confidence domains.
# `workload_assumption` appears ONLY on responses that synthesize
# derived counts from a workload profile — omit the key when no
# workload was assumed (don't include with a default value).
ConfidenceDomain = Literal[
    "pricing",
    "fit_check",
    "throughput",
    "model_architecture",
    "gpu_specs",
    "workload_assumption",
    "freshness",
]


class Source(BaseModel):
    """One upstream contribution to a `TrustEnvelope`. Carries
    the freshness signal and (when the upstream's license
    requires it) the verbatim attribution string."""

    model_config = ConfigDict(extra="forbid")

    name: SourceName
    detail: str
    last_updated: datetime
    license_attribution: str | None = None


class TrustEnvelope(BaseModel):
    """Trust-contract metadata wrapping a numerical tool output.
    `confidence` is computed as `min(confidence_breakdown.values())`
    — weakest-link by design. Aggregating via average or geometric
    mean would let a 0.0 'requires_measurement' domain be masked
    by high confidence elsewhere; that's the failure mode the
    weakest-link rule exists to prevent."""

    model_config = ConfigDict(extra="forbid")

    sources: list[Source]
    confidence_breakdown: dict[ConfidenceDomain, float]
    assumptions: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    freshness: dict[str, datetime] = Field(default_factory=dict)
    verify_links: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Weakest-link rollup of `confidence_breakdown`. Computed
        rather than stored so it can never drift from the
        breakdown — every Pydantic dump and every accessor sees
        the same derived value."""
        if not self.confidence_breakdown:
            return 0.0
        return min(self.confidence_breakdown.values())
