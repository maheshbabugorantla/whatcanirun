"""M09 Slice E: `compare_deployment_modes` MCP tool.

Side-by-side comparison of `cloud_gpu_rental` vs
`hosted_api_token` for one op-point, conditioned on a workload
profile. The per-prompt cost figure is derived from
`(avg_input_tokens, avg_output_tokens)`, so the wrapping trust
envelope carries `workload_assumption` per spec/SHARED.md.

Per-side CostCells are preserved (each with its own
trust_envelope) so an LLM client can read through to the
mode-specific provenance. The DeploymentComparison-level envelope
is for the synthesized comparison (per-prompt cost + verdict).
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.plan.cost_cells import CostCell
from whatcanirun.trust.envelope import (
    ConfidenceDomain,
    Source,
    SourceName,
    TrustEnvelope,
)

Verdict = Literal["cloud_gpu_rental", "hosted_api_token", "tie", "unknown"]

# Per-prompt cost equality tolerance — the "tie" verdict applies
# when the two costs are within this fraction of each other. A
# 5% band avoids declaring one side a winner over rounding noise
# at low absolute volumes.
_TIE_FRACTION = 0.05


class DeploymentComparison(BaseModel):
    """Output shape for `compare_deployment_modes`. Pydantic so
    the FastMCP wire schema is stable; `extra="forbid"` because
    this is an OWNED output type."""

    model_config = ConfigDict(extra="forbid")

    workload_profile_slug: str
    cloud_gpu_rental: CostCell | None
    hosted_api_token: CostCell | None
    cost_per_prompt_cloud_usd: float | None
    cost_per_prompt_hosted_usd: float | None
    cheaper_per_prompt: Verdict
    trust_envelope: TrustEnvelope


def _cost_per_prompt_cloud(cell: CostCell, workload: WorkloadProfile) -> float | None:
    """Per-prompt cost on the self-hosted side. The
    `cost_per_m_output_usd_self_hosted` figure already amortizes
    the GPU hourly rate over decode TPS, so the per-prompt math
    is just `avg_output_tokens * $/M_output / 1_000_000`. Input
    tokens are free at decode time on self-hosted (prefill cost
    is folded into the hourly amortization)."""
    rate = cell.cost_per_m_output_usd_self_hosted
    if rate is None:
        return None
    return workload.avg_output_tokens * rate / 1_000_000


def _cost_per_prompt_hosted(cell: CostCell, workload: WorkloadProfile) -> float | None:
    """Per-prompt cost on the hosted-API side. Input AND output
    tokens are billed separately at the provider's per-million
    rates."""
    price_in = cell.price_per_m_input_usd
    price_out = cell.price_per_m_output_usd
    if price_in is None or price_out is None:
        return None
    return (
        workload.avg_input_tokens * price_in + workload.avg_output_tokens * price_out
    ) / 1_000_000


def _verdict(cloud_cost: float | None, hosted_cost: float | None) -> Verdict:
    """Bottom-line comparison the LLM client relays in one
    sentence. `tie` covers the small region where the two costs
    are within `_TIE_FRACTION` of each other (5%); `unknown`
    covers the partial-data case (one or both costs missing)."""
    if cloud_cost is None or hosted_cost is None:
        return "unknown"
    if cloud_cost <= 0 or hosted_cost <= 0:
        return "unknown"
    ratio = cloud_cost / hosted_cost
    if abs(ratio - 1.0) <= _TIE_FRACTION:
        return "tie"
    return "cloud_gpu_rental" if cloud_cost < hosted_cost else "hosted_api_token"


def _merge_sources(*cells: CostCell | None) -> list[Source]:
    """Dedup sources across input CostCells by `(name, detail)`
    so the wrapping envelope doesn't duplicate the same upstream
    contribution. Order preserved (first-seen wins)."""
    seen: dict[tuple[SourceName, str], Source] = {}
    for cell in cells:
        if cell is None:
            continue
        for src in cell.trust_envelope.sources:
            seen.setdefault((src.name, src.detail), src)
    return list(seen.values())


def _merge_freshness(*cells: CostCell | None) -> dict[str, dt.datetime]:
    """Min-by-source freshness rollup across input cells. If two
    cells share a source, the older timestamp wins (weakest-link
    semantics propagate to the freshness map too)."""
    merged: dict[str, dt.datetime] = {}
    for cell in cells:
        if cell is None:
            continue
        for source_name, ts in cell.trust_envelope.freshness.items():
            if source_name not in merged or ts < merged[source_name]:
                merged[source_name] = ts
    return merged


def build_deployment_comparison(
    *,
    cloud_cell: CostCell | None,
    hosted_cell: CostCell | None,
    workload: WorkloadProfile,
    now: dt.datetime,
) -> DeploymentComparison:
    """Pure builder. Synthesizes per-prompt costs from the
    workload profile, picks the cheaper-per-prompt verdict, and
    wraps the whole thing in a trust envelope that includes
    `workload_assumption` (per spec/SHARED.md — workload-derived
    figures must carry that domain).

    `cloud_cell` and `hosted_cell` are nullable so the builder
    handles the partial-data case (Slice L's Case 2 dispatch
    routes to `UnknownModelResponse` at the tool boundary, but
    the pure builder still degrades gracefully)."""
    _ = now  # reserved for future freshness-confidence rollup
    cloud_per_prompt = _cost_per_prompt_cloud(cloud_cell, workload) if cloud_cell else None
    hosted_per_prompt = _cost_per_prompt_hosted(hosted_cell, workload) if hosted_cell else None

    breakdown: dict[ConfidenceDomain, float] = {
        # Per spec/SHARED.md § Calibration: tool-arg-supplied
        # workload profile = 0.95 (only `own_measured` style
        # user-supplied tokens reach 1.0; defaulted = 0.2 — which
        # M09 elicits to avoid).
        "workload_assumption": 0.95,
    }

    # Carry forward the worst per-side confidence per shared
    # domain so the wrapping envelope inherits the weakest link.
    for cell in (cloud_cell, hosted_cell):
        if cell is None:
            continue
        for domain, score in cell.trust_envelope.confidence_breakdown.items():
            existing = breakdown.get(domain)
            breakdown[domain] = min(existing, score) if existing is not None else score

    envelope = TrustEnvelope(
        sources=_merge_sources(cloud_cell, hosted_cell),
        confidence_breakdown=breakdown,
        assumptions={
            "workload_profile": workload.slug,
            "avg_input_tokens": workload.avg_input_tokens,
            "avg_output_tokens": workload.avg_output_tokens,
        },
        caveats=[
            "Per-prompt cost is conditioned on the workload profile's "
            "(avg_input_tokens, avg_output_tokens). Real traffic with "
            "different token shape will produce different per-prompt cost."
        ],
        freshness=_merge_freshness(cloud_cell, hosted_cell),
        verify_links=[],
    )

    return DeploymentComparison(
        workload_profile_slug=workload.slug,
        cloud_gpu_rental=cloud_cell,
        hosted_api_token=hosted_cell,
        cost_per_prompt_cloud_usd=cloud_per_prompt,
        cost_per_prompt_hosted_usd=hosted_per_prompt,
        cheaper_per_prompt=_verdict(cloud_per_prompt, hosted_per_prompt),
        trust_envelope=envelope,
    )


async def compare_deployment_modes(
    model_slug: str,
    gpu_slug: str,
    quant_slug: str,
    batch_size: int,
    context_length: int,
    workload_profile_slug: str,
) -> DeploymentComparison:
    """`compare_deployment_modes` MCP tool entry point.

    Slug-resolution + cost-cells query is Slice L's job. The
    pure builder `build_deployment_comparison` is testable today;
    end-to-end wiring lands with the unknown-model dispatcher.
    """
    raise NotImplementedError(
        "compare_deployment_modes slug-resolution + cost-cells query "
        "is wired in Slice L (unknown-model dispatcher)."
    )
