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
from whatcanirun.mcp_tools.dispatch import UnknownModelResponse
from whatcanirun.plan.cost_cells import CostCell
from whatcanirun.trust.builders import build_deployment_comparison_envelope
from whatcanirun.trust.envelope import TrustEnvelope

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


def build_deployment_comparison(
    *,
    cloud_cell: CostCell | None,
    hosted_cell: CostCell | None,
    workload: WorkloadProfile,
    now: dt.datetime,
) -> DeploymentComparison:
    """Pure builder. Synthesizes per-prompt costs from the
    workload profile, picks the cheaper-per-prompt verdict, and
    wraps the whole thing in a trust envelope (built by
    `build_deployment_comparison_envelope` in trust/builders.py
    so the workload_assumption + verify_links handling stays in
    one place).

    `cloud_cell` and `hosted_cell` are nullable so the builder
    handles the partial-data case (Slice L's Case 2 dispatch
    routes to `UnknownModelResponse` at the tool boundary, but
    the pure builder still degrades gracefully)."""
    _ = now  # reserved for future freshness-confidence rollup
    cloud_per_prompt = _cost_per_prompt_cloud(cloud_cell, workload) if cloud_cell else None
    hosted_per_prompt = _cost_per_prompt_hosted(hosted_cell, workload) if hosted_cell else None

    envelope = build_deployment_comparison_envelope(
        cloud_cell=cloud_cell,
        hosted_cell=hosted_cell,
        workload=workload,
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
) -> DeploymentComparison | UnknownModelResponse:
    """`compare_deployment_modes` MCP tool entry point.

    Per spec/M09 § Case 2 + Tool-by-tool Case 2 behavior:
    compare_deployment_modes collapses Case 2 to Case 3 — its
    whole purpose is to compare cloud_gpu_rental vs
    hosted_api_token, and without architecture data the cloud
    side is impossible. A DeploymentComparison with
    cloud_gpu_rental=None would obscure the failure mode; better
    to return UnknownModelResponse and let the client elicit.
    """
    from datetime import UTC, datetime

    from whatcanirun.mcp_tools.deps import load_runtime_deps
    from whatcanirun.mcp_tools.dispatch import find_model_in_catalog
    from whatcanirun.plan.cost_cells import CostCellFilters, query_cost_cells

    deps = await load_runtime_deps()
    if find_model_in_catalog(model_slug, deps) is None:
        return UnknownModelResponse(requested_model_slug=model_slug)

    workload = next(
        (w for w in deps.workload_profiles if w.slug == workload_profile_slug),
        None,
    )
    if workload is None:
        raise LookupError(
            f"workload_profile_slug {workload_profile_slug!r} not found. "
            "Call list_catalog to see supported workload profiles."
        )

    cells = query_cost_cells(
        gpu_prices=deps.gpu_prices,
        llm_prices=deps.llm_prices,
        gpu_catalog=deps.gpu_catalog,
        model_catalog=deps.model_catalog,
        quantizations=deps.quantizations,
        bench_cells=deps.bench_cells,
        aa_observations=None,
        filters=CostCellFilters(
            model_slug=model_slug,
            gpu_slug=gpu_slug,
            quant_slug=quant_slug,
            batch_size=batch_size,
            context_length=context_length,
        ),
    )

    cloud_cell = next((c for c in cells if c.deployment_mode == "cloud_gpu_rental"), None)
    hosted_cell = next((c for c in cells if c.deployment_mode == "hosted_api_token"), None)

    return build_deployment_comparison(
        cloud_cell=cloud_cell,
        hosted_cell=hosted_cell,
        workload=workload,
        now=datetime.now(UTC),
    )
