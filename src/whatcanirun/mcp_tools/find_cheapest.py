"""M09 Slice D: `find_cheapest_deployment` MCP tool.

Ranks `CostCell` rows by per-token output cost and returns the
top_n. Mode-mixing is intentional — hosted-API token rates and
amortized cloud-GPU-rental $/M-output figures are directly
comparable for budget purposes, and the price-comparison
value-add is letting users see both kinds of options
side-by-side ranked by what they actually pay per million
output tokens.

Each row already carries its own `trust_envelope` (built by M08's
`query_cost_cells`). The tool does NOT add a wrapping envelope
around the list; the LLM client surfaces each row's envelope
individually. That matches how every per-row tool in this spec
treats its output — the list of CostCells IS the answer, and
each row's trust envelope IS the per-row provenance.
"""

from __future__ import annotations

from whatcanirun.mcp_tools.dispatch import UnknownModelResponse
from whatcanirun.plan.cost_cells import CostCell


def _rankable_cost_per_m_output(cell: CostCell) -> float | None:
    """Extract the cost-per-million-output-tokens figure that
    applies to this cell's deployment mode.

    - `hosted_api_token`: `price_per_m_output_usd` (the provider's
      quoted rate)
    - `cloud_gpu_rental`: `cost_per_m_output_usd_self_hosted`
      (M08's amortized GPU $/hr ÷ decode TPS calculation)

    Returns None when neither figure is populated — typically a
    `cloud_gpu_rental` cell whose TPS resolved to
    `requires_measurement`, leaving no basis for amortization.
    """
    if cell.deployment_mode == "hosted_api_token":
        return cell.price_per_m_output_usd
    return cell.cost_per_m_output_usd_self_hosted


def find_cheapest_deployments(
    cells: list[CostCell],
    *,
    top_n: int = 10,
) -> list[CostCell]:
    """Pure ranking function. Sort ascending by per-mode cost,
    exclude unrankable cells, cap at `top_n`.

    Cells with no rankable cost (e.g. `cloud_gpu_rental` with
    `tps_estimate.source=='requires_measurement'` AND no hosted
    quote either) are dropped silently — including them with a
    None key would either corrupt the sort or push them to the
    end where they'd be misleading. The user asked 'what's
    cheapest', not 'what's available'; M08's `query_cost_cells`
    is the right surface for the latter.

    Raises ValueError on `top_n < 0`; accepts `top_n=0` as a
    valid (empty-result) request.
    """
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0; got {top_n}")
    if top_n == 0:
        return []

    rankable = [(cell, _rankable_cost_per_m_output(cell)) for cell in cells]
    keyed = [(cell, cost) for (cell, cost) in rankable if cost is not None]
    keyed.sort(key=lambda pair: pair[1])
    return [cell for cell, _ in keyed[:top_n]]


async def find_cheapest_deployment(
    model_slug: str,
    quant_slug: str | None = None,
    batch_size: int = 1,
    context_length: int = 4096,
    region: str | None = None,
    top_n: int = 10,
) -> list[CostCell] | UnknownModelResponse:
    """`find_cheapest_deployment` MCP tool entry point.

    `region` is accepted for forward-compatibility but is a no-op
    in v1 — CP doesn't expose region per gpu-price row in a
    structured way.

    Per spec/M09 § Case 2: find_cheapest_deployment supports the
    partial-CostCell path for hosted_api_token rows when the model
    is in CP's catalog but not in our tracked-models set. The
    Case 2 branch builds per-provider partial CostCells via
    `build_case_2_partial_cells` and runs them through the same
    `find_cheapest_deployments` ranker as Case 1, so the wire
    shape (`list[CostCell]`) is identical across cases.
    """
    _ = region  # accepted for v2 forward-compat; v1 no-op
    from whatcanirun.mcp_tools.deps import load_runtime_deps
    from whatcanirun.mcp_tools.dispatch import (
        Case1Resolved,
        Case2HostedOnly,
        dispatch_model_request,
        model_catalog_with_resolved,
    )
    from whatcanirun.plan.cost_cells import CostCellFilters, query_cost_cells
    from whatcanirun.trust.builders import build_case_2_partial_cells

    deps = await load_runtime_deps()
    dispatched = await dispatch_model_request(model_slug, deps)
    if isinstance(dispatched, UnknownModelResponse):
        return dispatched
    # Case 2: CP-only model — build partial hosted_api_token CostCells
    # per spec/M09 § Tool-by-tool Case 2 behavior. Each LlmPriceRow
    # becomes one cell; the trust envelope carries
    # model_architecture=0.0 + the Case 2 caveat so the LLM client
    # surfaces the partial-data signal verbatim.
    if isinstance(dispatched, Case2HostedOnly):
        partial_cells = build_case_2_partial_cells(
            model_slug=model_slug,
            catalog_row=dispatched.catalog_row,
            prices=dispatched.prices,
            batch_size=batch_size,
            context_length=context_length,
            llm_prices_generated_at=deps.llm_prices_generated_at,
        )
        return find_cheapest_deployments(partial_cells, top_n=top_n)
    assert isinstance(dispatched, Case1Resolved)

    cells = query_cost_cells(
        gpu_prices=deps.gpu_prices,
        llm_prices=deps.llm_prices,
        gpu_catalog=deps.gpu_catalog,
        # `deps.model_catalog` was loaded before dispatch ran, so
        # Case 1b lazy-sync isn't reflected. Splice the resolved
        # model in so the just-synced row is visible.
        model_catalog=model_catalog_with_resolved(deps, dispatched.model),
        quantizations=deps.quantizations,
        bench_cells=deps.bench_cells,
        aa_observations=None,
        filters=CostCellFilters(
            model_slug=model_slug,
            quant_slug=quant_slug,
            batch_size=batch_size,
            context_length=context_length,
        ),
    )
    return find_cheapest_deployments(cells, top_n=top_n)
