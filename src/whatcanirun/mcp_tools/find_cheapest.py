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
) -> list[CostCell]:
    """`find_cheapest_deployment` MCP tool entry point.

    Resolves the model + quant slugs against the local catalog
    caches, queries M08's cost-cells layer with the supplied
    op-point + region filter, then ranks via
    `find_cheapest_deployments`.

    The slug-resolution + cost-cells query is Slice L's job
    (unknown-model dispatcher + full cache plumbing). This stub
    keeps the registration discoverable while signaling the gap.

    `region` is accepted for forward-compatibility but is a no-op
    in v1 — CP doesn't expose region per gpu-price row in a
    structured way. v2 plumbing TBD.
    """
    raise NotImplementedError(
        "find_cheapest_deployment slug-resolution + cost-cells query "
        "is wired in Slice L. The pure ranker `find_cheapest_deployments` "
        "is testable independently."
    )
