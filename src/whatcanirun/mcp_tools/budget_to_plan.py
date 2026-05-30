"""M09 Slice F: `budget_to_plan` MCP tool — the headline.

Takes a dollar budget + model + workload profile and returns
ranked rows showing how the budget translates into hours of GPU
rental, total prompts purchasable, total output tokens, and
wall-clock time at the cell's decode TPS.

Each BudgetPlanRow carries the underlying CostCell, the derived
synthesis fields, and a trust envelope. The envelope has
`workload_assumption` populated per spec/SHARED.md — the
est_total_prompts figure is the textbook workload-derived number
the trust contract treats as needing explicit disclosure.

Slice M will route `workload_profile_slug=None` to a
WorkloadElicitationResponse rather than silently defaulting;
this slice covers the supplied-slug path.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.plan.cost_cells import CostCell
from whatcanirun.trust.envelope import ConfidenceDomain, TrustEnvelope


class BudgetPlanRow(BaseModel):
    """One row of `budget_to_plan` output. Wraps the underlying
    CostCell with budget-derived synthesis fields and a trust
    envelope carrying `workload_assumption`.

    `extra="forbid"` because this is an OWNED output type."""

    model_config = ConfigDict(extra="forbid")

    cost_cell: CostCell
    hours_available: float | None
    est_total_prompts: int
    est_total_output_tokens: int
    est_wallclock_minutes: float | None
    cost_per_m_output_usd: float
    trust_envelope: TrustEnvelope


def _per_prompt_cost(cell: CostCell, workload: WorkloadProfile) -> float | None:
    """Per-prompt cost on the cell's deployment mode, conditioned
    on the workload's token shape. Returns None when the cell
    has no rankable cost basis (same exclusion rule as
    find_cheapest_deployment)."""
    if cell.deployment_mode == "hosted_api_token":
        price_in = cell.price_per_m_input_usd
        price_out = cell.price_per_m_output_usd
        if price_in is None or price_out is None:
            return None
        return (
            workload.avg_input_tokens * price_in + workload.avg_output_tokens * price_out
        ) / 1_000_000
    rate = cell.cost_per_m_output_usd_self_hosted
    if rate is None:
        return None
    return workload.avg_output_tokens * rate / 1_000_000


def _rankable_cost_per_m_output(cell: CostCell) -> float | None:
    """The $/M output figure used for ascending sort. Hosted-API
    cells use their quoted price; cloud cells use the amortized
    self-hosted figure."""
    if cell.deployment_mode == "hosted_api_token":
        return cell.price_per_m_output_usd
    return cell.cost_per_m_output_usd_self_hosted


def _build_row(
    *,
    cell: CostCell,
    budget_usd: float,
    workload: WorkloadProfile,
) -> BudgetPlanRow | None:
    """Materialize one BudgetPlanRow from a CostCell + budget +
    workload. Returns None when the cell has no rankable cost
    basis (e.g. cloud cell with requires_measurement TPS)."""
    per_prompt = _per_prompt_cost(cell, workload)
    rate_per_m = _rankable_cost_per_m_output(cell)
    if per_prompt is None or per_prompt <= 0 or rate_per_m is None:
        return None

    est_total_prompts = math.floor(budget_usd / per_prompt)
    est_total_output_tokens = est_total_prompts * workload.avg_output_tokens

    hours_available: float | None
    if cell.deployment_mode == "hosted_api_token" or cell.hourly_usd is None:
        hours_available = None
    else:
        hours_available = budget_usd / cell.hourly_usd

    est_wallclock_minutes: float | None
    if cell.decode_tps is None or cell.decode_tps <= 0:
        est_wallclock_minutes = None
    else:
        est_wallclock_minutes = est_total_output_tokens / cell.decode_tps / 60.0

    # Build the row's envelope by carrying forward the cell's
    # confidence breakdown and adding `workload_assumption` per
    # spec/SHARED.md. The 0.95 value is the calibrated score for
    # a tool-arg-supplied workload profile (user-supplied custom
    # tokens would be 1.0; silent default would be 0.2 which M09
    # avoids by eliciting in Slice M).
    breakdown: dict[ConfidenceDomain, float] = dict(cell.trust_envelope.confidence_breakdown)
    breakdown["workload_assumption"] = 0.95

    envelope = TrustEnvelope(
        sources=list(cell.trust_envelope.sources),
        confidence_breakdown=breakdown,
        assumptions={
            **cell.trust_envelope.assumptions,
            "workload_profile": workload.slug,
            "avg_input_tokens": workload.avg_input_tokens,
            "avg_output_tokens": workload.avg_output_tokens,
            "budget_usd": budget_usd,
        },
        caveats=[
            *cell.trust_envelope.caveats,
            "est_total_prompts and est_wallclock_minutes are conditioned "
            "on the named workload_profile. Real traffic with a different "
            "(avg_input_tokens, avg_output_tokens) shape will yield "
            "different counts.",
        ],
        freshness=dict(cell.trust_envelope.freshness),
        verify_links=list(cell.trust_envelope.verify_links),
    )

    return BudgetPlanRow(
        cost_cell=cell,
        hours_available=hours_available,
        est_total_prompts=est_total_prompts,
        est_total_output_tokens=est_total_output_tokens,
        est_wallclock_minutes=est_wallclock_minutes,
        cost_per_m_output_usd=rate_per_m,
        trust_envelope=envelope,
    )


def build_budget_plan(
    *,
    budget_usd: float,
    cells: list[CostCell],
    workload: WorkloadProfile,
    top_n: int = 3,
) -> list[BudgetPlanRow]:
    """Pure builder. Produces ranked BudgetPlanRows from a list of
    CostCells.

    Excludes cells with no rankable cost basis. Sorts ascending by
    `cost_per_m_output_usd` (cheapest first). Caps the result at
    `top_n`. A zero budget returns an empty list (no rows make
    sense at $0); a negative budget raises ValueError.

    The wallclock figure uses decode-only TPS — prefill is typically
    much faster, so decode-time dominates total wallclock at
    production scales. A future enhancement could split the two.
    """
    if budget_usd < 0:
        raise ValueError(f"budget_usd must be non-negative; got {budget_usd}")
    if budget_usd == 0 or top_n <= 0:
        return []

    rows = [_build_row(cell=cell, budget_usd=budget_usd, workload=workload) for cell in cells]
    materialized = [r for r in rows if r is not None]
    materialized.sort(key=lambda r: r.cost_per_m_output_usd)
    return materialized[:top_n]


async def budget_to_plan(
    budget_usd: float,
    model_slug: str,
    workload_profile_slug: str | None = None,
    quant_slug: str | None = None,
    top_n: int = 3,
) -> list[BudgetPlanRow]:
    """`budget_to_plan` MCP tool entry point — the headline.

    Slug-resolution + cost-cells query + workload-elicitation
    routing live in Slice L + Slice M respectively. The pure
    builder `build_budget_plan` is testable today; end-to-end
    wiring lands when those slices ship.
    """
    raise NotImplementedError(
        "budget_to_plan slug-resolution + cost-cells query is wired "
        "in Slice L; the workload-elicitation routing is Slice M. "
        "The pure builder `build_budget_plan` is testable today."
    )
