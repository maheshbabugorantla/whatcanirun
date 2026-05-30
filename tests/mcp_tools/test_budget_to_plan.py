"""M09 Slice F: `budget_to_plan` MCP tool — TDD (headline).

The headline. Takes a dollar budget + model + workload profile and
returns ranked rows showing how that budget translates into:

- `hours_available`: budget_usd / hourly_usd (None for hosted-API)
- `est_total_prompts`: prompts the budget buys at the workload's
  per-prompt cost
- `est_total_output_tokens`: prompts * avg_output_tokens
- `est_wallclock_minutes`: None when TPS is requires_measurement
- `cost_per_m_output_usd`: the underlying $/M output figure

Each row carries a trust envelope with `workload_assumption`
populated (per spec/M09 § Workload assumption handling).

Slice M will route `workload_profile_slug=None` to a
WorkloadElicitationResponse rather than silently defaulting; this
slice covers the supplied-slug happy path.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.inference.fit_check import FitResult
from whatcanirun.inference.tps_estimator import TpsEstimate
from whatcanirun.mcp_tools.budget_to_plan import (
    BudgetPlanRow,
    build_budget_plan,
)
from whatcanirun.plan.cost_cells import CostCell
from whatcanirun.trust.envelope import Source, TrustEnvelope

# ---------------------------------------------------------------- factories


def _envelope() -> TrustEnvelope:
    return TrustEnvelope(
        sources=[
            Source(
                name="computeprices",
                detail="test",
                last_updated=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
            )
        ],
        confidence_breakdown={"pricing": 0.95, "freshness": 0.95},
    )


def _tps(value: float | None) -> TpsEstimate:
    if value is None:
        return TpsEstimate(
            value=None,
            source="requires_measurement",
            confidence=0.0,
            refusal_reason="test",
        )
    return TpsEstimate(value=value, source="bandwidth_heuristic_single_stream", confidence=0.6)


def _fit() -> FitResult:
    return FitResult(
        fits=True,
        weight_gb=70.6,
        kv_cache_gb=2.0,
        framework_overhead_gb=10.6,
        total_required_gb=83.2,
        available_gb=80.0,
        headroom_gb=-3.2,
        blocking_reasons=[],
    )


def _gpu_cell(
    *,
    provider_slug: str = "deep-infra",
    hourly_usd: float = 2.50,
    cost_per_m_output: float | None = 0.50,
    tps_value: float | None = 120.0,
) -> CostCell:
    return CostCell(
        gpu_slug="h100sxm",
        provider_slug=provider_slug,
        model_slug="qwen-3-coder-30b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        deployment_mode="cloud_gpu_rental",
        hourly_usd=hourly_usd,
        pricing_type="on_demand",
        decode_tps=tps_value,
        tps_estimate=_tps(tps_value),
        fit_result=_fit(),
        cost_per_m_output_usd_self_hosted=cost_per_m_output,
        trust_envelope=_envelope(),
    )


def _hosted_cell(
    *,
    provider_slug: str = "openrouter",
    price_in: float = 0.20,
    price_out: float = 0.60,
) -> CostCell:
    return CostCell(
        gpu_slug=None,
        provider_slug=provider_slug,
        model_slug="qwen-3-coder-30b",
        quant_slug=None,
        tp_size=None,
        batch_size=1,
        context_length=4096,
        deployment_mode="hosted_api_token",
        price_per_m_input_usd=price_in,
        price_per_m_output_usd=price_out,
        tps_estimate=_tps(None),
        trust_envelope=_envelope(),
    )


def _chat_assistant() -> WorkloadProfile:
    return WorkloadProfile(
        slug="chat_assistant",
        display_name="Chat assistant",
        avg_input_tokens=500,
        avg_output_tokens=200,
        is_default=True,
        description="test",
    )


# ---------------------------------------------------------------- tests


def test_returns_list_of_budget_plan_rows() -> None:
    """Spec/M09 §5: `budget_to_plan` returns a list of
    BudgetPlanRow. The empty-input case still returns a list,
    not None — a no-rows result is different from a missing
    response."""
    rows = build_budget_plan(
        budget_usd=20.0, cells=[_gpu_cell()], workload=_chat_assistant(), top_n=3
    )
    assert isinstance(rows, list)
    assert all(isinstance(r, BudgetPlanRow) for r in rows)


def test_rows_sorted_ascending_by_cost_per_m_output() -> None:
    """Spec/M09 acceptance: '`budget_to_plan` golden path:
    `(budget_usd=20, model_slug='qwen-3-coder-30b')` returns ≥3
    BudgetPlanRow, sorted ASC by `cost_per_m_output_usd`'.
    The cheapest row is the best deal — the LLM client surfaces
    the top row as the headline."""
    cells = [
        _gpu_cell(provider_slug="provider-a", cost_per_m_output=0.80),
        _gpu_cell(provider_slug="provider-b", cost_per_m_output=0.20),
        _gpu_cell(provider_slug="provider-c", cost_per_m_output=0.50),
    ]
    rows = build_budget_plan(budget_usd=20.0, cells=cells, workload=_chat_assistant(), top_n=10)
    costs = [r.cost_per_m_output_usd for r in rows]
    assert costs == sorted(costs)


def test_hours_available_for_gpu_rental() -> None:
    """For `cloud_gpu_rental` rows, `hours_available = budget_usd
    / hourly_usd`. A $20 budget against $2.50/hour GPU should
    yield 8 hours."""
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[_gpu_cell(hourly_usd=2.50)],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    assert row.hours_available == pytest.approx(20.0 / 2.50)


def test_hours_available_none_for_hosted_api() -> None:
    """Per spec/M09 §5: `hours_available: float | None — null for
    hosted_api_token`. The concept doesn't apply: hosted-API has
    no hourly rental; the budget converts directly to prompts."""
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[_hosted_cell()],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    assert row.hours_available is None


def test_est_total_prompts_from_workload_and_per_prompt_cost() -> None:
    """`est_total_prompts = floor(budget_usd / per_prompt_cost)`.
    For a hosted cell at $0.20/$0.60 per 1M with a 500/200
    workload: per-prompt = (500*0.20 + 200*0.60)/1M = 0.00022.
    $20 / 0.00022 = ~90909 prompts."""
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[_hosted_cell(price_in=0.20, price_out=0.60)],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    per_prompt = (500 * 0.20 + 200 * 0.60) / 1_000_000
    assert row.est_total_prompts == int(20.0 / per_prompt)


def test_est_total_output_tokens_is_prompts_times_avg_out() -> None:
    """`est_total_output_tokens = est_total_prompts *
    workload.avg_output_tokens`. A direct multiplication
    consistency check — if a refactor introduces an off-by-one
    or unit conversion bug, this catches it."""
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[_hosted_cell()],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    assert row.est_total_output_tokens == row.est_total_prompts * 200


def test_est_wallclock_minutes_none_when_tps_unknown() -> None:
    """Per spec/M09 §5: `est_wallclock_minutes: float | None —
    null when throughput is requires_measurement`. Without a TPS
    figure the math can't produce a duration, so the field stays
    None rather than fabricating one."""
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[_hosted_cell()],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    assert row.est_wallclock_minutes is None


def test_est_wallclock_minutes_derived_from_tps_and_total_output() -> None:
    """`est_wallclock_minutes = est_total_output_tokens / decode_tps
    / 60`. The decode-only assumption is intentional and
    documented — prefill is typically much faster so decode time
    dominates total wallclock at production scales."""
    cell = _gpu_cell(tps_value=100.0)
    row = build_budget_plan(
        budget_usd=20.0,
        cells=[cell],
        workload=_chat_assistant(),
        top_n=10,
    )[0]
    expected_minutes = row.est_total_output_tokens / 100.0 / 60
    assert row.est_wallclock_minutes == pytest.approx(expected_minutes)


def test_envelope_includes_workload_assumption() -> None:
    """Per spec/SHARED.md every budget-to-plan row must carry
    `workload_assumption` in its trust envelope (the
    est_total_prompts figure is the textbook workload-derived
    number). Per spec/M09 § Workload assumption handling, this
    is exactly the domain the elicitation flow drives upward to
    0.95 from the default 0.2."""
    row = build_budget_plan(
        budget_usd=20.0, cells=[_hosted_cell()], workload=_chat_assistant(), top_n=3
    )[0]
    breakdown = row.trust_envelope.confidence_breakdown
    assert breakdown.get("workload_assumption") == pytest.approx(0.95)


def test_envelope_assumptions_name_the_workload_profile() -> None:
    """Per spec/M09 relay rule 6: surfacing the workload profile
    requires the envelope's `assumptions['workload_profile']`
    field. Without it the LLM client would silently relay the
    derived prompt count without disclosing what shape it
    assumed."""
    row = build_budget_plan(
        budget_usd=20.0, cells=[_hosted_cell()], workload=_chat_assistant(), top_n=3
    )[0]
    assert row.trust_envelope.assumptions["workload_profile"] == "chat_assistant"


def test_excludes_cells_with_no_rankable_cost() -> None:
    """A cell with no cost basis (requires_measurement on cloud,
    no hosted quote) can't be ranked or amortized. Same logic as
    find_cheapest_deployment — drop silently rather than emit a
    row with None figures."""
    cells = [
        _gpu_cell(provider_slug="good", cost_per_m_output=0.50),
        _gpu_cell(
            provider_slug="opaque",
            cost_per_m_output=None,
            tps_value=None,
        ),
    ]
    rows = build_budget_plan(budget_usd=20.0, cells=cells, workload=_chat_assistant(), top_n=10)
    assert len(rows) == 1
    assert rows[0].cost_cell.provider_slug == "good"


def test_top_n_caps_result_count() -> None:
    """`top_n` defaults to 3 in the spec. The function must honor
    arbitrary top_n values."""
    cells = [_gpu_cell(provider_slug=f"p{i}", cost_per_m_output=0.10 * (i + 1)) for i in range(7)]
    rows = build_budget_plan(budget_usd=20.0, cells=cells, workload=_chat_assistant(), top_n=3)
    assert len(rows) == 3


def test_zero_budget_returns_no_rows() -> None:
    """A zero budget buys zero prompts on any cell. The function
    must accept this and return an empty list rather than dividing
    by zero or returning negative prompts."""
    rows = build_budget_plan(
        budget_usd=0.0, cells=[_gpu_cell()], workload=_chat_assistant(), top_n=10
    )
    assert rows == []


def test_negative_top_n_raises_value_error() -> None:
    """Copilot review #15 round 4: `find_cheapest_deployments`
    raises on negative top_n; `build_budget_plan` was silently
    treating it as 'return []'. That inconsistency made it easy
    for client input bugs (sign error, off-by-one) to silently
    hide an empty plan. The function now matches the cheapest
    ranker: top_n=0 is a valid no-op poke; top_n<0 fails fast."""
    with pytest.raises(ValueError, match="top_n"):
        build_budget_plan(
            budget_usd=20.0,
            cells=[_gpu_cell()],
            workload=_chat_assistant(),
            top_n=-3,
        )


def test_zero_top_n_returns_empty_list() -> None:
    """`top_n=0` is a valid no-op pagination request — some
    clients use 0 as a 'just tell me whether anything exists' poke.
    Must NOT raise (that's the negative-top_n behavior); must
    return an empty list."""
    rows = build_budget_plan(
        budget_usd=20.0,
        cells=[_gpu_cell()],
        workload=_chat_assistant(),
        top_n=0,
    )
    assert rows == []


def test_negative_budget_raises_value_error() -> None:
    """A negative budget is nonsensical and would produce
    negative prompt counts. Fail fast at the function boundary."""
    with pytest.raises(ValueError, match="budget"):
        build_budget_plan(
            budget_usd=-5.0,
            cells=[_gpu_cell()],
            workload=_chat_assistant(),
            top_n=10,
        )


def test_budget_to_plan_registered_as_mcp_tool(_: Any = None) -> None:
    """Registration smoke test."""
    import asyncio

    from whatcanirun.server import mcp

    tools = asyncio.run(mcp.get_tools())
    assert "budget_to_plan" in tools, (
        f"`budget_to_plan` tool not registered on `mcp`; registered tools: {sorted(tools)}"
    )
