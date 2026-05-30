"""M09 Slice D: `find_cheapest_deployment` MCP tool — TDD.

The tool ranks CostCell rows by the per-token output cost, returning
the top_n. The deliberate mode-mixing in the result list is what
makes this useful: a user asking "what's cheapest for Llama-3.3-70B
at this op-point?" gets the cheapest hosted-API token rate AND the
cheapest cloud-GPU-rental amortized $/M-output side-by-side, ranked
in the same list.

Cost extraction is mode-dependent: hosted_api_token cells carry the
quote in `price_per_m_output_usd`; cloud_gpu_rental cells carry the
amortized figure in `cost_per_m_output_usd_self_hosted`. Cells with
neither (e.g. `tps_estimate.source=='requires_measurement'` and no
hosted-API quote either) can't be ranked and are excluded from the
output rather than emitted with a None ranking.

Each row already carries its own `trust_envelope` (built by M08's
`query_cost_cells`); the tool doesn't construct a wrapper envelope
— a `list[CostCell]` IS the trust-envelope-aware return because
the LLM client surfaces each row's envelope individually.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from whatcanirun.inference.fit_check import FitResult
from whatcanirun.inference.tps_estimator import TpsEstimate
from whatcanirun.mcp_tools.find_cheapest import find_cheapest_deployments
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


def _tps_value(value: float) -> TpsEstimate:
    return TpsEstimate(
        value=value,
        source="bandwidth_heuristic_single_stream",
        confidence=0.6,
    )


def _tps_requires_measurement() -> TpsEstimate:
    return TpsEstimate(
        value=None,
        source="requires_measurement",
        confidence=0.0,
        refusal_reason="batch>1 in v1",
    )


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
    gpu_slug: str = "h100sxm",
    provider_slug: str = "deep-infra",
    cost_per_m_output: float | None = 0.50,
    tps: TpsEstimate | None = None,
) -> CostCell:
    return CostCell(
        gpu_slug=gpu_slug,
        provider_slug=provider_slug,
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        deployment_mode="cloud_gpu_rental",
        hourly_usd=2.50,
        pricing_type="on_demand",
        decode_tps=120.0,
        tps_estimate=tps or _tps_value(120.0),
        fit_result=_fit(),
        cost_per_m_output_usd_self_hosted=cost_per_m_output,
        trust_envelope=_envelope(),
    )


def _hosted_cell(
    *,
    provider_slug: str = "openrouter",
    price_per_m_output: float = 0.30,
) -> CostCell:
    return CostCell(
        gpu_slug=None,
        provider_slug=provider_slug,
        model_slug="llama-3-3-70b",
        quant_slug=None,
        tp_size=None,
        batch_size=1,
        context_length=4096,
        deployment_mode="hosted_api_token",
        price_per_m_input_usd=0.10,
        price_per_m_output_usd=price_per_m_output,
        tps_estimate=_tps_requires_measurement(),
        trust_envelope=_envelope(),
    )


# ---------------------------------------------------------------- tests


def test_returns_cells_sorted_ascending_by_cost_per_m_output() -> None:
    """Spec/M09 §2: 'find_cheapest_deployment ... the basic
    price-comparison tool. No budget; just what's cheapest for
    this op-point?' Ascending by per-token cost is the only
    sensible ordering — the first row IS the answer to 'what's
    cheapest'."""
    cells = [
        _gpu_cell(provider_slug="provider-a", cost_per_m_output=0.80),
        _gpu_cell(provider_slug="provider-b", cost_per_m_output=0.20),
        _gpu_cell(provider_slug="provider-c", cost_per_m_output=0.50),
    ]
    ranked = find_cheapest_deployments(cells, top_n=10)
    # `find_cheapest_deployments` drops None-cost rows so every
    # cell in `ranked` carries a non-None self-hosted figure.
    costs: list[float] = []
    for cell in ranked:
        assert cell.cost_per_m_output_usd_self_hosted is not None
        costs.append(cell.cost_per_m_output_usd_self_hosted)
    assert costs == sorted(costs)


def test_mixes_cloud_and_hosted_modes_in_one_ranking() -> None:
    """The result list is intentionally mode-mixed: hosted-API
    token rates and amortized GPU-rental $/M-output figures are
    DIRECTLY comparable for budget purposes (they're both
    `$ per million output tokens`). Returning them side-by-side
    is the price-comparison value-add."""
    cells = [
        _gpu_cell(cost_per_m_output=0.50),
        _hosted_cell(price_per_m_output=0.30),
        _gpu_cell(provider_slug="other", cost_per_m_output=0.70),
    ]
    ranked = find_cheapest_deployments(cells, top_n=10)
    modes = [c.deployment_mode for c in ranked]
    assert "cloud_gpu_rental" in modes
    assert "hosted_api_token" in modes
    # The hosted-API row at 0.30 must outrank the cheapest GPU at 0.50.
    assert ranked[0].deployment_mode == "hosted_api_token"


def test_top_n_caps_result_count() -> None:
    """`top_n` is the spec's pagination knob. A request with
    `top_n=3` against 5 cells returns 3, not the full set."""
    cells = [_gpu_cell(provider_slug=f"p{i}", cost_per_m_output=0.10 * (i + 1)) for i in range(5)]
    ranked = find_cheapest_deployments(cells, top_n=3)
    assert len(ranked) == 3


def test_excludes_cells_with_no_rankable_cost() -> None:
    """Cells with neither a hosted-API $/M-output rate nor an
    amortized self-hosted $/M-output figure (e.g. `tps_estimate
    .source=='requires_measurement'` AND no hosted quote) can't
    be ranked. Including them with a None key would corrupt the
    sort or push them to the end where they're misleading. The
    correct behavior is to drop them silently — the user asked
    'what's cheapest', not 'what's available'."""
    cells = [
        _gpu_cell(cost_per_m_output=0.50),
        # GPU cell where TPS is unknown -> cost_per_m_output_self_hosted=None
        _gpu_cell(
            provider_slug="opaque-provider",
            cost_per_m_output=None,
            tps=_tps_requires_measurement(),
        ),
        _hosted_cell(price_per_m_output=0.30),
    ]
    ranked = find_cheapest_deployments(cells, top_n=10)
    assert len(ranked) == 2
    assert all(
        (c.cost_per_m_output_usd_self_hosted is not None) or (c.price_per_m_output_usd is not None)
        for c in ranked
    )


def test_each_row_carries_its_own_trust_envelope() -> None:
    """The trust contract is satisfied per-row, not per-result-list.
    Each CostCell carries the envelope built by M08's
    `query_cost_cells`; the tool does NOT add a wrapping envelope
    around the list (the list is just an ordering of trust-aware
    rows). A regression that strips the per-row envelopes fails
    here."""
    cells = [_gpu_cell(cost_per_m_output=0.50), _hosted_cell(price_per_m_output=0.30)]
    ranked = find_cheapest_deployments(cells, top_n=10)
    for cell in ranked:
        assert isinstance(cell.trust_envelope, TrustEnvelope)
        assert cell.trust_envelope.sources  # non-empty


def test_empty_input_returns_empty_list() -> None:
    """Defensive: M08's `query_cost_cells` returns `[]` when the
    filters exclude every candidate. The tool must surface that
    as an empty result, not raise."""
    ranked = find_cheapest_deployments([], top_n=10)
    assert ranked == []


def test_zero_top_n_returns_empty_list(_: Any = None) -> None:
    """`top_n=0` is a no-op pagination request. The function
    must accept it without raising — some clients use 0 as a
    'just tell me whether anything exists' poke."""
    cells = [_gpu_cell(cost_per_m_output=0.50)]
    ranked = find_cheapest_deployments(cells, top_n=0)
    assert ranked == []


def test_negative_top_n_raises_value_error() -> None:
    """`top_n=-3` is nonsensical. Fail fast at the function
    boundary rather than producing a confusingly empty or
    reversed result downstream."""
    cells = [_gpu_cell(cost_per_m_output=0.50)]
    with pytest.raises(ValueError, match="top_n"):
        find_cheapest_deployments(cells, top_n=-3)


def test_find_cheapest_deployment_registered_as_mcp_tool() -> None:
    """The tool must be wired into the FastMCP instance. A
    registration regression fails here rather than at MCP-client
    connection time."""
    import asyncio

    from whatcanirun.server import mcp

    tools = asyncio.run(mcp.get_tools())
    assert "find_cheapest_deployment" in tools, (
        f"`find_cheapest_deployment` tool not registered on `mcp`; "
        f"registered tools: {sorted(tools)}"
    )
