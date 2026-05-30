"""M09 Slice E: `compare_deployment_modes` MCP tool — TDD.

The tool takes one op-point + workload profile and returns a
side-by-side of `cloud_gpu_rental` vs `hosted_api_token`. Each
side carries its CostCell (with its own trust envelope) plus a
synthesized per-prompt cost figure conditioned on the workload
profile. The DeploymentComparison-level trust envelope adds
`workload_assumption` because the per-prompt cost is a derived
number from `(workload.avg_input_tokens, workload.avg_output_tokens)`.

`cheaper_per_prompt` is the bottom-line verdict the LLM client
relays in one sentence. "tie" handles the small region where the
two figures are close enough not to be decisively different;
"unknown" handles the partial-data case (one side missing or
both sides at None cost).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.inference.fit_check import FitResult
from whatcanirun.inference.tps_estimator import TpsEstimate
from whatcanirun.mcp_tools.compare_deployment import (
    DeploymentComparison,
    build_deployment_comparison,
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


def _tps_value(value: float) -> TpsEstimate:
    return TpsEstimate(
        value=value,
        source="bandwidth_heuristic_single_stream",
        confidence=0.6,
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


def _cloud_cell(*, cost_per_m_output: float = 0.50) -> CostCell:
    return CostCell(
        gpu_slug="h100sxm",
        provider_slug="deep-infra",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        deployment_mode="cloud_gpu_rental",
        hourly_usd=2.50,
        pricing_type="on_demand",
        decode_tps=120.0,
        tps_estimate=_tps_value(120.0),
        fit_result=_fit(),
        cost_per_m_output_usd_self_hosted=cost_per_m_output,
        trust_envelope=_envelope(),
    )


def _hosted_cell(*, price_in: float = 0.20, price_out: float = 0.60) -> CostCell:
    return CostCell(
        gpu_slug=None,
        provider_slug="openrouter",
        model_slug="llama-3-3-70b",
        quant_slug=None,
        tp_size=None,
        batch_size=1,
        context_length=4096,
        deployment_mode="hosted_api_token",
        price_per_m_input_usd=price_in,
        price_per_m_output_usd=price_out,
        tps_estimate=TpsEstimate(value=None, source="requires_measurement", confidence=0.0),
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


def _now() -> dt.datetime:
    return dt.datetime(2026, 5, 28, 2, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------- tests


def test_returns_typed_deployment_comparison() -> None:
    """The pure builder returns a `DeploymentComparison` Pydantic
    so the FastMCP wire schema is stable. A future refactor that
    returns a plain dict fails here."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(),
        hosted_cell=_hosted_cell(),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert isinstance(out, DeploymentComparison)


def test_per_prompt_costs_derived_from_workload() -> None:
    """Per-prompt cost on the hosted side is
    `(avg_in * price_in + avg_out * price_out) / 1_000_000`. Per-
    prompt cost on the cloud side is
    `(avg_out * cost_per_m_output_self_hosted) / 1_000_000`
    (input tokens are free at decode-time on self-hosted; the
    GPU $/hr already amortizes prefill time)."""
    cloud = _cloud_cell(cost_per_m_output=0.50)
    hosted = _hosted_cell(price_in=0.20, price_out=0.60)
    wl = _chat_assistant()  # 500 in, 200 out

    out = build_deployment_comparison(
        cloud_cell=cloud,
        hosted_cell=hosted,
        workload=wl,
        now=_now(),
    )

    # cloud: 200 * 0.50 / 1_000_000 = 0.0001
    assert out.cost_per_prompt_cloud_usd == pytest.approx(200 * 0.50 / 1_000_000)
    # hosted: (500 * 0.20 + 200 * 0.60) / 1_000_000 = (100 + 120)/1M
    assert out.cost_per_prompt_hosted_usd == pytest.approx((500 * 0.20 + 200 * 0.60) / 1_000_000)


def test_cheaper_per_prompt_verdict_picks_lower_cost() -> None:
    """`cheaper_per_prompt` is the bottom-line one-word verdict
    the LLM client relays. With hosted at $0.20/$0.60 per 1M for
    a 500/200 workload and cloud at $0.50/1M output, the cloud
    side is much cheaper ($0.0001 vs $0.00022)."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(cost_per_m_output=0.50),
        hosted_cell=_hosted_cell(price_in=0.20, price_out=0.60),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert out.cheaper_per_prompt == "cloud_gpu_rental"


def test_cheaper_per_prompt_handles_hosted_winning() -> None:
    """Inversely, an expensive cloud GPU + cheap hosted API
    flips the verdict. Confirm the comparator isn't hard-wired."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(cost_per_m_output=10.0),
        hosted_cell=_hosted_cell(price_in=0.01, price_out=0.05),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert out.cheaper_per_prompt == "hosted_api_token"


def test_cheaper_per_prompt_is_unknown_when_either_side_missing() -> None:
    """Spec/M09 Case 2: hosted-only data → partial CostCell with
    `requires_measurement` on the self-hosted side. The
    comparison tool falls through to Slice L's
    UnknownModelResponse for that case at the tool-router layer,
    but the pure builder must still degrade gracefully when only
    one side has data — return `unknown` rather than crash."""
    out = build_deployment_comparison(
        cloud_cell=None,
        hosted_cell=_hosted_cell(),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert out.cheaper_per_prompt == "unknown"


def test_envelope_includes_workload_assumption_domain() -> None:
    """The per-prompt cost is a derived figure from
    `(avg_input_tokens, avg_output_tokens)`. Per spec/SHARED.md
    the `workload_assumption` domain MUST appear in the trust
    envelope's breakdown for any tool that synthesizes such a
    derived figure."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(),
        hosted_cell=_hosted_cell(),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert "workload_assumption" in out.trust_envelope.confidence_breakdown


def test_envelope_assumptions_name_the_workload_profile() -> None:
    """Per spec/M09 relay rule 6: 'When
    `confidence_breakdown.workload_assumption` is present, ALWAYS
    surface the assumed workload profile from
    `assumptions["workload_profile"]`'. The envelope must carry
    the slug under that exact key so the relay rule kicks in."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(),
        hosted_cell=_hosted_cell(),
        workload=_chat_assistant(),
        now=_now(),
    )
    assert out.trust_envelope.assumptions.get("workload_profile") == "chat_assistant"


def test_envelope_confidence_is_weakest_link() -> None:
    """Computed-property check — `confidence` is min of breakdown
    values. A future refactor that hand-stores it would drift
    silently; this assertion catches the drift at the trust-
    envelope edge."""
    out = build_deployment_comparison(
        cloud_cell=_cloud_cell(),
        hosted_cell=_hosted_cell(),
        workload=_chat_assistant(),
        now=_now(),
    )
    breakdown = out.trust_envelope.confidence_breakdown
    assert out.trust_envelope.confidence == min(breakdown.values())


def test_both_cells_preserved_with_their_own_envelopes() -> None:
    """The per-side CostCells retain their own trust envelopes
    even though the DeploymentComparison-level envelope wraps the
    synthesized comparison. An LLM client that wants to surface
    the per-side breakdown reads through to the CostCell envelopes;
    a regression that strips them fails here."""
    cloud = _cloud_cell()
    hosted = _hosted_cell()
    out = build_deployment_comparison(
        cloud_cell=cloud,
        hosted_cell=hosted,
        workload=_chat_assistant(),
        now=_now(),
    )
    assert out.cloud_gpu_rental is not None
    assert out.hosted_api_token is not None
    assert isinstance(out.cloud_gpu_rental.trust_envelope, TrustEnvelope)
    assert isinstance(out.hosted_api_token.trust_envelope, TrustEnvelope)


def test_compare_deployment_modes_registered_as_mcp_tool(_: Any = None) -> None:
    """Registration smoke test."""
    import asyncio

    from whatcanirun.server import mcp

    tools = asyncio.run(mcp.get_tools())
    assert "compare_deployment_modes" in tools, (
        f"`compare_deployment_modes` tool not registered on `mcp`; "
        f"registered tools: {sorted(tools)}"
    )
