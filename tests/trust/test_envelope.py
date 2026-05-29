"""`TrustEnvelope` + `Source` Pydantic shapes per spec/SHARED.md.

M08 builds partial envelopes for each CostCell. M09 enriches them
when the MCP tool layer wraps a response (workload_assumption
domain populated only when the tool synthesized derived counts).

`confidence` is `min(confidence_breakdown.values())` — weakest
link by design. The trust contract demands that the LLM client
sees the WORST domain, not the average.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from whatcanirun.trust.envelope import (
    ConfidenceDomain,
    Source,
    TrustEnvelope,
)


def _src() -> Source:
    return Source(
        name="computeprices",
        detail="GET /api/v1/gpu-prices, 1h cache",
        last_updated=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
        license_attribution=None,
    )


def _envelope_kwargs() -> dict:
    return dict(
        sources=[_src()],
        confidence_breakdown={
            "pricing": 0.95,
            "fit_check": 0.9,
            "throughput": 0.6,
            "model_architecture": 0.9,
            "gpu_specs": 0.85,
            "freshness": 0.8,
        },
        assumptions={"workload_profile": "chat_assistant"},
        caveats=["AA reports a serving aggregate across providers"],
        freshness={"computeprices": dt.datetime(2026, 5, 28, tzinfo=dt.UTC)},
        verify_links=["https://www.computeprices.com/api/v1/gpu-prices"],
    )


# ------------------------------------------------------------- shape


def test_minimal_envelope_constructs() -> None:
    env = TrustEnvelope(**_envelope_kwargs())
    assert env.confidence_breakdown["throughput"] == 0.6


def test_extra_field_rejected() -> None:
    """`extra="forbid"` — same convention as the other owned
    schemas (FitResult, TpsEstimate, WorkloadProfile, ...).
    A typo in M08's envelope construction must fail loudly,
    not silently drop into an unknown bucket."""
    kwargs = _envelope_kwargs() | {"bogus_field": "oops"}
    with pytest.raises(ValidationError):
        TrustEnvelope(**kwargs)


# ------------------------------------------------------- weakest link


def test_confidence_is_min_of_breakdown_values() -> None:
    """Spec § Rollup semantics: `confidence` is computed as
    `min(confidence_breakdown.values())`. Weakest-link by design.
    The LLM client surfaces this top-level number AND the worst
    domain; never an average or geometric mean (both would
    sandbag a 0.0 'requires_measurement' result into something
    that looks fine)."""
    env = TrustEnvelope(**_envelope_kwargs())
    expected_min = min(env.confidence_breakdown.values())
    assert env.confidence == expected_min  # 0.6 (throughput)


def test_confidence_with_zero_domain_returns_zero() -> None:
    """If any domain is 0.0 (e.g. throughput=requires_measurement),
    the overall confidence is 0.0. This is the load-bearing case —
    a refusal in one domain must NOT be masked by high confidence
    in others."""
    kwargs = _envelope_kwargs()
    kwargs["confidence_breakdown"] = {**kwargs["confidence_breakdown"], "throughput": 0.0}
    env = TrustEnvelope(**kwargs)
    assert env.confidence == 0.0


# ------------------------------------------- workload_assumption domain


def test_workload_assumption_optional_and_distinct_from_other_domains() -> None:
    """spec/SHARED.md: workload_assumption is populated ONLY by
    tools that synthesize derived counts from a workload (e.g.
    budget_to_plan's est_total_prompts). Omit the key entirely
    when no workload was assumed — don't default to a fudge
    value, don't include the key with 0.0.

    For a `CostCell` query (M08), we don't synthesize a count —
    we report per-cell pricing. The envelope omits workload_
    assumption. M09's budget_to_plan adds it when wrapping."""
    kwargs = _envelope_kwargs()
    # No workload_assumption key — represents a query that didn't
    # touch workload-derived math.
    assert "workload_assumption" not in kwargs["confidence_breakdown"]
    env = TrustEnvelope(**kwargs)
    assert "workload_assumption" not in env.confidence_breakdown
    # confidence is min of the 6 domains present, not affected.
    assert env.confidence == 0.6


def test_workload_assumption_present_when_tool_assumes_workload() -> None:
    """When a tool (M09 budget_to_plan) DOES synthesize a count
    from an assumed workload, the domain is present. Value
    convention per spec: ~0.95 when user-elicited, ~0.2 for
    silent defaults."""
    kwargs = _envelope_kwargs()
    kwargs["confidence_breakdown"] = {
        **kwargs["confidence_breakdown"],
        "workload_assumption": 0.95,  # user-elicited
    }
    env = TrustEnvelope(**kwargs)
    assert env.confidence_breakdown["workload_assumption"] == 0.95


# ------------------------------------------------------------- Source


def test_source_with_attribution() -> None:
    """AA-sourced cells require the verbatim attribution string
    (M04). M08 cells derived from AA data carry this on their
    Source entry."""
    from whatcanirun.pricing.artificial_analysis import AA_ATTRIBUTION_STRING

    src = Source(
        name="artificial_analysis",
        detail="median_output_tokens_per_second for gpt-oss-120b",
        last_updated=dt.datetime(2026, 5, 27, tzinfo=dt.UTC),
        license_attribution=AA_ATTRIBUTION_STRING,
    )
    assert src.license_attribution is not None
    assert "Artificial Analysis" in src.license_attribution


def test_source_name_enum_closed() -> None:
    """`Source.name` is a closed Literal — typos like
    `'compute_prices'` (underscore vs no underscore) must fail
    validation."""
    with pytest.raises(ValidationError):
        Source(
            name="compute_prices",  # type: ignore[arg-type]
            detail="x",
            last_updated=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
            license_attribution=None,
        )


# ------------------------------------------------------ ConfidenceDomain


def test_confidence_domain_keys_match_spec() -> None:
    """The exact 7 domain names from spec/SHARED.md must be in
    the type. If the spec ever adds another, this test goes red
    so we update both spec and type together."""
    # Inspect the Literal's __args__
    domains = set(ConfidenceDomain.__args__)  # type: ignore[attr-defined]
    expected = {
        "pricing",
        "fit_check",
        "throughput",
        "model_architecture",
        "gpu_specs",
        "workload_assumption",
        "freshness",
    }
    assert domains == expected
