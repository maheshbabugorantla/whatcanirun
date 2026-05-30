"""M09 Slice C: `fit_check` MCP tool — TDD.

`fit_check` wraps M06's `compute_fit` and constructs the trust
envelope for the three domains the response depends on:

- `fit_check`         — accuracy of the VRAM-fit math itself
- `model_architecture`— freshness/quality of the HF config.json
                        that supplied `n_layers`, `n_kv_heads`,
                        `head_dim`, `total_params_b`
- `gpu_specs`         — freshness/quality of the GPU VRAM number

`freshness` is the weakest-link rollup across HF + CP timestamps.

Per spec/M09 § Public surface §4 and spec/SHARED.md § "When relaying
tool output to the user" rule 3, the response must propagate
`sufficiency_caveat` from the underlying `FitResult` even when
`fits=True` — the trust envelope's `caveats` list and the
`FitResult.sufficiency_caveat` field BOTH carry it so a client
can't accidentally surface the fits-bool without the disclaimer.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.mcp_tools.fit_check import (
    FitCheckToolResponse,
    build_fit_check_response,
)
from whatcanirun.pricing.projections import GpuCatalogRow
from whatcanirun.trust.envelope import TrustEnvelope

# ---------------------------------------------------------------- factories


def _model(
    *,
    slug: str = "test-model",
    total_params_b: float = 70.6,
    last_synced_at: dt.datetime | None = None,
) -> Model:
    return Model(
        slug=slug,
        hf_repo_id=f"vendor/{slug}",
        display_name=slug,
        total_params_b=total_params_b,
        active_params_b=None,
        n_layers=80,
        n_attention_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        max_position_embeddings=131072,
        native_dtype="bfloat16",
        architecture_family="llama",
        kv_cache_strategy="standard_gqa",
        raw_config={},
        raw_safetensors_meta={},
        hf_revision_sha="abcd1234",
        last_synced_at=last_synced_at or dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


def _gpu(slug: str = "h100sxm", vram_gb: int = 80) -> GpuCatalogRow:
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="Hopper",
        vram_gb=vram_gb,
        release_date=None,
        specs={},
        raw={},
    )


def _quant(slug: str = "fp16", bits_per_weight: int = 16) -> Quantization:
    return Quantization(
        slug=slug,
        bits_per_weight=bits_per_weight,
        kv_cache_bits_default=16,
        introduced_architecture="Ampere",
        notes="",
        experimental=False,
    )


def _now() -> dt.datetime:
    """Anchor for freshness calculations — a fixed timestamp 2 hours
    after the synthetic model's `last_synced_at`. The HF freshness
    breakpoint per spec/SHARED.md is 30 days, so a 2-hour offset
    keeps the model_architecture confidence in the 0.95 band."""
    return dt.datetime(2026, 5, 28, 2, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------- tests


def test_build_fit_check_response_returns_typed_wrapper() -> None:
    """The pure builder returns a `FitCheckToolResponse` wrapping
    both the M06 `FitResult` and the M09 `TrustEnvelope`. Tools
    that wrap pure-math results need a stable Pydantic shape so
    the FastMCP serializer produces a predictable wire schema."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu("h100sxm", vram_gb=80),
        quant=_quant("fp8", bits_per_weight=8),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    assert isinstance(response, FitCheckToolResponse)
    assert isinstance(response.trust_envelope, TrustEnvelope)


def test_envelope_breakdown_covers_fit_check_arch_gpu_freshness_domains() -> None:
    """Per spec/SHARED.md § Calibration the fit_check response must
    populate every domain whose data went into the answer. For
    fit_check that's: `fit_check`, `model_architecture`,
    `gpu_specs`, `freshness`. `workload_assumption` is OMITTED —
    no derived prompt count, so the key must not appear (per
    spec/SHARED.md § ConfidenceDomain semantics)."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    breakdown = response.trust_envelope.confidence_breakdown
    assert set(breakdown.keys()) == {
        "fit_check",
        "model_architecture",
        "gpu_specs",
        "freshness",
    }


def test_envelope_confidence_is_weakest_link() -> None:
    """`TrustEnvelope.confidence` is computed as the min over
    `confidence_breakdown.values()`. A direct property test guards
    against a future refactor that hand-stores `confidence` and
    drifts from the breakdown."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    breakdown = response.trust_envelope.confidence_breakdown
    assert response.trust_envelope.confidence == min(breakdown.values())


def test_envelope_sources_attribute_hf_and_cp() -> None:
    """The two upstreams that contributed data are HF (config.json)
    and ComputePrices (gpu catalog). Both must appear in `sources`
    so the LLM client can relay the per-source freshness map."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    source_names = {s.name for s in response.trust_envelope.sources}
    assert "huggingface" in source_names
    assert "computeprices" in source_names


def test_envelope_assumptions_echo_op_point_and_tp_size() -> None:
    """Spec/M06 acceptance criterion 3 + spec/M09 trust contract:
    the envelope's `assumptions` must echo the op-point parameters
    so a client surfacing the response can show the user 'this
    answer assumes tp_size=2, batch=8, ctx=8192' — relevant if
    they want to ask the same question with different inputs."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=2,
        batch_size=8,
        context_length=8192,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    assumptions = response.trust_envelope.assumptions
    assert assumptions.get("tp_size") == 2
    assert assumptions.get("batch_size") == 8
    assert assumptions.get("context_length") == 8192


def test_envelope_freshness_maps_source_to_timestamp() -> None:
    """Per spec/SHARED.md the `freshness` map is keyed by source
    NAME (not domain) and carries the last_updated timestamp of
    each contributing upstream. The fit_check tool must expose
    both `huggingface` (model.last_synced_at) and `computeprices`
    (gpu_specs_last_updated)."""
    synced = dt.datetime(2026, 5, 28, tzinfo=dt.UTC)
    gpu_updated = dt.datetime(2026, 5, 28, 1, 0, tzinfo=dt.UTC)
    response = build_fit_check_response(
        model=_model(last_synced_at=synced),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=gpu_updated,
    )
    freshness = response.trust_envelope.freshness
    assert freshness["huggingface"] == synced
    assert freshness["computeprices"] == gpu_updated


def test_envelope_carries_sufficiency_caveat_verbatim() -> None:
    """Spec/M09 relay rule 3: `fits=True` is necessary but not
    sufficient. The envelope's `caveats` list must include the
    `FitResult.sufficiency_caveat` verbatim so a client that
    relays caveats without dereferencing the embedded FitResult
    still surfaces the disclaimer."""
    response = build_fit_check_response(
        model=_model(),
        gpu=_gpu("h100sxm", vram_gb=80),
        quant=_quant("fp8", bits_per_weight=8),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    sufficiency = response.fit_result.sufficiency_caveat
    assert any(sufficiency in c for c in response.trust_envelope.caveats), (
        f"sufficiency caveat not echoed in envelope caveats; got: {response.trust_envelope.caveats}"
    )


def test_doesnt_fit_envelope_still_well_formed() -> None:
    """An undersized GPU produces `fits=False` with `blocking_reasons`
    populated. The trust envelope must still be present and
    well-formed; the trust contract is independent of the verdict.
    A future bug that skips envelope construction on the doesn't-fit
    branch would surface here."""
    # 405B at fp16 on a single 24GB GPU — definitely doesn't fit.
    response = build_fit_check_response(
        model=_model(slug="llama-3-1-405b", total_params_b=405.0),
        gpu=_gpu("rtx4090", vram_gb=24),
        quant=_quant("fp16", bits_per_weight=16),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    assert response.fit_result.fits is False
    assert response.fit_result.blocking_reasons
    assert response.trust_envelope.confidence > 0  # envelope still meaningful


def test_stale_hf_data_lowers_model_architecture_confidence() -> None:
    """Per spec/SHARED.md § Staleness policy: HF config.json older
    than 30 days drops model_architecture confidence from 0.95 to
    0.80. The weakest-link rollup means `confidence` should reflect
    that drop, not the optimistic 0.95."""
    very_old = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)  # >30 days before _now()
    response = build_fit_check_response(
        model=_model(last_synced_at=very_old),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    breakdown = response.trust_envelope.confidence_breakdown
    # The freshness rollup pulls the breakdown's freshness key down
    # to the HF age band (<= 0.80 per the spec function).
    assert breakdown["freshness"] <= 0.80


def test_degenerate_input_raises_validation_error() -> None:
    """`compute_fit` raises ValueError on non-positive batch_size,
    context_length, or tp_size. The tool wrapper propagates the
    error rather than constructing a hollow envelope around a
    degenerate result."""
    with pytest.raises(ValueError, match="batch_size"):
        build_fit_check_response(
            model=_model(),
            gpu=_gpu(),
            quant=_quant(),
            tp_size=1,
            batch_size=0,  # degenerate
            context_length=4096,
            now=_now(),
            gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
        )


def test_fit_check_registered_as_mcp_tool() -> None:
    """The tool must be wired into the FastMCP instance as a tool
    named `fit_check`. A registration regression fails here, not
    at MCP-client connection time."""
    import asyncio

    from whatcanirun.server import mcp

    tools = asyncio.run(mcp.get_tools())
    assert "fit_check" in tools, (
        f"`fit_check` tool not registered on `mcp`; registered tools: {sorted(tools)}"
    )


# ---------------------------------------------------------------- params-spec
# Parametrize one tiny exhaustive coverage of the model_architecture
# confidence band so future drift in the spec's age->confidence
# breakpoints is caught explicitly.


@pytest.mark.parametrize(
    ("hf_age_days", "expected_min_freshness"),
    [
        (1, 0.95),  # well within the 30-day fresh window
        (45, 0.80),  # past the 30-day fresh window, in the 0.80 floor
    ],
)
def test_hf_freshness_breakpoints_match_spec(
    hf_age_days: int,
    expected_min_freshness: float,
    _factory: Any = None,  # placeholder so parametrize ids look clean
) -> None:
    """spec/SHARED.md § Staleness policy: HF<=30d → 0.95, else → 0.80.
    A future change to that function in `trust/calibration.py`
    that drifts the breakpoint fails here so the divergence is
    visible at the trust-envelope edge, not buried in a refactor."""
    synced = _now() - dt.timedelta(days=hf_age_days)
    response = build_fit_check_response(
        model=_model(last_synced_at=synced),
        gpu=_gpu(),
        quant=_quant(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
        now=_now(),
        gpu_specs_last_updated=_now() - dt.timedelta(hours=1),
    )
    assert response.trust_envelope.confidence_breakdown["freshness"] == pytest.approx(
        min(expected_min_freshness, 0.95)  # cp side is fresh in both rows
    )
