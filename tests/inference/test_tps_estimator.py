"""M07 `estimate_tps` — 4-tier throughput provenance.

Pure function with explicit confidence values per tier:
  Tier 1a own_measured            confidence=0.95   (v2 only — bench_cells defaults to [])
  Tier 2  provider_anchor (AA)    confidence=0.7
  Tier 3  bandwidth_heuristic     confidence=0.6    (single-stream)
  Tier 4  requires_measurement    confidence=0.0    (refusal)

Tier 1b (public_benchmark_anchor at confidence=0.80) was removed
with the M10 deferral (2026-05-31). The bench_cells parameter
still exists at the estimator boundary so v2's M17
GuideLLM-measured cells can revive Tier 1a; v1 callers omit it
and the loop is a no-op. Reasoning models require AA-row effort
match.

Test fixtures use synthetic Model / GpuCatalogRow / Quantization
instances per the M06 pattern — `estimate_tps` is pure so it
makes no sense to gate tests on the cache being populated.
"""

from __future__ import annotations

import datetime as dt
from datetime import date
from typing import Any

import pytest

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.inference.tps_estimator import (
    KERNEL_EFFICIENCY_SINGLE_STREAM,
    TpsEstimate,
    estimate_tps,
)
from whatcanirun.pricing.artificial_analysis import AaModelRow
from whatcanirun.pricing.projections import GpuCatalogRow

# ---------------------------------------------------------------- factories


def _model(
    *,
    slug: str = "llama-3-3-70b",
    total_params_b: float = 70.6,
    active_params_b: float | None = None,
    n_layers: int = 80,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    kv_cache_strategy: str = "standard_gqa",
) -> Model:
    return Model(
        slug=slug,
        hf_repo_id=f"vendor/{slug}",
        display_name=slug,
        total_params_b=total_params_b,
        active_params_b=active_params_b,
        n_layers=n_layers,
        n_attention_heads=64,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=8192,
        max_position_embeddings=131072,
        native_dtype="bfloat16",
        architecture_family="llama",
        kv_cache_strategy=kv_cache_strategy,  # type: ignore[arg-type]
        raw_config={},
        raw_safetensors_meta={},
        hf_revision_sha="x",
        last_synced_at=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


def _gpu(slug: str, vram_gb: int, memory_bandwidth_gbps: int) -> GpuCatalogRow:
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="Hopper",
        vram_gb=vram_gb,
        release_date=None,
        specs={"memory_bandwidth_gbps": memory_bandwidth_gbps},
        raw={},
    )


def _quant(slug: str, bits_per_weight: int, kv_cache_bits_default: int) -> Quantization:
    return Quantization(
        slug=slug,
        bits_per_weight=bits_per_weight,
        kv_cache_bits_default=kv_cache_bits_default,
        introduced_architecture="Hopper",
        notes="",
        experimental=False,
    )


def _aa_row(slug: str, tps: float) -> AaModelRow:
    return AaModelRow.project(
        {
            "id": f"uuid-{slug}",
            "slug": slug,
            "name": slug,
            "model_creator": {"id": "uuid-v", "name": "Vendor", "slug": "vendor"},
            "release_date": "2026-01-01",
            "median_output_tokens_per_second": tps,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        }
    )


def _anchor(
    *,
    gpu: str = "h100",
    model: str = "llama-3-3-70b",
    quant: str = "fp8",
    tp: int = 1,
    batch: int = 1,
    ctx: int = 4096,
    tps: float = 35.2,
    source: str = "public_benchmark_anchor",
) -> BenchmarkCell:
    return BenchmarkCell(
        gpu_slug=gpu,
        model_slug=model,
        quant_slug=quant,
        tp_size=tp,
        batch_size=batch,
        context_length=ctx,
        decode_tps=tps,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source=source,  # type: ignore[arg-type]
        source_url="https://www.spheron.network/blog/llama-3-3-70b-fp8",
        notes="Single H100 SXM, FP8 quantization, batch=1, ctx=4096.",
    )


def _h100() -> GpuCatalogRow:
    return _gpu("h100", 80, 3350)


def _fp8() -> Quantization:
    return _quant("fp8", 8, 8)


def _bf16() -> Quantization:
    return _quant("bf16", 16, 16)


# ============================================================ Slice 1
# Tier 1a own_measured — v2 only, currently dead in v1 (BenchmarkCell
# validator rejects own_measured at row-construction time). We test
# the tier's lookup logic by bypassing the validator with model_construct,
# so a future v2 unlock can flip the validator off and the existing
# logic still works.


def test_tier1a_own_measured_wins_at_confidence_095() -> None:
    """Spec slice 1: Tier 1a — `own_measured` exact-match row at
    confidence=0.95. v1 production never reaches this branch
    because BenchmarkCell rejects own_measured at construction;
    we use `model_construct` here to simulate a v2 row + assert
    the lookup logic + confidence value are correct so the v2
    unlock is a single validator flip, not a re-test of the tier
    machinery."""
    measured = BenchmarkCell.model_construct(
        gpu_slug="h100",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=40.5,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source="own_measured",
        source_url="https://internal/guidellm-run-42",
        notes="(v2 simulated)",
    )
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[measured],
        aa_observations=None,
    )
    assert result.value == 40.5
    assert result.source == "own_measured"
    assert result.confidence == 0.95


# ============================================================ Slice 2
# Tier 1b public_benchmark_anchor — REMOVED with M10 deferral
# (2026-05-31). Tests for the public-anchor tier deleted; the
# v2-ready Tier 1a code path is exercised by the tier-ordering
# tests in Slice 6 + the parametrize in Slice 8.


# ============================================================ Slice 3
# Tier 2 provider_anchor (AA fallback at batch=1).


def test_tier2_aa_provider_anchor_at_batch_1_confidence_07() -> None:
    """Spec slice 3: no bench_cells row, but AA has the model —
    Tier 2 returns AA's median TPS at confidence=0.7. Caveat
    mentioned per spec (anchor_detail names AA + the aggregate
    nature)."""
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=[_aa_row("llama-3-3-instruct-70b", tps=89.6)],
        # Slug mapping resolution is delegated to the caller via
        # an aa_slug pre-resolved kwarg; M07's tps_estimator takes
        # the already-filtered AA row(s) for the current model.
    )
    assert result.source == "provider_anchor"
    assert result.value == 89.6
    assert result.confidence == 0.7
    assert result.anchor_detail is not None
    assert (
        "aa" in result.anchor_detail.lower()
        or "artificial analysis" in result.anchor_detail.lower()
    )


def test_tier2_skipped_when_batch_gt_1() -> None:
    """AA's median is a single-stream aggregate. For batch>1 it's
    meaningless, so Tier 2 doesn't fire — falls through to Tier 4
    (refusal). Pinned to the spec's exact criterion: 'AA has
    median_output_tokens_per_second for this model AND batch_size==1'."""
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=8,
        context_length=4096,
        bench_cells=[],
        aa_observations=[_aa_row("llama-3-3-instruct-70b", tps=89.6)],
    )
    assert result.source == "requires_measurement"
    assert result.value is None


# ============================================================ Slice 4
# Tier 3 bandwidth heuristic.


def test_tier3_bandwidth_heuristic_llama70b_fp8_h100_matches_spec_anchor() -> None:
    """Spec slice 4: anchor verification.
      weights_bytes_per_token = 70.6 * 1e9 * 8 / 8 = 70.6e9
      peak_tps = 3350 * 1e9 / 70.6e9 = 47.45
      value = 47.45 * 0.75 = 35.59 tok/s
    Spec says ~35.9 expected; both ~35-36 range. Pin to ±1 tok/s."""
    result = estimate_tps(
        model=_model(total_params_b=70.6),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=None,
    )
    assert result.source == "bandwidth_heuristic_single_stream"
    assert result.confidence == 0.6
    assert result.value is not None
    assert 35.0 <= result.value <= 37.0, f"expected ~35.5 tok/s, got {result.value}"


def test_tier3_uses_named_kernel_efficiency_constant() -> None:
    """Spec acceptance criterion: `KERNEL_EFFICIENCY_SINGLE_STREAM
    = 0.75` is a named module constant with citation comment.
    Pin its value here so a drive-by edit can't silently change
    the heuristic."""
    assert KERNEL_EFFICIENCY_SINGLE_STREAM == 0.75


def test_tier3_skipped_when_batch_gt_1() -> None:
    """Spec common pitfall: 'Don't scale heuristic with batch.
    Verified ~6x wrong at batch=128. batch>1 falls through to
    Tier 4, period.' Test pins exactly that."""
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=4,
        context_length=4096,
        bench_cells=[],
        aa_observations=None,
    )
    assert result.source == "requires_measurement"


def test_tier3_missing_bandwidth_falls_through() -> None:
    """If CP's specs dict doesn't carry `memory_bandwidth_gbps`,
    the heuristic can't fire — fall through to Tier 4 with a
    clear refusal_reason, not a divide-by-zero or KeyError."""
    no_bw_gpu = GpuCatalogRow(
        slug="bandwidth-missing",
        name="X",
        manufacturer="NVIDIA",
        architecture="X",
        vram_gb=80,
        release_date=None,
        specs={},
        raw={},
    )
    result = estimate_tps(
        model=_model(),
        gpu=no_bw_gpu,
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=None,
    )
    assert result.source == "requires_measurement"


# ============================================================ Slice 5
# Tier 4 refusal.


def test_tier4_batch_gt_1_no_anchor_refuses_honestly() -> None:
    """Spec slice 5: batch>1 + no measured/anchor → refusal.
    value=None, confidence=0.0, refusal_reason verbatim per spec."""
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=32,
        context_length=4096,
        bench_cells=[],
        aa_observations=None,
    )
    assert result.value is None
    assert result.source == "requires_measurement"
    assert result.confidence == 0.0
    assert result.refusal_reason is not None
    # Spec's exact phrasing for the refusal reason — pin so an
    # edit that softens it (e.g. removes "honestly") goes red.
    assert "batched throughput not modeled by heuristic" in result.refusal_reason


# ============================================================ Slice 6
# Tier ordering.


def test_tier_ordering_1a_beats_2() -> None:
    """When both Tier 1a (own_measured, v2-only) and Tier 2 (AA)
    match, 1a wins. Pins the v2-ready dead-code path: if a future
    caller passes own_measured bench_cells alongside AA rows, the
    own_measured anchor takes precedence over the provider
    aggregate.

    (Tier 1b public_benchmark_anchor was removed with the M10
    deferral, so the original 1a-beats-1b test is gone too.)"""
    measured = BenchmarkCell.model_construct(
        gpu_slug="h100",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=40.5,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source="own_measured",
        source_url="https://internal/run-42",
        notes="(v2 simulated)",
    )
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[measured],
        aa_observations=[_aa_row("llama-3-3-instruct-70b", tps=89.6)],
    )
    assert result.value == 40.5  # own_measured wins
    assert result.source == "own_measured"


# ============================================================ Slice 7
# Reasoning effort dimension.


def test_tier2_aa_matches_requested_reasoning_effort() -> None:
    """Spec slice 7: reasoning model query with multiple AA rows
    (one per effort variant) — Tier 2 returns the row matching
    the requested effort. Wrong-effort rows DON'T match."""
    rows_low = AaModelRow.project(
        {
            "id": "u-low",
            "slug": "gpt-oss-120b-low",
            "name": "gpt-oss-120b (low)",
            "model_creator": {"id": "u", "name": "OpenAI", "slug": "openai"},
            "release_date": "2025-08-05",
            "median_output_tokens_per_second": 200.0,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        },
        reasoning_effort="low",
    )
    rows_high = AaModelRow.project(
        {
            "id": "u-high",
            "slug": "gpt-oss-120b-high",
            "name": "gpt-oss-120b (high)",
            "model_creator": {"id": "u", "name": "OpenAI", "slug": "openai"},
            "release_date": "2025-08-05",
            "median_output_tokens_per_second": 100.0,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        },
        reasoning_effort="high",
    )
    result = estimate_tps(
        model=_model(slug="gpt-oss-120b"),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=[rows_low, rows_high],
        reasoning_effort="high",
    )
    assert result.value == 100.0  # high variant
    assert result.source == "provider_anchor"


def test_tier2_falls_through_when_no_aa_row_matches_requested_effort() -> None:
    """If we request -high but AA only has -low, no match — falls
    through to Tier 3 (bandwidth heuristic if batch=1) or Tier 4."""
    only_low = AaModelRow.project(
        {
            "id": "u-low",
            "slug": "gpt-oss-120b-low",
            "name": "gpt-oss-120b (low)",
            "model_creator": {"id": "u", "name": "OpenAI", "slug": "openai"},
            "release_date": "2025-08-05",
            "median_output_tokens_per_second": 200.0,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        },
        reasoning_effort="low",
    )
    result = estimate_tps(
        model=_model(slug="gpt-oss-120b"),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=[only_low],
        reasoning_effort="high",  # AA only has low
    )
    # No AA match → falls through. At batch=1 Tier 3 fires.
    assert result.source == "bandwidth_heuristic_single_stream"


def test_tier2_matches_when_no_reasoning_effort_requested_and_aa_row_has_none() -> None:
    """Non-reasoning case: caller doesn't pass reasoning_effort,
    AA row has reasoning_effort=None — they match."""
    plain = _aa_row("llama-3-3-instruct-70b", tps=89.6)
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[],
        aa_observations=[plain],
        reasoning_effort=None,
    )
    assert result.source == "provider_anchor"


# ============================================================ Slice 8
# Exact confidence values + properties + purity.


@pytest.mark.parametrize(
    ("source", "expected_confidence"),
    [
        ("own_measured", 0.95),
        # public_benchmark_anchor removed with M10 deferral.
        ("provider_anchor", 0.7),
        ("bandwidth_heuristic_single_stream", 0.6),
        ("requires_measurement", 0.0),
    ],
)
def test_confidence_values_are_exact_no_fudge(source: str, expected_confidence: float) -> None:
    """Spec acceptance criterion: confidence values 0.95 / 0.7 /
    0.6 / 0.0 — exact, no fudge factors. Pin via a parametrize so
    a drive-by edit can't quietly raise tier-2 to 0.75 (or whatever).
    Public-anchor 0.80 was removed with the M10 deferral."""
    # Build the minimum inputs to exercise each tier.
    if source == "own_measured":
        cells = [
            BenchmarkCell.model_construct(
                gpu_slug="h100",
                model_slug="llama-3-3-70b",
                quant_slug="fp8",
                tp_size=1,
                batch_size=1,
                context_length=4096,
                decode_tps=42.0,
                prefill_tps=None,
                ttft_ms=None,
                engine="vllm",
                engine_version="0.6.x",
                measured_at=date(2026, 3, 15),
                source="own_measured",
                source_url="https://internal",
                notes="",
            )
        ]
        aa = None
    elif source == "provider_anchor":
        cells = []
        aa = [_aa_row("llama-3-3-instruct-70b", tps=89.6)]
    elif source == "bandwidth_heuristic_single_stream":
        cells = []
        aa = None
    else:  # requires_measurement
        cells = []
        aa = None

    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=32 if source == "requires_measurement" else 1,
        context_length=4096,
        bench_cells=cells,
        aa_observations=aa,
    )
    assert result.confidence == expected_confidence, (
        f"source={source}: expected {expected_confidence}, got {result.confidence}"
    )


def test_property_no_value_without_source() -> None:
    """Spec acceptance criterion: no `TpsEstimate.value` non-None
    without a populated `source`. Sweep a few representative
    configurations."""
    configs: list[dict[str, Any]] = [
        dict(bench_cells=[_anchor()], aa_observations=None),
        dict(bench_cells=[], aa_observations=[_aa_row("llama-3-3-instruct-70b", 89.6)]),
        dict(bench_cells=[], aa_observations=None),  # Tier 3
    ]
    for cfg in configs:
        result = estimate_tps(
            model=_model(),
            gpu=_h100(),
            quant=_fp8(),
            batch_size=1,
            context_length=4096,
            **cfg,
        )
        if result.value is not None:
            assert result.source != "requires_measurement"


def test_purity_estimate_tps_does_not_touch_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec acceptance criterion: estimate_tps is pure (no I/O).
    Same booby-trap pattern as M06's fit_check purity test —
    install/restore `open` and `__import__` manually so pytest's
    teardown doesn't trip the trap."""
    import builtins

    args = dict(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[_anchor()],
        aa_observations=None,
    )
    expected = estimate_tps(**args)

    real_open = builtins.open
    real_import = builtins.__import__

    def _no_open(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("estimate_tps must not open files")

    import_attempts: list[str] = []

    def _trap_import(name: str, *a: Any, **k: Any) -> Any:
        import_attempts.append(name)
        return real_import(name, *a, **k)

    builtins.open = _no_open  # type: ignore[assignment]
    builtins.__import__ = _trap_import  # type: ignore[assignment]
    try:
        actual = estimate_tps(**args)
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        builtins.__import__ = real_import  # type: ignore[assignment]

    assert actual.model_dump() == expected.model_dump()
    assert import_attempts == [], f"estimate_tps performed runtime imports: {import_attempts!r}"


def test_result_is_pydantic_with_all_required_fields() -> None:
    """TpsEstimate has the documented fields per spec § Public
    surface. Pin via a smoke construction so a refactor that
    drops a field (e.g. anchor_detail) goes red."""
    result = estimate_tps(
        model=_model(),
        gpu=_h100(),
        quant=_fp8(),
        batch_size=1,
        context_length=4096,
        bench_cells=[_anchor()],
        aa_observations=None,
    )
    assert isinstance(result, TpsEstimate)
    fields = result.model_dump()
    for required in (
        "value",
        "source",
        "confidence",
        "anchor_detail",
        "source_url",
        "refusal_reason",
    ):
        assert required in fields
