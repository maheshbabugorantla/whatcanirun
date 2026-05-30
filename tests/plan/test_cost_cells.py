"""M08 cost_cells join layer — `CostCell`, `CostCellFilters`,
`query_cost_cells`.

`query_cost_cells` is the tool-call hot path. Pure Python list
comprehensions over in-memory caches. ADR-014: NO SQL in this
path; DuckDB belongs only in `render_cost_cells_resource()`. The
grep test in `test_no_sql_in_business_logic.py` enforces.

Synthetic fixtures throughout — same M06/M07 pattern. The
function is pure; tests don't need a real CP cache populated.
"""

from __future__ import annotations

import datetime as dt
from datetime import date, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.inference.fit_check import FitResult
from whatcanirun.inference.tps_estimator import TpsEstimate
from whatcanirun.plan.cost_cells import (
    CostCell,
    CostCellFilters,
    query_cost_cells,
)
from whatcanirun.pricing.artificial_analysis import AaModelRow
from whatcanirun.pricing.projections import GpuCatalogRow, GpuPriceRow, LlmPriceRow
from whatcanirun.trust.envelope import Source, TrustEnvelope

# ---------------------------------------------------------------- factories


def _model(slug: str = "llama-3-3-70b", total_params_b: float = 70.6) -> Model:
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
        hf_revision_sha="x",
        last_synced_at=datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


def _gpu(slug: str = "h100", vram_gb: int = 80, bandwidth: int = 3350) -> GpuCatalogRow:
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="Hopper",
        vram_gb=vram_gb,
        release_date=None,
        specs={"memory_bandwidth_gbps": bandwidth},
        raw={},
    )


def _quant(slug: str = "fp8", bpw: int = 8, kvb: int = 8) -> Quantization:
    return Quantization(
        slug=slug,
        bits_per_weight=bpw,
        kv_cache_bits_default=kvb,
        introduced_architecture="Hopper",
        notes="",
        experimental=False,
    )


def _gpu_price(
    *,
    gpu_slug: str = "h100",
    provider_slug: str = "lambda",
    hourly: float = 2.49,
    pricing_type: str = "on_demand",
) -> GpuPriceRow:
    return GpuPriceRow(
        provider="Lambda Labs",
        provider_slug=provider_slug,
        gpu="H100 SXM",
        gpu_slug=gpu_slug,
        vram_gb=80,
        gpu_count=1,
        price_per_hour_usd=hourly,
        pricing_type=pricing_type,  # type: ignore[arg-type]
        commitment_months=None,
        currency="USD",
        source_url="https://lambdalabs.com/pricing",
        last_updated=datetime(2026, 5, 28, tzinfo=dt.UTC),
        raw={},
    )


def _llm_price(
    *,
    provider_slug: str = "together",
    model_slug: str = "llama-3-3-70b",
    input_per_m: float | None = 0.88,
    output_per_m: float | None = 0.88,
) -> LlmPriceRow:
    return LlmPriceRow(
        provider="Together AI",
        provider_slug=provider_slug,
        model_slug=model_slug,
        price_per_1m_input_usd=input_per_m,
        price_per_1m_output_usd=output_per_m,
        price_per_1m_cached_input_usd=None,
        pricing_type="standard",
        last_updated=datetime(2026, 5, 28, tzinfo=dt.UTC),
        raw={},
    )


def _filters(**overrides: Any) -> CostCellFilters:
    """Helper — most tests want a baseline filters object."""
    return CostCellFilters(**overrides)


# ============================================================ Slice A
# CostCell schema validation.


def _minimal_envelope() -> TrustEnvelope:
    """A bare envelope with one source — enough to satisfy the
    CostCell required field. Tests don't need real M09 enrichment;
    they just need the field populated."""
    return TrustEnvelope(
        sources=[
            Source(
                name="computeprices",
                detail="GET /api/v1/gpu-prices",
                last_updated=datetime(2026, 5, 28, tzinfo=dt.UTC),
            )
        ],
        confidence_breakdown={"pricing": 0.95},
    )


def _minimal_tps() -> TpsEstimate:
    return TpsEstimate(
        value=35.0,
        source="public_benchmark_anchor",
        confidence=0.80,
        source_url="https://example.com",
    )


def _minimal_fit() -> FitResult:
    """Self-consistent fixture: a 7B model on H100 80GB.
    weight=7, overhead=2.1, kv=0.67, total=9.77, available=80,
    headroom=70.23, fits=True. Internally consistent so it can be
    reused in behavioral tests without surprising readers."""
    return FitResult(
        fits=True,
        weight_gb=7.0,
        kv_cache_gb=0.67,
        framework_overhead_gb=2.1,
        total_required_gb=9.77,
        available_gb=80.0,
        headroom_gb=70.23,
        blocking_reasons=[],
        assumptions={
            "kv_bytes": 1.0,
            "overhead_pct": 0.15,
            "overhead_floor_gb": 2.0,
            "tp_size": 1,
            "kv_cache_strategy": "standard_gqa",
        },
    )


def test_full_cost_cell_constructs() -> None:
    """Spec § CostCell schema: all documented fields populated.
    extra="forbid" — unknown fields fail validation."""
    cell = CostCell(
        gpu_slug="h100",
        provider_slug="lambda",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        deployment_mode="cloud_gpu_rental",
        hourly_usd=2.49,
        pricing_type="on_demand",
        price_per_m_input_usd=None,
        price_per_m_output_usd=None,
        decode_tps=35.0,
        tps_estimate=_minimal_tps(),
        fit_result=_minimal_fit(),
        cost_per_m_output_usd_self_hosted=19.76,
        trust_envelope=_minimal_envelope(),
    )
    assert cell.availability_modeled is False
    assert "Price source does not guarantee" in cell.availability_caveat


def test_cost_cell_rejects_unknown_field() -> None:
    """extra=forbid catches typos in M08's construction code."""
    with pytest.raises(ValidationError):
        CostCell(
            gpu_slug="h100",
            provider_slug="lambda",
            model_slug="x",
            batch_size=1,
            context_length=4096,
            deployment_mode="cloud_gpu_rental",
            tps_estimate=_minimal_tps(),
            trust_envelope=_minimal_envelope(),
            mystery_field="oops",  # type: ignore[call-arg]
        )


def test_hosted_api_token_cell_omits_gpu_fields() -> None:
    """spec § Out of scope + § CostCell schema:
    hosted_api_token rows have null gpu_slug/quant_slug/tp_size/
    hourly_usd/pricing_type/fit_result. price_per_m_* populated."""
    cell = CostCell(
        gpu_slug=None,
        provider_slug="together",
        model_slug="llama-3-3-70b",
        quant_slug=None,
        tp_size=None,
        batch_size=1,
        context_length=4096,
        deployment_mode="hosted_api_token",
        hourly_usd=None,
        pricing_type=None,
        price_per_m_input_usd=0.88,
        price_per_m_output_usd=0.88,
        decode_tps=None,
        tps_estimate=TpsEstimate(
            value=None,
            source="requires_measurement",
            confidence=0.0,
            refusal_reason="hosted_api_token throughput is provider-internal",
        ),
        fit_result=None,
        cost_per_m_output_usd_self_hosted=None,
        trust_envelope=_minimal_envelope(),
    )
    assert cell.fit_result is None
    assert cell.hourly_usd is None


# ============================================================ Slice B
# query_cost_cells with a single filter.


def test_query_returns_cells_for_matching_gpu_only() -> None:
    """Spec slice B: filter by gpu_slug → only matching rows.
    Mixed CP prices for h100 + l40s; query filters to h100."""
    cells = query_cost_cells(
        gpu_prices=[
            _gpu_price(gpu_slug="h100", provider_slug="lambda"),
            _gpu_price(gpu_slug="l40s", provider_slug="runpod", hourly=0.79),
        ],
        llm_prices=[],
        gpu_catalog=[_gpu("h100"), _gpu("l40s", vram_gb=48, bandwidth=864)],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(gpu_slug="h100", batch_size=1, context_length=4096),
    )
    assert len(cells) >= 1
    assert all(c.gpu_slug == "h100" for c in cells)


def test_query_with_no_filters_returns_full_join() -> None:
    """No filter set — every (model, quant, price) combo emerges,
    one cell per."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    assert len(cells) >= 1


def _anchor(tps: float = 35.0) -> BenchmarkCell:
    return BenchmarkCell(
        gpu_slug="h100",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=tps,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source="public_benchmark_anchor",
        source_url="https://example.com/anchor",
        notes="Single H100 SXM, FP8, batch=1, ctx=4096.",
    )


# ============================================================ Slice C
# only_fits filter.


def test_only_fits_excludes_non_fitting_combinations() -> None:
    """Spec slice C: with fits=True and fits=False combinations,
    only_fits=True returns ONLY the fitting one.

    Setup: 7B FP8 model.
      H100 80GB:  7 GB weight + 2.1 overhead + 0.67 KV = 9.77 → fits
      L40S 48GB:  same → fits
      Tiny 4GB GPU: same → doesn't fit (9.77 > 4)
    """
    small_model = _model(slug="llama-3-1-8b", total_params_b=7.0)
    cells = query_cost_cells(
        gpu_prices=[
            _gpu_price(gpu_slug="h100"),
            _gpu_price(gpu_slug="tiny", provider_slug="runpod", hourly=0.10),
        ],
        llm_prices=[],
        gpu_catalog=[
            _gpu("h100"),
            _gpu("tiny", vram_gb=4, bandwidth=200),
        ],
        model_catalog=[small_model],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=None,
        filters=_filters(only_fits=True, batch_size=1, context_length=4096),
    )
    # Every returned cell must have fits=True
    assert all(c.fit_result is None or c.fit_result.fits for c in cells)
    # The tiny-VRAM GPU's combination didn't fit and is excluded.
    assert not any(c.gpu_slug == "tiny" for c in cells)
    # H100 fits and survives.
    assert any(c.gpu_slug == "h100" for c in cells)


# ============================================================ Slice D
# cost_per_m_output_usd_self_hosted math.


def test_self_hosted_cost_math_matches_spec_worked_example() -> None:
    """Spec slice D: $5/hr, tps=100 → $5/3600 * 1e6/100 = $13.89.

    Cost math is only computed when fits=True (don't fabricate
    cost on a non-fitting deployment), so the setup uses a 7B
    FP8 model that comfortably fits in H100 80GB. An anchor cell
    forces tps=100 deterministically.

      cost/M = ($5 / 3600 sec) * (1_000_000 tokens / 100 tps)
             = 0.001388... * 10000 = 13.888...
    """
    small_model = _model(slug="llama-3-1-8b", total_params_b=7.0)
    anchor = BenchmarkCell(
        gpu_slug="h100",
        model_slug="llama-3-1-8b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=100.0,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source="public_benchmark_anchor",
        source_url="https://example.com/anchor",
        notes="",
    )
    cells = query_cost_cells(
        gpu_prices=[_gpu_price(hourly=5.0)],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[small_model],
        quantizations=[_quant()],
        bench_cells=[anchor],
        aa_observations=None,
        filters=_filters(only_fits=False, batch_size=1, context_length=4096),
    )
    h100_fp8 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert h100_fp8.fit_result is not None and h100_fp8.fit_result.fits is True
    assert h100_fp8.cost_per_m_output_usd_self_hosted == pytest.approx(13.889, rel=1e-3)


def test_self_hosted_cost_none_when_decode_tps_none() -> None:
    """When tps_estimator refuses (Tier 4, value=None — e.g. no
    anchor + batch>1), there's no honest cost denominator. The
    cell still exists (with the refusal in tps_estimate) but
    cost_per_m_output_usd_self_hosted stays None."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price(hourly=5.0)],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],  # no anchor
        aa_observations=None,
        filters=_filters(batch_size=4, context_length=4096),  # batch>1 forces Tier 4
    )
    h100_fp8 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert h100_fp8.cost_per_m_output_usd_self_hosted is None
    assert h100_fp8.tps_estimate.source == "requires_measurement"


# ============================================================ Slice E
# hosted_api_token mode.


def test_hosted_api_token_produces_correct_cell_shape() -> None:
    """Spec slice E: LLM API price row → deployment_mode=
    hosted_api_token, hourly_usd=None, pricing_type=None,
    fit_result=None, gpu_slug=None, quant_slug=None, tp_size=None,
    price_per_m_*_usd populated."""
    cells = query_cost_cells(
        gpu_prices=[],
        llm_prices=[_llm_price(provider_slug="together", input_per_m=0.88, output_per_m=0.88)],
        gpu_catalog=[],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    hosted = [c for c in cells if c.deployment_mode == "hosted_api_token"]
    assert len(hosted) == 1
    h = hosted[0]
    assert h.gpu_slug is None
    assert h.quant_slug is None
    assert h.tp_size is None
    assert h.hourly_usd is None
    assert h.pricing_type is None
    assert h.fit_result is None
    assert h.price_per_m_input_usd == 0.88
    assert h.price_per_m_output_usd == 0.88


# ============================================================ Slice F
# pricing_type=spot.


def test_spot_pricing_surfaces_with_availability_modeled_false() -> None:
    """Spec slice F: spot-priced GpuPriceRow → CostCell.
    pricing_type='spot', availability_modeled=False, caveat
    populated. The cell exists (we WILL price it), but the
    trust envelope's availability_caveat names the preemption
    risk so the LLM can surface it."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price(pricing_type="spot", hourly=1.20)],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    spot_cells = [c for c in cells if c.pricing_type == "spot"]
    assert len(spot_cells) >= 1
    assert all(c.availability_modeled is False for c in spot_cells)
    assert all("preemption" in c.availability_caveat.lower() for c in spot_cells)


def test_on_demand_pricing_carries_correct_type() -> None:
    """On-demand cells: pricing_type='on_demand'. Same
    availability caveat surface (we model PRICING, not
    RENTABILITY for either type)."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price(pricing_type="on_demand")],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    assert any(c.pricing_type == "on_demand" for c in cells)


# =========================================================== Slice G
# render_cost_cells_resource — DuckDB. (Parquet output.)


def test_render_resource_returns_readable_parquet_bytes() -> None:
    """Spec slice G + acceptance criterion 3: DuckDB is invoked
    ONLY by render_cost_cells_resource. Returns bytes that
    decode as a parquet table with the expected columns."""
    import io

    import pyarrow.parquet as pq

    from whatcanirun.plan.cost_cells import render_cost_cells_resource

    parquet_bytes = render_cost_cells_resource(
        gpu_prices=[_gpu_price()],
        llm_prices=[_llm_price(provider_slug="together", input_per_m=0.88, output_per_m=0.88)],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
    )
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0

    table = pq.read_table(io.BytesIO(parquet_bytes))
    columns = set(table.column_names)
    # Spec § CostCell schema columns we round-trip through
    # DuckDB. Trust envelope is intentionally omitted from the
    # resource projection (it's metadata that doesn't tabularize
    # cleanly; the resource format is for downstream analytics).
    for required in (
        "gpu_slug",
        "provider_slug",
        "model_slug",
        "deployment_mode",
        "decode_tps",
    ):
        assert required in columns, f"missing column {required!r}"


# ============================================================ Properties


def test_sliding_window_model_skipped_not_crashes_whole_query() -> None:
    """Copilot review (round 1): compute_fit() raises
    NotImplementedError when model.kv_cache_strategy='sliding_window'
    (M06 deferred sliding_window_size plumbing). Without a try/
    except, a single sliding-window model in the catalog aborts
    the WHOLE query_cost_cells call — every other (gpu, model,
    quant) combo is lost.

    Pin the behavior: the unsupported combination is silently
    skipped; supported combinations still emerge."""
    sliding_model = _model(slug="mistral-sliding")
    sliding_model_dict = sliding_model.model_dump()
    sliding_model_dict["kv_cache_strategy"] = "sliding_window"
    sliding = Model.model_validate(sliding_model_dict)

    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[sliding, _model()],  # one bad + one good
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    # The good model survives.
    assert any(c.model_slug == "llama-3-3-70b" for c in cells)
    # The sliding-window model produces NO cell (skipped).
    assert not any(c.model_slug == "mistral-sliding" for c in cells)


def test_envelope_cites_huggingface_when_fit_check_used_model_data() -> None:
    """Copilot review (round 1): TrustEnvelope claims confidence
    for `fit_check`, `model_architecture`, `gpu_specs`, but the
    sources list only included `computeprices`. Since
    compute_fit consumes HF-synced Model fields (n_layers,
    n_kv_heads, head_dim, total_params_b, kv_cache_strategy from
    raw_config), the envelope MUST cite `huggingface` as a
    contributing upstream so the LLM client can disclose where
    the architecture data came from.

    M03 ships `Model.last_synced_at`; that's the freshness
    signal we surface."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    hf_sources = [s for s in h100.trust_envelope.sources if s.name == "huggingface"]
    assert hf_sources, (
        "gpu_rental cell missing huggingface Source entry; "
        "confidence_breakdown['model_architecture'] cites HF data"
    )
    # Freshness map names the HF sync timestamp.
    assert "huggingface" in h100.trust_envelope.freshness


def test_benchmark_source_last_updated_uses_measured_at_not_price_freshness() -> None:
    """Copilot review (round 2): the Tier 1b public_benchmark_anchor
    Source's `last_updated` was set to `price.last_updated`
    (ComputePrices freshness), which is unrelated to when the
    benchmark itself was measured. Per the trust contract,
    Source.last_updated should reflect the benchmark's own
    freshness (BenchmarkCell.measured_at) so consumers reading
    `sources[].last_updated` see the right timestamp."""
    anchor_measured = date(2025, 11, 1)
    anchor = BenchmarkCell(
        gpu_slug="h100",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=35.2,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=anchor_measured,
        source="public_benchmark_anchor",
        source_url="https://example.com/anchor",
        notes="",
    )
    price_freshness = datetime(2026, 5, 28, tzinfo=dt.UTC)
    gpu_price = _gpu_price()
    # Manually set price.last_updated to a known later value
    object.__setattr__(gpu_price, "last_updated", price_freshness)

    cells = query_cost_cells(
        gpu_prices=[gpu_price],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[anchor],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    anchor_sources = [s for s in h100.trust_envelope.sources if s.name == "public_benchmark_anchor"]
    assert anchor_sources, "public_benchmark_anchor Source missing"
    # Source.last_updated should be the anchor's measured_at, not
    # the price row's last_updated.
    assert anchor_sources[0].last_updated.date() == anchor_measured


def test_hosted_api_envelope_populates_verify_links() -> None:
    """Copilot review (round 2): hosted_api_token envelope had
    empty verify_links — consumers had no URL to audit the
    upstream pricing source. LlmPriceRow doesn't carry a per-row
    source_url, but the ComputePrices llm-prices endpoint URL is
    a defensible baseline verification link."""
    cells = query_cost_cells(
        gpu_prices=[],
        llm_prices=[_llm_price()],
        gpu_catalog=[],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    hosted = next(c for c in cells if c.deployment_mode == "hosted_api_token")
    assert hosted.trust_envelope.verify_links, (
        "hosted_api_token envelope has empty verify_links — consumer "
        "can't audit the upstream pricing source"
    )
    assert any("computeprices.com" in link.lower() for link in hosted.trust_envelope.verify_links)


def test_gpu_rental_verify_links_includes_computeprices_endpoint() -> None:
    """Copilot review (round 2): verify_links had the provider's
    pricing page (e.g. lambdalabs.com/pricing) but the
    corresponding Source claimed the upstream was the
    ComputePrices API. verify_links must include the CP endpoint
    URL too so a consumer can match Source.name='computeprices'
    to a CP-pointing URL."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert any("computeprices.com" in link.lower() for link in h100.trust_envelope.verify_links), (
        "gpu_rental verify_links missing computeprices.com URL — Source.name="
        "'computeprices' has no matching audit link"
    )


def test_aa_source_last_updated_uses_aa_freshness_when_provided() -> None:
    """Copilot review (round 2): the artificial_analysis Source's
    last_updated was set to price.last_updated (CP freshness),
    which is unrelated to when AA was captured. Threading
    `aa_freshness` through query_cost_cells (M09 will pass the
    AA cache timestamp) gets the right signal into Source."""
    aa_row = AaModelRow.project(
        {
            "id": "u",
            "slug": "llama-3-3-instruct-70b",
            "name": "llama",
            "model_creator": {"id": "v", "name": "Meta", "slug": "meta"},
            "release_date": "2024-12-06",
            "median_output_tokens_per_second": 89.6,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        }
    )
    aa_capture_ts = datetime(2026, 5, 27, tzinfo=dt.UTC)
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=[aa_row],
        aa_data_freshness=aa_capture_ts,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    aa_sources = [s for s in h100.trust_envelope.sources if s.name == "artificial_analysis"]
    assert aa_sources
    assert aa_sources[0].last_updated == aa_capture_ts


def test_envelope_lists_source_matching_tps_provenance_aa() -> None:
    """Trust-contract gap pre-push /review caught: when the
    tps_estimate source is AA (Tier 2 provider_anchor), the
    envelope's confidence_breakdown reflects AA's 0.7 confidence,
    but the sources list must NAME artificial_analysis with the
    license_attribution string AA's ToS requires (spec/M04).
    Without that, the LLM client sees the 0.7 throughput
    confidence with no upstream cited."""
    from whatcanirun.pricing.artificial_analysis import AA_ATTRIBUTION_STRING

    aa_row = AaModelRow.project(
        {
            "id": "u",
            "slug": "llama-3-3-instruct-70b",
            "name": "llama",
            "model_creator": {"id": "v", "name": "Meta", "slug": "meta"},
            "release_date": "2024-12-06",
            "median_output_tokens_per_second": 89.6,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        }
    )
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],  # force fall-through to Tier 2 (AA)
        aa_observations=[aa_row],
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert h100.tps_estimate.source == "provider_anchor"
    aa_sources = [s for s in h100.trust_envelope.sources if s.name == "artificial_analysis"]
    assert aa_sources, (
        "AA-tier cell missing artificial_analysis Source entry; LLM "
        "client would see a confidence number with no upstream cited"
    )
    assert aa_sources[0].license_attribution == AA_ATTRIBUTION_STRING


def test_envelope_lists_source_matching_tps_provenance_bandwidth() -> None:
    """Same trust-contract gap for Tier 3 bandwidth heuristic.
    The number derives from CP's specs.memory_bandwidth_gbps —
    the envelope must name `bandwidth_heuristic` as the
    contributing source so the LLM client can disclose 'derived
    from memory bandwidth, single-stream calibration 0.75'."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],  # no anchor → forces Tier 3
        aa_observations=None,  # no AA → past Tier 2
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert h100.tps_estimate.source == "bandwidth_heuristic_single_stream"
    bw_sources = [s for s in h100.trust_envelope.sources if s.name == "bandwidth_heuristic"]
    assert bw_sources, (
        "bandwidth-heuristic cell missing bandwidth_heuristic Source "
        "entry; trust contract requires naming the upstream of every "
        "confidence value"
    )


def test_envelope_verify_links_includes_tps_source_url_when_present() -> None:
    """Tier 1a/1b populate `TpsEstimate.source_url`. The envelope
    must include it in verify_links so the LLM client can show
    the user where the anchor came from."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],  # Tier 1b — populates source_url
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    assert h100.tps_estimate.source == "public_benchmark_anchor"
    assert h100.tps_estimate.source_url is not None
    assert h100.tps_estimate.source_url in h100.trust_envelope.verify_links


def test_every_cell_carries_unmodified_availability_caveat() -> None:
    """Trust contract: every CostCell ships with the exact
    availability caveat text. M09's MCP layer surfaces it
    verbatim. A drive-by edit that softens or paraphrases
    breaks the caveat contract."""
    cells = query_cost_cells(
        gpu_prices=[_gpu_price(), _gpu_price(pricing_type="spot", hourly=1.20)],
        llm_prices=[_llm_price()],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[_anchor()],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    for c in cells:
        assert c.availability_modeled is False
        assert "Price source does not guarantee current rentable capacity" in c.availability_caveat


def test_freshness_map_includes_every_source_last_updated() -> None:
    """Copilot review (round 3): `TrustEnvelope.freshness` is the
    per-source timestamp map — every entry in `sources` should
    have its `name` keyed into `freshness` with its `last_updated`.
    The pre-R3 code hardcoded only `computeprices` + `huggingface`,
    so a tier-2 AA / tier-3 bandwidth / tier-1 anchor envelope
    cited those tiers in `sources` but never surfaced their
    timestamps in `freshness`. Consumers reading per-upstream
    staleness had to re-parse `sources[]`."""
    aa_row = AaModelRow.project(
        {
            "id": "u",
            "slug": "llama-3-3-instruct-70b",
            "name": "llama",
            "model_creator": {"id": "v", "name": "Meta", "slug": "meta"},
            "release_date": "2024-12-06",
            "median_output_tokens_per_second": 89.6,
            "median_time_to_first_token_seconds": 0.5,
            "median_time_to_first_answer_token": 1.0,
            "pricing": {},
            "evaluations": {},
        }
    )
    aa_capture_ts = datetime(2026, 5, 27, tzinfo=dt.UTC)
    cells = query_cost_cells(
        gpu_prices=[_gpu_price()],
        llm_prices=[],
        gpu_catalog=[_gpu()],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=[aa_row],
        aa_data_freshness=aa_capture_ts,
        filters=_filters(batch_size=1, context_length=4096),
    )
    h100 = next(c for c in cells if c.gpu_slug == "h100" and c.quant_slug == "fp8")
    freshness = h100.trust_envelope.freshness
    # Every Source in the envelope must have a matching freshness entry.
    for src in h100.trust_envelope.sources:
        assert src.name in freshness, (
            f"Source {src.name!r} cited in sources[] but missing from "
            f"freshness map — spec/SHARED.md says freshness is the "
            f"per-source timestamp map, not a hand-picked subset"
        )
        assert freshness[src.name] == src.last_updated
    # Specifically: AA tier-2 puts artificial_analysis in freshness.
    assert "artificial_analysis" in freshness
    assert freshness["artificial_analysis"] == aa_capture_ts


def test_hosted_api_freshness_map_includes_computeprices() -> None:
    """Same Copilot R3 rule applies to the hosted_api_token
    envelope: `freshness` should mirror the sources list, not be
    a separately-maintained constant. Currently both are
    consistent (just `computeprices`); pin it so a future
    helper change (e.g. adding HF Source to hosted_api when M09
    wraps responses) can't drift them apart silently."""
    cells = query_cost_cells(
        gpu_prices=[],
        llm_prices=[_llm_price()],
        gpu_catalog=[],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=None,
        filters=_filters(batch_size=1, context_length=4096),
    )
    hosted = next(c for c in cells if c.deployment_mode == "hosted_api_token")
    freshness = hosted.trust_envelope.freshness
    for src in hosted.trust_envelope.sources:
        assert src.name in freshness
        assert freshness[src.name] == src.last_updated


def test_render_resource_preserves_typed_columns_when_hosted_only() -> None:
    """Copilot review (round 3): `pa.Table.from_pylist(rows)`
    infers column types from the data. When a render input only
    produces `hosted_api_token` cells, the gpu_rental-specific
    columns (`gpu_slug`, `quant_slug`, `tp_size`, `hourly_usd`,
    `pricing_type`, `decode_tps`, `fits`, ...) are all-None and
    Arrow infers `pa.null()` for them — an unstable schema that
    diverges from `_empty_table` and may fail downstream readers.

    Fix: pass the documented schema to `from_pylist` for non-empty
    tables too, so the parquet output's schema is the same in
    every render."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    from whatcanirun.plan.cost_cells import render_cost_cells_resource

    parquet_bytes = render_cost_cells_resource(
        gpu_prices=[],
        llm_prices=[_llm_price()],
        gpu_catalog=[],
        model_catalog=[_model()],
        quantizations=[_quant()],
        bench_cells=[],
        aa_observations=None,
    )
    table = pq.read_table(io.BytesIO(parquet_bytes))
    schema = table.schema
    # Columns that are all-None in a hosted-only render. Arrow's
    # default inference would map these to pa.null() — the test
    # asserts the explicit documented schema wins instead.
    assert schema.field("gpu_slug").type == pa.string()
    assert schema.field("quant_slug").type == pa.string()
    assert schema.field("tp_size").type == pa.int64()
    assert schema.field("hourly_usd").type == pa.float64()
    assert schema.field("pricing_type").type == pa.string()
    assert schema.field("decode_tps").type == pa.float64()
    assert schema.field("fits").type == pa.bool_()


def test_anchor_last_updated_filters_on_tier_source() -> None:
    """Copilot review (round 7): when both an `own_measured` cell
    and a `public_benchmark_anchor` cell exist for the same
    op-point (a v2 scenario once M17 ships GuideLLM-measured
    rows), `_anchor_last_updated` must match the tier that
    `estimate_tps` actually picked — otherwise `Source.last_updated`
    reflects the wrong benchmark's freshness.

    v1's `BenchmarkCell` validator rejects `source='own_measured'`
    at construction, so the test uses `model_construct` to
    simulate the v2 row without triggering the guard."""
    from whatcanirun.plan.cost_cells import _anchor_last_updated

    public_anchor = _anchor()  # source=public_benchmark_anchor, measured_at=2026-03-15
    own_anchor = BenchmarkCell.model_construct(
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
        measured_at=date(2026, 1, 10),
        source="own_measured",
        source_url="https://example.com/own",
        notes="v2 forward-compat fixture (bypasses v1 validator).",
    )
    fallback = datetime(2026, 5, 28, tzinfo=dt.UTC)

    # The estimator picked Tier 1a — helper must return the
    # own_measured cell's measured_at, not the public anchor's.
    own_ts = _anchor_last_updated(
        [own_anchor, public_anchor],
        "h100",
        "llama-3-3-70b",
        "fp8",
        1,
        4096,
        "own_measured",
        fallback,
    )
    assert own_ts.date() == date(2026, 1, 10), (
        "_anchor_last_updated returned the wrong cell — the helper "
        "should filter on tier source, not return whichever cell "
        "appears first in bench_cells"
    )

    # The estimator picked Tier 1b — helper must return the
    # public_benchmark_anchor cell's measured_at.
    public_ts = _anchor_last_updated(
        [own_anchor, public_anchor],
        "h100",
        "llama-3-3-70b",
        "fp8",
        1,
        4096,
        "public_benchmark_anchor",
        fallback,
    )
    assert public_ts.date() == date(2026, 3, 15)


def test_anchor_last_updated_falls_back_when_no_matching_tier_cell() -> None:
    """Companion to the above: when bench_cells has cells at the
    op-point but none with the matching tier source, the helper
    falls back to `fallback` rather than misattributing
    cross-tier."""
    from whatcanirun.plan.cost_cells import _anchor_last_updated

    public_anchor = _anchor()
    fallback = datetime(2026, 5, 28, tzinfo=dt.UTC)
    # No own_measured cells in bench_cells — Tier 1a lookup should
    # fall back, not return the public_benchmark_anchor's
    # measured_at.
    ts = _anchor_last_updated(
        [public_anchor],
        "h100",
        "llama-3-3-70b",
        "fp8",
        1,
        4096,
        "own_measured",
        fallback,
    )
    assert ts == fallback
