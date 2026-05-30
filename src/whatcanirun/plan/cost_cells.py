"""M08 cost cells join layer — `CostCell`, `CostCellFilters`,
`query_cost_cells`, `render_cost_cells_resource`.

ADR-014 enforces a hard architectural split:

  - `query_cost_cells` is the TOOL-CALL HOT PATH. Pure Python list
    comprehensions over in-memory caches. NO SQL. NO DuckDB.
    Easier to debug, faster at v1 scale (~hundreds of rows), and
    keeps tool-call business logic testable without a DB.

  - `render_cost_cells_resource` is the resource materialization
    for `cost-cells://current`. DuckDB is the ONLY mechanism used
    here. This is the SOLE function in this module allowed to
    `import duckdb`.

The `test_no_sql_in_business_logic.py` grep test enforces the
split by inspecting the source of `query_cost_cells` and its
helpers, refusing any `con.sql` / `con.execute` / `import duckdb`
that leaks into the hot path.

Spec § CostCell schema, § Public surface, § Derived field math.
"""

from __future__ import annotations

import datetime as dt_module
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.benchmark_cells import BenchmarkCell, BenchmarkSource
from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.inference.fit_check import FitResult, compute_fit
from whatcanirun.inference.tps_estimator import TpsEstimate, estimate_tps
from whatcanirun.pricing.artificial_analysis import AaModelRow, ReasoningEffort
from whatcanirun.pricing.projections import GpuCatalogRow, GpuPriceRow, LlmPriceRow
from whatcanirun.trust.envelope import Source, TrustEnvelope

DeploymentMode = Literal["cloud_gpu_rental", "hosted_api_token"]
PricingType = Literal["on_demand", "spot"]

# The CP API endpoints that contributed each cell's data —
# verify_links cite these so the LLM client can show the user
# exactly which CP endpoint to audit. Per-row Source.url
# (gpu_price.source_url is the provider's PRICING PAGE, not the
# CP API) goes alongside.
_CP_GPU_PRICES_URL = "https://www.computeprices.com/api/v1/gpu-prices"
_CP_LLM_PRICES_URL = "https://www.computeprices.com/api/v1/llm-prices"

# Verbatim per spec § CostCell schema. The text is part of the
# trust contract — M09 surfaces this in tool responses, and a
# drive-by edit that softens or paraphrases breaks the contract.
# Centralizing as a module constant prevents drift between schema
# default and runtime construction.
_AVAILABILITY_CAVEAT = (
    "Price source does not guarantee current rentable capacity. Spot pricing "
    "is also subject to preemption and minimum-commitment terms not modeled here."
)


class CostCell(BaseModel):
    """One row of the cost cells projection: a single (gpu,
    provider, model, quant, batch, ctx) combination priced + sized
    against M06's fit_check and anchored to M07's TpsEstimate.
    Carries a TrustEnvelope so the LLM client can disclose every
    contributing source.
    """

    model_config = ConfigDict(extra="forbid")

    gpu_slug: str | None
    provider_slug: str
    model_slug: str
    quant_slug: str | None = None
    tp_size: int | None = None
    batch_size: int
    context_length: int
    deployment_mode: DeploymentMode

    hourly_usd: float | None = None
    pricing_type: PricingType | None = None
    price_per_m_input_usd: float | None = None
    price_per_m_output_usd: float | None = None

    decode_tps: float | None = None
    tps_estimate: TpsEstimate
    fit_result: FitResult | None = None
    cost_per_m_output_usd_self_hosted: float | None = None

    availability_modeled: bool = False
    availability_caveat: str = _AVAILABILITY_CAVEAT

    trust_envelope: TrustEnvelope


@dataclass
class CostCellFilters:
    """Filter set for `query_cost_cells`. None means 'don't
    filter on this dimension'. `only_fits=True` excludes rows
    where M06's fit_check returned `fits=False`. `batch_size` and
    `context_length` are op-point parameters — they always
    influence the M06/M07 lookups even when not used as
    filters.

    Cross-dimension note: a non-None `gpu_slug` or `quant_slug`
    also implicitly suppresses the `hosted_api_token` branch
    (hosted-API rows have `gpu_slug=None` / `quant_slug=None` and
    can't satisfy those filters). To force hosted rows through
    that suppression, pass `deployment_mode='hosted_api_token'`
    explicitly — the explicit mode wins."""

    model_slug: str | None = None
    gpu_slug: str | None = None
    provider_slug: str | None = None
    quant_slug: str | None = None
    batch_size: int = 1
    context_length: int = 4096
    deployment_mode: DeploymentMode | None = None
    only_fits: bool = False
    workload_profile_slug: str | None = None
    reasoning_effort: ReasoningEffort | None = None


def query_cost_cells(
    *,
    gpu_prices: list[GpuPriceRow],
    llm_prices: list[LlmPriceRow],
    gpu_catalog: list[GpuCatalogRow],
    model_catalog: list[Model],
    quantizations: list[Quantization],
    bench_cells: list[BenchmarkCell],
    aa_observations: list[AaModelRow] | None,
    filters: CostCellFilters,
    aa_data_freshness: datetime | None = None,
) -> list[CostCell]:
    """Tool-call hot path. Pure Python list comprehensions over
    in-memory caches. NO SQL.

    For each cloud_gpu_rental candidate (gpu_price x model x quant)
    we run M06's compute_fit and M07's estimate_tps, then build a
    CostCell. For each hosted_api_token candidate (llm_price x
    model) we skip fit/tps math (provider runs the model) and
    just project the per-token prices.

    Filtering happens at three points:
      1. Pre-iteration: prune gpu_prices / llm_prices / models /
         quants by filter keys before the cross-product
      2. Per-combination: only_fits skips after compute_fit
      3. Per-mode: deployment_mode filter skips entire branches

    Performance: at v1 scale (~1000 gpu_prices x 17 models x 5
    quants = ~85k candidates) this is ~10ms with no DB round-trip.
    SQL would add startup + serialize overhead with no win.
    """
    cells: list[CostCell] = []

    # Build a slug → catalog lookup for fast gpu resolution.
    gpu_by_slug = {g.slug: g for g in gpu_catalog}

    # Pre-filter all dimensions.
    models_to_consider = [
        m for m in model_catalog if filters.model_slug is None or m.slug == filters.model_slug
    ]
    quants_to_consider = [
        q for q in quantizations if filters.quant_slug is None or q.slug == filters.quant_slug
    ]

    # ---------- cloud_gpu_rental branch ----------
    if filters.deployment_mode != "hosted_api_token":
        for price in gpu_prices:
            if filters.gpu_slug is not None and price.gpu_slug != filters.gpu_slug:
                continue
            if filters.provider_slug is not None and price.provider_slug != filters.provider_slug:
                continue
            gpu = gpu_by_slug.get(price.gpu_slug)
            if gpu is None:
                # Price row references a GPU we don't have catalog
                # data for — skip rather than crash. Honest data
                # gap surfacing happens at the M09 trust envelope.
                continue

            for model in models_to_consider:
                if model.total_params_b is None:
                    # Without total_params_b we can't compute weight
                    # bytes, so fit_check has nothing to compare
                    # against VRAM. Skip rather than emit a partial
                    # cell — M09's `budget_to_plan` surfaces the
                    # data gap via its workload-elicitation flow.
                    continue

                for quant in quants_to_consider:
                    try:
                        fit = compute_fit(
                            model=model,
                            gpu=gpu,
                            quant=quant,
                            tp_size=1,  # v1: single-GPU; tp>1 is M09+
                            batch_size=filters.batch_size,
                            context_length=filters.context_length,
                        )
                    except NotImplementedError:
                        # M06 raises for kv_cache_strategy='sliding_window'
                        # (deferred until sliding_window_size plumbing
                        # lands). Skip this combination rather than
                        # aborting the whole query — other (gpu, model,
                        # quant) combos in this batch are still useful.
                        # The unsupported combo simply doesn't appear in
                        # the cost cells; M09's missing-cell handling
                        # surfaces this honestly to the user when they
                        # query a sliding-window model.
                        continue
                    if filters.only_fits and not fit.fits:
                        continue

                    tps = estimate_tps(
                        model=model,
                        gpu=gpu,
                        quant=quant,
                        batch_size=filters.batch_size,
                        context_length=filters.context_length,
                        bench_cells=bench_cells,
                        aa_observations=aa_observations,
                        reasoning_effort=filters.reasoning_effort,
                    )

                    decode_tps = tps.value
                    cost_per_m = _self_hosted_cost(
                        hourly_usd=price.price_per_hour_usd,
                        decode_tps=decode_tps,
                        fits=fit.fits,
                    )

                    # CP carries an additional "reserved" pricing
                    # type our CostCell schema doesn't model in v1
                    # (reserved deployment math is M2-out-of-scope
                    # per spec § Out of scope). Map it to None on
                    # the cell rather than dropping the row — the
                    # price + fit + tps are still useful info; the
                    # pricing_type just isn't one of our two
                    # closed-Literal values.
                    cell_pricing_type: PricingType | None = (
                        price.pricing_type if price.pricing_type in ("on_demand", "spot") else None
                    )

                    cells.append(
                        CostCell(
                            gpu_slug=price.gpu_slug,
                            provider_slug=price.provider_slug,
                            model_slug=model.slug,
                            quant_slug=quant.slug,
                            tp_size=1,
                            batch_size=filters.batch_size,
                            context_length=filters.context_length,
                            deployment_mode="cloud_gpu_rental",
                            hourly_usd=price.price_per_hour_usd,
                            pricing_type=cell_pricing_type,
                            price_per_m_input_usd=None,
                            price_per_m_output_usd=None,
                            decode_tps=decode_tps,
                            tps_estimate=tps,
                            fit_result=fit,
                            cost_per_m_output_usd_self_hosted=cost_per_m,
                            trust_envelope=_partial_envelope_for_gpu_rental(
                                price=price,
                                tps=tps,
                                model=model,
                                bench_cells=bench_cells,
                                gpu_slug=price.gpu_slug,
                                model_slug=model.slug,
                                quant_slug=quant.slug,
                                batch_size=filters.batch_size,
                                context_length=filters.context_length,
                                aa_data_freshness=aa_data_freshness,
                            ),
                        )
                    )

    # ---------- hosted_api_token branch ----------
    # Skip entirely when the caller asks for gpu_slug or quant_slug
    # filtering — hosted-API rows have `gpu_slug=None` /
    # `quant_slug=None` (the provider runs the weights, not on the
    # user's GPU), so they can't satisfy those filters. Exception:
    # an explicit `deployment_mode="hosted_api_token"` is the
    # stronger signal — honor it even if gpu/quant filters look
    # superficially inconsistent.
    explicit_hosted = filters.deployment_mode == "hosted_api_token"
    hosted_filtered_out = not explicit_hosted and (
        filters.gpu_slug is not None or filters.quant_slug is not None
    )
    if filters.deployment_mode != "cloud_gpu_rental" and not hosted_filtered_out:
        for llm_price in llm_prices:
            if (
                filters.provider_slug is not None
                and llm_price.provider_slug != filters.provider_slug
            ):
                continue
            if filters.model_slug is not None and llm_price.model_slug != filters.model_slug:
                continue
            # No Model lookup here — the provider runs the weights,
            # so we don't need n_layers / n_kv_heads / head_dim /
            # any of M06's inputs. If a future v1 row needs Model
            # metadata (e.g. model_creator on the trust envelope),
            # add the lookup THEN — not pre-emptively.

            # No fit_check, no tps math — hosted API doesn't load
            # weights into OUR VRAM. Per spec § Common pitfalls:
            # "hosted_api_token doesn't need fit_check. Set
            # fit_result=None and skip the math."
            cells.append(
                CostCell(
                    gpu_slug=None,
                    provider_slug=llm_price.provider_slug,
                    model_slug=llm_price.model_slug,
                    quant_slug=None,
                    tp_size=None,
                    batch_size=filters.batch_size,
                    context_length=filters.context_length,
                    deployment_mode="hosted_api_token",
                    hourly_usd=None,
                    pricing_type=None,
                    price_per_m_input_usd=llm_price.price_per_1m_input_usd,
                    price_per_m_output_usd=llm_price.price_per_1m_output_usd,
                    decode_tps=None,
                    tps_estimate=TpsEstimate(
                        value=None,
                        source="requires_measurement",
                        confidence=0.0,
                        refusal_reason=(
                            "hosted_api_token throughput is provider-internal and not modeled here"
                        ),
                    ),
                    fit_result=None,
                    cost_per_m_output_usd_self_hosted=None,
                    trust_envelope=_partial_envelope_for_hosted_api(llm_price),
                )
            )

    return cells


def _self_hosted_cost(
    *,
    hourly_usd: float,
    decode_tps: float | None,
    fits: bool,
) -> float | None:
    """Per spec § Derived field math:
        cost_per_m_output_usd_self_hosted =
            (hourly_usd / 3600) * (1_000_000 / decode_tps)
    Only computed when fits=True AND decode_tps is non-None. Any
    other combination returns None — never fabricate a cost from
    a non-fitting deployment or a requires_measurement throughput.
    """
    if not fits or decode_tps is None or decode_tps <= 0:
        return None
    return (hourly_usd / 3600.0) * (1_000_000.0 / decode_tps)


def _partial_envelope_for_gpu_rental(
    *,
    price: GpuPriceRow,
    tps: TpsEstimate,
    model: Model,
    bench_cells: list[BenchmarkCell],
    gpu_slug: str,
    model_slug: str,
    quant_slug: str,
    batch_size: int,
    context_length: int,
    aa_data_freshness: datetime | None,
) -> TrustEnvelope:
    """M08 builds the per-cell envelope with the data it has
    direct access to: pricing source + freshness, throughput
    confidence from M07's TpsEstimate, model-architecture source
    from M03's HF sync. M09 ENRICHES this at MCP-wrap time with
    workload_assumption (when synthesizing counts), full caveats
    list, etc.

    Trust contract: every contributing source MUST be named in
    the envelope's sources list. A confidence_breakdown value
    coming from AA (Tier 2 provider_anchor) without a matching
    `artificial_analysis` Source entry would leave the LLM
    client unable to cite the upstream. Same for bandwidth
    heuristic (Tier 3) — even though the number derives from
    CP's specs.memory_bandwidth_gbps which we already cite, the
    heuristic IS a distinct provenance worth naming so M09 can
    surface "single-stream calibration 0.75, ±50% accuracy" in
    the caveats list. And `confidence_breakdown['model_architecture']`
    is anchored on HF data via compute_fit (n_layers, n_kv_heads,
    head_dim, raw_config) — so `huggingface` MUST appear in
    sources + freshness with the M03 sync timestamp.

    `tps.source_url` (populated for Tier 1a/1b anchors) is
    threaded into verify_links so the LLM client can link the
    user to the original methodology disclosure.
    """
    sources = [
        Source(
            name="computeprices",
            detail=f"GET /api/v1/gpu-prices, {price.provider_slug}/{price.gpu_slug}",
            last_updated=price.last_updated,
        ),
        Source(
            name="huggingface",
            detail=f"config.json + family detection for {model.hf_repo_id}",
            last_updated=model.last_synced_at,
        ),
    ]
    # verify_links cite BOTH the CP API endpoint (the actual
    # upstream that contributed the pricing data; matches
    # Source.name='computeprices') AND the provider's pricing
    # page (which the price row's source_url points at — gives
    # the user a human-readable view).
    verify_links = [_CP_GPU_PRICES_URL, price.source_url]

    # Anchor freshness: when the TPS number came from an explicit
    # BenchmarkCell row (Tier 1a/1b), Source.last_updated should
    # reflect the BENCHMARK's measured_at, not the GPU price's
    # last_updated (which is a CP catalog freshness, unrelated to
    # when the anchor was actually measured). The lookup happens
    # ONLY inside the Tier 1a/1b branches that consume it — Tier
    # 2/3/4 don't need it, and scanning bench_cells unconditionally
    # would add avoidable O(len(bench_cells)) work per cell.

    if tps.source == "own_measured":
        # v2 only — but the dispatch logic is here so the v2
        # unlock is a no-op at the trust envelope layer.
        anchor_ts = _anchor_last_updated(
            bench_cells,
            gpu_slug,
            model_slug,
            quant_slug,
            batch_size,
            context_length,
            "own_measured",
            price.last_updated,
        )
        sources.append(
            Source(
                name="own_measured_benchmark",
                detail=tps.anchor_detail or "",
                last_updated=anchor_ts,
            )
        )
        if tps.source_url:
            verify_links.append(tps.source_url)
    elif tps.source == "public_benchmark_anchor":
        anchor_ts = _anchor_last_updated(
            bench_cells,
            gpu_slug,
            model_slug,
            quant_slug,
            batch_size,
            context_length,
            "public_benchmark_anchor",
            price.last_updated,
        )
        sources.append(
            Source(
                name="public_benchmark_anchor",
                detail=tps.anchor_detail or "",
                last_updated=anchor_ts,
            )
        )
        if tps.source_url:
            verify_links.append(tps.source_url)
    elif tps.source == "provider_anchor":
        # AA contributed the throughput number. AA's free-tier
        # license requires the verbatim attribution string on
        # every consumer-visible source entry — that's the M04
        # AA_ATTRIBUTION_STRING constant. Caller (M09) threads
        # aa_data_freshness through so Source.last_updated
        # reflects the AA cache timestamp; falls back to
        # price.last_updated when not supplied (less accurate but
        # honest about which signal we have).
        from whatcanirun.pricing.artificial_analysis import (
            AA_ATTRIBUTION_STRING,
        )

        sources.append(
            Source(
                name="artificial_analysis",
                detail=tps.anchor_detail or "AA median_output_tokens_per_second",
                last_updated=aa_data_freshness or price.last_updated,
                license_attribution=AA_ATTRIBUTION_STRING,
            )
        )
    elif tps.source == "bandwidth_heuristic_single_stream":
        # The number derives from CP's specs.memory_bandwidth_gbps
        # which the computeprices Source already cites — using the
        # price's last_updated is the right signal here (it's CP's
        # freshness). M09 surfaces "single-stream calibration
        # KERNEL_EFFICIENCY=0.75, ±50% at small batch" in the
        # caveats so the user understands this isn't a measured
        # anchor.
        sources.append(
            Source(
                name="bandwidth_heuristic",
                detail=tps.anchor_detail or "single-stream bandwidth heuristic",
                last_updated=price.last_updated,
            )
        )
    # `requires_measurement` (Tier 4) refuses honestly — no
    # additional source to cite. The throughput confidence is
    # 0.0 in confidence_breakdown which the LLM client surfaces
    # as "we don't have data for this combination".

    return TrustEnvelope(
        sources=sources,
        confidence_breakdown={
            "pricing": 0.95,
            "fit_check": 0.9,
            "throughput": tps.confidence,
            "model_architecture": 0.9,
            "gpu_specs": 0.85,
            "freshness": 0.8,
        },
        freshness=_freshness_from_sources(sources),
        verify_links=verify_links,
    )


def _partial_envelope_for_hosted_api(price: LlmPriceRow) -> TrustEnvelope:
    sources = [
        Source(
            name="computeprices",
            detail=f"GET /api/v1/llm-prices, {price.provider_slug}/{price.model_slug}",
            last_updated=price.last_updated,
        )
    ]
    return TrustEnvelope(
        sources=sources,
        confidence_breakdown={
            "pricing": 0.95,
            "throughput": 0.0,  # hosted API throughput not modeled
            "freshness": 0.8,
        },
        freshness=_freshness_from_sources(sources),
        # LlmPriceRow doesn't carry a per-row source_url (CP's
        # response only puts that on gpu-prices). The CP API
        # endpoint is the right baseline audit link — Source.name=
        # 'computeprices' resolves cleanly to this URL.
        verify_links=[_CP_LLM_PRICES_URL],
    )


def _freshness_from_sources(sources: list[Source]) -> dict[str, datetime]:
    """Project the assembled sources list into the freshness map
    spec/SHARED.md mandates. Keeping the two in lockstep (rather
    than hand-maintaining a parallel dict literal) makes it
    impossible to cite a Source whose timestamp the consumer can't
    look up — the M08 round-3 Copilot finding."""
    return {src.name: src.last_updated for src in sources}


def _anchor_last_updated(
    bench_cells: list[BenchmarkCell],
    gpu_slug: str,
    model_slug: str,
    quant_slug: str,
    batch_size: int,
    context_length: int,
    tier_source: BenchmarkSource,
    fallback: datetime,
) -> datetime:
    """Tier 1a/1b helper: return the matched BenchmarkCell's
    measured_at as a UTC datetime, or `fallback` (typically the
    price's last_updated) when no cell matches. Lookup is deferred
    into this helper so callers in non-anchor tiers (Tier 2/3/4)
    avoid the O(len(bench_cells)) scan they don't need.

    `tier_source` mirrors the BenchmarkCell.source value
    `estimate_tps` matched on (`own_measured` for Tier 1a,
    `public_benchmark_anchor` for Tier 1b). v2 will see both
    kinds of cells for the same op-point; filtering on tier
    source here keeps Source.last_updated aligned with the
    cell estimate_tps actually selected, rather than whichever
    appears first in bench_cells."""
    cell = _find_matched_bench_cell(
        bench_cells=bench_cells,
        gpu_slug=gpu_slug,
        model_slug=model_slug,
        quant_slug=quant_slug,
        batch_size=batch_size,
        context_length=context_length,
        tier_source=tier_source,
    )
    if cell is None:
        return fallback
    return datetime.combine(cell.measured_at, datetime.min.time(), tzinfo=dt_module.UTC)


def _find_matched_bench_cell(
    *,
    bench_cells: list[BenchmarkCell],
    gpu_slug: str,
    model_slug: str,
    quant_slug: str,
    batch_size: int,
    context_length: int,
    tier_source: BenchmarkSource,
) -> BenchmarkCell | None:
    """Find the BenchmarkCell row that M07's estimate_tps would
    have matched for this op-point and tier — used by the
    envelope helper so Source.last_updated reflects the BENCHMARK's
    measured_at rather than the GPU price's freshness. Mirrors
    the matching rule in
    `whatcanirun.inference.tps_estimator._cell_matches` (tp_size=1
    single-GPU) PLUS the per-tier source filter (estimate_tps
    scans for own_measured first, then public_benchmark_anchor —
    this helper has to match the SAME tier the estimator picked,
    otherwise v2's mixed-tier bench_cells confuses provenance)."""
    for cell in bench_cells:
        if (
            cell.source == tier_source
            and cell.gpu_slug == gpu_slug
            and cell.model_slug == model_slug
            and cell.quant_slug == quant_slug
            and cell.tp_size == 1
            and cell.batch_size == batch_size
            and cell.context_length == context_length
        ):
            return cell
    return None


# ============================================================ Resource path
# ADR-014 — DuckDB allowed ONLY below this line. The
# test_no_sql_in_business_logic grep test guards the surface
# above.


def render_cost_cells_resource(
    *,
    gpu_prices: list[GpuPriceRow],
    llm_prices: list[LlmPriceRow],
    gpu_catalog: list[GpuCatalogRow],
    model_catalog: list[Model],
    quantizations: list[Quantization],
    bench_cells: list[BenchmarkCell],
    aa_observations: list[AaModelRow] | None,
) -> bytes:
    """Materialize ALL current cost cells as Parquet bytes for the
    `cost-cells://current` MCP resource. The ONLY function in this
    module that uses DuckDB.

    Implementation note: rather than re-implementing the join math
    in SQL (which would mean re-implementing fit_check + tps in
    DuckDB UDFs), this function calls `query_cost_cells` to get
    the in-memory rows and uses DuckDB only to project them into
    Parquet via Arrow. The DuckDB invocation is for the
    Parquet-writing machinery, not for the join logic.
    """
    import io

    import duckdb
    import pyarrow as pa

    cells = query_cost_cells(
        gpu_prices=gpu_prices,
        llm_prices=llm_prices,
        gpu_catalog=gpu_catalog,
        model_catalog=model_catalog,
        quantizations=quantizations,
        bench_cells=bench_cells,
        aa_observations=aa_observations,
        filters=CostCellFilters(),
    )

    # Project each cell to a flat dict suitable for tabular
    # storage. trust_envelope is structured metadata; only the
    # rolled-up `confidence` (as `trust_confidence`) round-trips
    # here. The full envelope tree (sources, freshness, caveats,
    # verify_links) stays accessible via `query_cost_cells` for
    # callers that need it — the parquet resource is a flat
    # analytics view, not a provenance dump.
    rows: list[dict[str, Any]] = []
    for c in cells:
        rows.append(
            {
                "gpu_slug": c.gpu_slug,
                "provider_slug": c.provider_slug,
                "model_slug": c.model_slug,
                "quant_slug": c.quant_slug,
                "tp_size": c.tp_size,
                "batch_size": c.batch_size,
                "context_length": c.context_length,
                "deployment_mode": c.deployment_mode,
                "hourly_usd": c.hourly_usd,
                "pricing_type": c.pricing_type,
                "price_per_m_input_usd": c.price_per_m_input_usd,
                "price_per_m_output_usd": c.price_per_m_output_usd,
                "decode_tps": c.decode_tps,
                "tps_source": c.tps_estimate.source,
                "tps_confidence": c.tps_estimate.confidence,
                "fits": c.fit_result.fits if c.fit_result is not None else None,
                "cost_per_m_output_usd_self_hosted": c.cost_per_m_output_usd_self_hosted,
                "availability_modeled": c.availability_modeled,
                "trust_confidence": c.trust_envelope.confidence,
            }
        )

    # Pass the documented schema explicitly so all-None columns
    # (e.g. gpu_slug / tp_size / hourly_usd in a hosted-API-only
    # render) keep their typed form instead of being inferred as
    # pa.null() — Copilot R3 finding. Schema MUST match
    # `_empty_table` so the resource format is the same across
    # renders.
    table = pa.Table.from_pylist(rows, schema=_resource_schema()) if rows else _empty_table()
    # `with` ensures the native DuckDB handle is released even if
    # pyarrow raises mid-write — leaving the connection open in a
    # long-running MCP server accumulates state across repeated
    # `cost-cells://current` reads.
    with duckdb.connect(":memory:") as con:
        con.register("cells", table)
        # `.sql(...).arrow()` returns a RecordBatchReader; tabularize
        # via to_arrow_table for the parquet write path.
        arrow_table = con.sql("SELECT * FROM cells").to_arrow_table()

    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(arrow_table, buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()


def _resource_schema() -> Any:
    """Documented Arrow schema for the cost-cells resource —
    shared between `_empty_table` (the zero-rows case) and the
    `from_pylist(rows, schema=...)` non-empty path so all-None
    columns can't degrade to `pa.null()` (R3 finding)."""
    import pyarrow as pa

    return pa.schema(
        [
            ("gpu_slug", pa.string()),
            ("provider_slug", pa.string()),
            ("model_slug", pa.string()),
            ("quant_slug", pa.string()),
            ("tp_size", pa.int64()),
            ("batch_size", pa.int64()),
            ("context_length", pa.int64()),
            ("deployment_mode", pa.string()),
            ("hourly_usd", pa.float64()),
            ("pricing_type", pa.string()),
            ("price_per_m_input_usd", pa.float64()),
            ("price_per_m_output_usd", pa.float64()),
            ("decode_tps", pa.float64()),
            ("tps_source", pa.string()),
            ("tps_confidence", pa.float64()),
            ("fits", pa.bool_()),
            ("cost_per_m_output_usd_self_hosted", pa.float64()),
            ("availability_modeled", pa.bool_()),
            ("trust_confidence", pa.float64()),
        ]
    )


def _empty_table() -> Any:
    """Empty pyarrow.Table with the documented schema. Allows
    the resource to render valid Parquet bytes even when no
    cells exist (M09's resource consumers need a parseable
    response, not a None)."""
    import pyarrow as pa

    return pa.Table.from_pylist([], schema=_resource_schema())
