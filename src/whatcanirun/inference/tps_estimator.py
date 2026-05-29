"""Pure-math throughput estimator with 5-tier provenance.

`estimate_tps` returns a `TpsEstimate` carrying a single decode-TPS
value and an explicit source tag. Each tier has a fixed
confidence value; the function refuses honestly
(`source="requires_measurement"`, value=None) when no tier
applies. v1 NEVER produces `source="own_measured"` — that's
v2's GuideLLM-measured cells only; the BenchmarkCell validator
rejects own_measured rows at construction, so this code branch
is dead in v1 production but tested via `model_construct` so the
v2 unlock is a single validator flip.

Decision tree (lower number wins when multiple match):
  1a  own_measured                 0.95  v2 only
  1b  public_benchmark_anchor      0.80  v1 default
  2   provider_anchor (AA median)  0.7   batch=1 + AA row matches
  3   bandwidth_heuristic          0.6   batch=1 + has bandwidth
  4   requires_measurement         0.0   refusal

NO I/O. NO globals (KERNEL_EFFICIENCY_SINGLE_STREAM is a named
constant per spec acceptance criterion). NO DB. Purity is part
of the contract M08/M09 callers rely on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.pricing.artificial_analysis import AaModelRow, ReasoningEffort
from whatcanirun.pricing.projections import GpuCatalogRow

# Per spec § Decision tree Tier 3. Anchored to verified anchors:
#   - Llama-3.3-70B FP8 H100 SXM batch=1: formula 35.6 tok/s,
#     real ~35 tok/s (Spheron) — within 1%
#   - Llama-3.1-8B BF16 H100 SXM batch=1: formula 157 tok/s,
#     real ~100 tok/s — within ±50% (single-stream calibration
#     varies by kernel; acceptable per ADR-010)
# A drive-by edit raising this above 1.0 would silently bias
# every Tier-3 estimate optimistic — pinned by test.
KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75

# Spec § Decision tree fixes these exactly. Any drift would
# silently change the confidence surface every trust envelope
# is reasoning over.
_CONFIDENCE_OWN_MEASURED = 0.95
_CONFIDENCE_PUBLIC_ANCHOR = 0.80
_CONFIDENCE_PROVIDER_ANCHOR = 0.7
_CONFIDENCE_BANDWIDTH = 0.6
_CONFIDENCE_REFUSAL = 0.0


TpsSource = Literal[
    "own_measured",
    "public_benchmark_anchor",
    "provider_anchor",
    "bandwidth_heuristic_single_stream",
    "requires_measurement",
]


class TpsEstimate(BaseModel):
    """One TPS estimate with explicit provenance. M08/M09 callers
    surface `source` and `confidence` verbatim in the trust
    envelope so the LLM client can disclose how the number was
    arrived at."""

    model_config = ConfigDict(extra="forbid")

    value: float | None
    source: TpsSource
    confidence: float
    anchor_detail: str | None = None
    source_url: str | None = None
    refusal_reason: str | None = None


# spec/M07 § Decision tree Tier 4 — exact text. Module-level so
# the caveat surface stays pinned and we never drift between
# acceptance test and runtime message.
_REFUSAL_REASON = (
    "batched throughput not modeled by heuristic. "
    "Submit a benchmark cell, switch to batch=1 single-stream "
    "estimate, or accept that this combination cannot be priced honestly."
)


def estimate_tps(
    *,
    model: Model,
    gpu: GpuCatalogRow,
    quant: Quantization,
    batch_size: int,
    context_length: int,
    bench_cells: list[BenchmarkCell],
    aa_observations: list[AaModelRow] | None,
    reasoning_effort: ReasoningEffort | None = None,
) -> TpsEstimate:
    """Walk the 5-tier provenance ladder. Returns the first tier
    that applies. Pure function — no I/O, no globals beyond named
    constants.

    `bench_cells`: candidate anchor rows (the full M10 seed
    parquet, passed in by the caller). Filtering by
    `(gpu_slug, model_slug, quant_slug, batch, ctx)` happens
    here.

    `aa_observations`: AA rows already filtered to those matching
    this model via `aa_slug_mapping` resolution. M09's dispatcher
    is the one that calls `resolve_aa_slug`; M07 takes the
    already-filtered list.

    `reasoning_effort`: for reasoning models, only AA rows with
    matching `reasoning_effort` count for Tier 2.

    `batch_size <= 0` or `context_length <= 0` raise `ValueError`
    — degenerate inputs are not "doesn't fit" cases, they're
    domain errors callers must fix.
    """
    if batch_size <= 0 or context_length <= 0:
        raise ValueError(
            f"batch_size and context_length must be positive; "
            f"got batch_size={batch_size}, context_length={context_length}"
        )

    # Tier 1a: own_measured exact match. v1 never reaches a true
    # branch here because BenchmarkCell rejects own_measured at
    # construction; the logic is here for v2 unlock + tier-
    # ordering test coverage. Tier 1a runs BEFORE 1b so a future
    # v2 row beats a v1 anchor for the same op-point.
    for cell in bench_cells:
        if cell.source == "own_measured" and _cell_matches(
            cell, gpu, model, quant, batch_size, context_length
        ):
            return TpsEstimate(
                value=cell.decode_tps,
                source="own_measured",
                confidence=_CONFIDENCE_OWN_MEASURED,
                anchor_detail=(
                    f"GuideLLM-measured anchor: {cell.engine} "
                    f"{cell.engine_version}, measured {cell.measured_at}"
                ),
                source_url=cell.source_url,
            )

    # Tier 1b: public_benchmark_anchor exact match. v1's default
    # provenance for op-points covered by the M10 seed parquet.
    for cell in bench_cells:
        if cell.source == "public_benchmark_anchor" and _cell_matches(
            cell, gpu, model, quant, batch_size, context_length
        ):
            return TpsEstimate(
                value=cell.decode_tps,
                source="public_benchmark_anchor",
                confidence=_CONFIDENCE_PUBLIC_ANCHOR,
                anchor_detail=(
                    f"Public anchor: {cell.engine} {cell.engine_version}, "
                    f"measured {cell.measured_at}. {cell.notes}"
                ),
                source_url=cell.source_url,
            )

    # Tier 2: AA provider_anchor. Only fires at batch_size==1
    # (AA's median is a single-stream aggregate; spec § Common
    # pitfalls forbids scaling it with batch). Reasoning models
    # need the AA row's reasoning_effort to match the requested
    # one — wrong-effort rows DON'T match.
    if batch_size == 1 and aa_observations:
        aa_row = _aa_observation_for_effort(aa_observations, reasoning_effort)
        if aa_row is not None and aa_row.median_output_tokens_per_second is not None:
            return TpsEstimate(
                value=aa_row.median_output_tokens_per_second,
                source="provider_anchor",
                confidence=_CONFIDENCE_PROVIDER_ANCHOR,
                anchor_detail=(
                    f"AA (Artificial Analysis) serving aggregate across "
                    f"providers for {aa_row.slug!r}; "
                    f"specific GPU and batch are not modeled."
                ),
            )

    # Tier 3: bandwidth heuristic. Only fires at batch_size==1
    # AND when CP's specs carry a memory_bandwidth_gbps. If the
    # bandwidth is missing (data gap on CP's side), fall through
    # to Tier 4 rather than divide by zero.
    if batch_size == 1 and model.total_params_b is not None:
        bandwidth_gbps = _memory_bandwidth_gbps(gpu)
        if bandwidth_gbps is not None and bandwidth_gbps > 0:
            weights_bytes_per_token = model.total_params_b * 1e9 * quant.bits_per_weight / 8.0
            if weights_bytes_per_token > 0:
                peak_tps = bandwidth_gbps * 1e9 / weights_bytes_per_token
                value = peak_tps * KERNEL_EFFICIENCY_SINGLE_STREAM
                return TpsEstimate(
                    value=value,
                    source="bandwidth_heuristic_single_stream",
                    confidence=_CONFIDENCE_BANDWIDTH,
                    anchor_detail=(
                        f"Single-stream bandwidth heuristic: "
                        f"{bandwidth_gbps} GB/s memory bandwidth, "
                        f"kernel efficiency {KERNEL_EFFICIENCY_SINGLE_STREAM} "
                        f"per Inference Engineering §2.4.2."
                    ),
                )

    # Tier 4: refusal. value=None per spec — never fabricate.
    return TpsEstimate(
        value=None,
        source="requires_measurement",
        confidence=_CONFIDENCE_REFUSAL,
        refusal_reason=_REFUSAL_REASON,
    )


# ---------------------------------------------------------------- internals


def _cell_matches(
    cell: BenchmarkCell,
    gpu: GpuCatalogRow,
    model: Model,
    quant: Quantization,
    batch_size: int,
    context_length: int,
) -> bool:
    """Exact (gpu, model, quant, batch, ctx) match. tp_size is in
    the cell's primary key but not in `estimate_tps`'s signature
    today — M07 only ranks tp_size=1 anchors (v1 ships single-
    GPU). M07-tps-estimator.md slice 1 example uses tp_size=1
    implicitly; matching on tp_size=1 here keeps that assumption
    explicit. A future tp_size>1 op-point will need this filter
    parametrized."""
    return (
        cell.gpu_slug == gpu.slug
        and cell.model_slug == model.slug
        and cell.quant_slug == quant.slug
        and cell.tp_size == 1
        and cell.batch_size == batch_size
        and cell.context_length == context_length
    )


def _aa_observation_for_effort(
    aa_observations: list[AaModelRow],
    requested_effort: ReasoningEffort | None,
) -> AaModelRow | None:
    """Find the AA row whose `reasoning_effort` matches the
    request. For non-reasoning queries (requested=None), only AA
    rows with reasoning_effort=None count — a `-low`/`-medium`/
    `-high` reasoning row should NOT silently fall through to
    cover a non-reasoning query."""
    for row in aa_observations:
        if row.reasoning_effort == requested_effort:
            return row
    return None


def _memory_bandwidth_gbps(gpu: GpuCatalogRow) -> float | None:
    """CP's specs dict carries `memory_bandwidth_gbps` (not the
    underscore form spec/M07 § Formulas uses — minor naming
    inconsistency; CP's is authoritative because that's what the
    live response actually carries). Returns None when the field
    is absent or non-numeric so Tier 3 falls through cleanly."""
    raw: Any = gpu.specs.get("memory_bandwidth_gbps")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None
