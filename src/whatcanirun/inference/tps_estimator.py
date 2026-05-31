"""Pure-math throughput estimator with 4-tier provenance.

`estimate_tps` returns a `TpsEstimate` carrying a single decode-TPS
value and an explicit source tag. Each tier has a fixed
confidence value; the function refuses honestly
(`source="requires_measurement"`, value=None) when no tier
applies.

Decision tree (lower number wins when multiple match):
  1a  own_measured                 0.95  v2 only — dead in v1 (no caller passes bench_cells)
  2   provider_anchor (AA median)  0.7   batch=1 + AA row matches
  3   bandwidth_heuristic          0.6   batch=1 + has bandwidth
  4   requires_measurement         0.0   refusal

**Tier 1b removed (M10 deferral, 2026-05-31):** the public benchmark
anchor tier originally bridged the confidence gap between Tier 3
(heuristic 0.6) and Tier 1a (own_measured 0.95) using
hand-curated cells from public blogs. The Tier 1b cell-curation
work proved infeasible — public benchmark sources don't publish
the steady-state per-stream decode-TPS shape our BenchmarkCell
schema expects, source URLs rot rapidly (verified across PR #17's
Spheron URL 404 + replacement-article bandwidth-physics failure),
and even paid first-principles sources (Kiely 2026 Inference
Engineering) teach the heuristic methodology rather than
publishing measured TPS. v1 ships with Tiers 2/3 as the
confidence ceiling for non-own-measured queries. v2's M17
GuideLLM-measured cells will revive Tier 1a (and `bench_cells`
becomes a real input again then).

Tier 1a stays in this module as **v2-ready dead code**: the loop
exists, the matching helper exists, the Source Literal still
includes `own_measured` — only the v1 production callers stop
passing `bench_cells` data. v2's unlock is to start populating
`bench_cells` from a GuideLLM-fed parquet; no further surgery
here is required.

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
# _CONFIDENCE_PUBLIC_ANCHOR (0.80) removed with Tier 1b deferral.
# v2 reintroduces it when the public-anchor tier comes back.
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
    aa_observations: list[AaModelRow] | None,
    reasoning_effort: ReasoningEffort | None = None,
    bench_cells: list[BenchmarkCell] | None = None,
) -> TpsEstimate:
    """Walk the 4-tier provenance ladder. Returns the first tier
    that applies. Pure function — no I/O, no globals beyond named
    constants.

    `bench_cells`: v2-only input. Defaults to `[]` for v1 callers,
    so Tier 1a (own_measured) is dead code in v1. v2's M17
    GuideLLM work revives this by passing a populated list.

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
    if bench_cells is None:
        bench_cells = []
    if batch_size <= 0 or context_length <= 0:
        raise ValueError(
            f"batch_size and context_length must be positive; "
            f"got batch_size={batch_size}, context_length={context_length}"
        )

    # Tier 1a: own_measured exact match. v2-only — v1 callers
    # always pass bench_cells=[] (or omit), so this loop is a
    # no-op in v1. v2's M17 GuideLLM-measured cells will populate
    # bench_cells with own_measured rows and this branch goes
    # live. BenchmarkCell currently rejects own_measured at
    # construction; v2 flips that validator off as the unlock.
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

    # Tier 1b removed (M10 deferral, 2026-05-31). See module
    # docstring for the rationale. The Source Literal still
    # includes "public_benchmark_anchor" so a future revival is a
    # one-spot re-insertion here; the cell schema and validator
    # ecosystem (PR #17 verification tooling) stay intact for v2.

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
                        f"per Kiely 2026, Inference Engineering §2.4.2."
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
