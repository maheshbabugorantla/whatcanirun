"""`BenchmarkCell` — one row of `seeds/benchmark_cells.parquet`.

Defines the shape M07's `estimate_tps` consumes for Tier 1a / 1b
provenance lookups. M10 owns populating the parquet with 20-30
hand-curated rows from public sources; M07 needs the row TYPE to
write the lookup logic. M07 ships a small bootstrap parquet
(5-10 anchors) so its acceptance criterion 5 ("seed validated to
contain only public_benchmark_anchor rows") can be exercised
end-to-end before M10 expands it.

CRITICAL v1 invariant per spec/M10: `source="own_measured"` is
REJECTED by validator, not just convention. v2's M17 introduces
own_measured cells via GuideLLM runs; until that ship date every
committed row MUST be `source="public_benchmark_anchor"`. The
validator below makes this enforcement explicit so a future
contributor adding an own_measured row can't slip past code
review on the convention alone.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Engine = Literal["vllm", "sglang", "tensorrt_llm", "tgi", "other"]
BenchmarkSource = Literal["public_benchmark_anchor", "own_measured"]


class BenchmarkCell(BaseModel):
    """One measured anchor row. Maps `(gpu, model, quant, tp,
    batch, ctx)` to a decoder TPS (and optional prefill / TTFT)."""

    model_config = ConfigDict(extra="forbid")

    # Op-point identifiers — join keys
    gpu_slug: str
    model_slug: str
    quant_slug: str
    tp_size: int
    batch_size: int
    context_length: int

    # Measured numbers
    decode_tps: float
    prefill_tps: float | None = None
    ttft_ms: float | None = None

    # Engine details
    engine: Engine
    engine_version: str
    measured_at: date

    # Provenance — v1 is ALWAYS `public_benchmark_anchor`. The
    # Literal alone allows both values for forward compatibility
    # with v2's M17, but the validator below rejects own_measured
    # at construction time. To re-enable own_measured (v2), the
    # validator gets feature-flagged off, not removed — the
    # acceptance test in M07 then needs updating to allow it.
    source: BenchmarkSource
    source_url: str = Field(min_length=1)
    notes: str

    @field_validator("source")
    @classmethod
    def reject_own_measured_in_v1(cls, v: BenchmarkSource) -> BenchmarkSource:
        """v1 NEVER accepts `own_measured` per spec/M10. M17 (v2)
        flips this guard off once GuideLLM-measured cells are
        provenance-ready. Until then, fail loudly so a
        well-intentioned contributor doesn't slip an unverified
        own_measured row past code review."""
        if v == "own_measured":
            raise ValueError(
                "v1 rejects source='own_measured'; only "
                "'public_benchmark_anchor' rows are accepted until M17 "
                "(v2) introduces GuideLLM-measured cells with verified "
                "provenance. See spec/M10 § Pydantic schema."
            )
        return v
