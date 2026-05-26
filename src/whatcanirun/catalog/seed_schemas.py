"""Pydantic schemas for the GPU and quantization supplement YAMLs.

Per ADR-015, supplements are our own controlled data — schemas use
`extra="forbid"` so YAML typos fail loudly instead of silently dropping
unknown keys (the upstream-data clients use `extra="ignore"`; different
problem, different policy).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FormFactor = Literal["SXM", "PCIe", "NVL", "OAM"]


class GpuSupplement(BaseModel):
    """One row of `seeds/gpus_supplement.yaml`.

    Joins ComputePrices `/api/v1/gpus` by `slug`. Fields here are the ones
    ComputePrices does not expose (fp8/fp4 tflops, form factor, kernel
    support).
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    fp8_tflops_dense: float | None
    fp4_tflops_dense: float | None
    form_factor: FormFactor
    supports_fp8: bool
    supports_fp4: bool
    attention_kernels_supported: list[str]
    notes: str
    datasheet_url: str


class Quantization(BaseModel):
    """One row of `seeds/quantizations.yaml`.

    Sourced from Inference Engineering §5.1.1. `experimental=True` marks
    formats whose accept-criteria semantics are not yet pinned down by
    measured cells (M10); M07/M10 are expected to filter on this flag.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    bits_per_weight: int
    kv_cache_bits_default: int
    introduced_architecture: str
    notes: str
    experimental: bool = Field(default=False)
