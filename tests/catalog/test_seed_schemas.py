"""Schema tests for catalog supplements.

Per ADR-015 the supplement YAMLs are OUR controlled data — schemas use
`extra="forbid"` so typos surface as test failures rather than silent drops.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whatcanirun.catalog.seed_schemas import GpuSupplement, Quantization

_VALID_GPU_ROW = {
    "slug": "h100",
    "fp8_tflops_dense": 1979.0,
    "fp4_tflops_dense": None,
    "form_factor": "SXM",
    "supports_fp8": True,
    "supports_fp4": False,
    "attention_kernels_supported": ["flash_attention_2", "paged_attention"],
    "notes": "Hopper SXM5 80GB.",
    "datasheet_url": "https://www.nvidia.com/en-us/data-center/h100/",
}

_VALID_QUANT_ROW = {
    "slug": "fp16",
    "bits_per_weight": 16,
    "kv_cache_bits_default": 16,
    "introduced_architecture": "Pascal",
    "notes": "IEEE 754 half-precision.",
}


class TestGpuSupplement:
    def test_accepts_valid_row(self) -> None:
        gpu = GpuSupplement(**_VALID_GPU_ROW)
        assert gpu.slug == "h100"
        assert gpu.fp8_tflops_dense == 1979.0
        assert gpu.form_factor == "SXM"

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            GpuSupplement(**_VALID_GPU_ROW, mystery_field=42)
        assert "mystery_field" in str(exc_info.value)

    def test_rejects_invalid_form_factor(self) -> None:
        with pytest.raises(ValidationError):
            GpuSupplement(**{**_VALID_GPU_ROW, "form_factor": "BananaCase"})

    def test_rejects_missing_required_field(self) -> None:
        row = dict(_VALID_GPU_ROW)
        del row["fp8_tflops_dense"]
        with pytest.raises(ValidationError) as exc_info:
            GpuSupplement(**row)
        assert "fp8_tflops_dense" in str(exc_info.value)


class TestQuantization:
    def test_accepts_valid_row(self) -> None:
        q = Quantization(**_VALID_QUANT_ROW)
        assert q.slug == "fp16"
        assert q.bits_per_weight == 16
        assert q.experimental is False  # default for stable formats

    def test_experimental_defaults_false(self) -> None:
        q = Quantization(**_VALID_QUANT_ROW)
        assert q.experimental is False

    def test_experimental_can_be_set(self) -> None:
        q = Quantization(**{**_VALID_QUANT_ROW, "slug": "nvfp4", "experimental": True})
        assert q.experimental is True

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Quantization(**_VALID_QUANT_ROW, mystery_field="oops")
        assert "mystery_field" in str(exc_info.value)
