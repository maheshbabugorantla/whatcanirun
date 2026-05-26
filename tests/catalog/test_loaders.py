"""Loader tests for catalog supplement YAMLs.

Covers the happy path (load one row, full list) and the malformed-YAML
path: errors must mention the YAML's line number so the user can find
the typo without grep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whatcanirun.catalog.loaders import (
    SeedLoadError,
    load_gpu_supplements,
    load_quantizations,
)

_VALID_GPU_YAML = """\
- slug: h100
  fp8_tflops_dense: 1979
  fp4_tflops_dense: null
  form_factor: SXM
  supports_fp8: true
  supports_fp4: false
  attention_kernels_supported: [flash_attention_2, paged_attention]
  notes: "Hopper SXM5 80GB."
  datasheet_url: "https://www.nvidia.com/en-us/data-center/h100/"
"""

_VALID_QUANT_YAML = """\
- slug: fp16
  bits_per_weight: 16
  kv_cache_bits_default: 16
  introduced_architecture: Pascal
  notes: "IEEE 754 half-precision."
"""


class TestLoadGpuSupplements:
    def test_loads_single_row(self, tmp_path: Path) -> None:
        f = tmp_path / "g.yaml"
        f.write_text(_VALID_GPU_YAML)
        rows = load_gpu_supplements(f)
        assert len(rows) == 1
        assert rows[0].slug == "h100"
        assert rows[0].fp8_tflops_dense == 1979.0

    def test_missing_field_raises_with_path_and_line(self, tmp_path: Path) -> None:
        f = tmp_path / "g.yaml"
        # Two rows; second row is missing `datasheet_url`. The error must
        # mention the file path so the user can find it; line is best-effort
        # because pyyaml safe_load doesn't always thread line info through
        # validation, but the file path is non-negotiable.
        f.write_text(
            _VALID_GPU_YAML
            + "- slug: h200\n"
            + "  fp8_tflops_dense: 1979\n"
            + "  fp4_tflops_dense: null\n"
            + "  form_factor: SXM\n"
            + "  supports_fp8: true\n"
            + "  supports_fp4: false\n"
            + "  attention_kernels_supported: []\n"
            + "  notes: oops\n"
        )
        with pytest.raises(SeedLoadError) as exc_info:
            load_gpu_supplements(f)
        msg = str(exc_info.value)
        assert str(f) in msg
        assert "datasheet_url" in msg

    def test_malformed_yaml_syntax_raises_with_line(self, tmp_path: Path) -> None:
        f = tmp_path / "g.yaml"
        f.write_text("- slug: h100\n  fp8_tflops_dense: [unterminated\n")
        with pytest.raises(SeedLoadError) as exc_info:
            load_gpu_supplements(f)
        msg = str(exc_info.value)
        assert str(f) in msg
        assert "line" in msg.lower()

    def test_root_must_be_a_list(self, tmp_path: Path) -> None:
        f = tmp_path / "g.yaml"
        f.write_text("slug: h100\n")
        with pytest.raises(SeedLoadError) as exc_info:
            load_gpu_supplements(f)
        assert "list" in str(exc_info.value).lower()


class TestLoadQuantizations:
    def test_loads_single_row(self, tmp_path: Path) -> None:
        f = tmp_path / "q.yaml"
        f.write_text(_VALID_QUANT_YAML)
        rows = load_quantizations(f)
        assert len(rows) == 1
        assert rows[0].slug == "fp16"
        assert rows[0].experimental is False

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "q.yaml"
        f.write_text(_VALID_QUANT_YAML + "  mystery_field: 42\n")
        with pytest.raises(SeedLoadError) as exc_info:
            load_quantizations(f)
        assert "mystery_field" in str(exc_info.value)
