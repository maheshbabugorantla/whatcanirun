"""Projection tests for ComputePrices row types.

Per ADR-015: every upstream response is stored verbatim in `raw`, while
Pydantic projects only the fields we currently consume (`extra="ignore"`).
Unknown fields that ship with future CP releases MUST round-trip through
`raw` without breaking validation.

These tests load real captured fixtures (no respx; pure dict projection)
so they double as snapshot guards against the captured fixture schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmCatalogRow,
    LlmPriceRow,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((_FIXTURES / name).read_text())["data"]


@pytest.fixture(scope="module")
def gpu_catalog_rows() -> list[dict[str, Any]]:
    return _load("cp_gpus_2026-05-26.json")


@pytest.fixture(scope="module")
def gpu_price_rows() -> list[dict[str, Any]]:
    return _load("cp_gpu_prices_2026-05-26.json")


@pytest.fixture(scope="module")
def llm_catalog_rows() -> list[dict[str, Any]]:
    return _load("cp_llm_models_2026-05-26.json")


@pytest.fixture(scope="module")
def llm_price_rows() -> list[dict[str, Any]]:
    return _load("cp_llm_prices_2026-05-26.json")


# ---------------------------------------------------------------- GpuCatalogRow


class TestGpuCatalogRow:
    def test_projects_real_row(self, gpu_catalog_rows: list[dict[str, Any]]) -> None:
        raw = gpu_catalog_rows[0]
        row = GpuCatalogRow.project(raw)
        assert row.slug == raw["slug"]
        assert row.name == raw["name"]
        assert row.vram_gb == raw["vram_gb"]
        assert row.manufacturer in {"NVIDIA", "AMD", "Intel"}

    def test_raw_carries_full_payload(self, gpu_catalog_rows: list[dict[str, Any]]) -> None:
        raw = gpu_catalog_rows[0]
        row = GpuCatalogRow.project(raw)
        # `raw` must include keys the projection drops (per ADR-015)
        assert row.raw == raw
        # And include keys not in the projection (e.g. `id`, `cloud_compatible`)
        assert "id" in row.raw
        assert "cloud_compatible" in row.raw

    def test_extra_field_is_ignored_not_rejected(
        self, gpu_catalog_rows: list[dict[str, Any]]
    ) -> None:
        # Simulate a future CP release adding a brand-new field.
        future_payload = {**gpu_catalog_rows[0], "future_field": "schema-evolution-test"}
        row = GpuCatalogRow.project(future_payload)
        assert row.raw["future_field"] == "schema-evolution-test"

    def test_all_fixture_rows_project(self, gpu_catalog_rows: list[dict[str, Any]]) -> None:
        rows = [GpuCatalogRow.project(r) for r in gpu_catalog_rows]
        assert len(rows) == 66

    def test_specs_dict_preserved_flexibly(self, gpu_catalog_rows: list[dict[str, Any]]) -> None:
        # specs is nested + evolving; typed loose per ADR-015
        h100 = next(r for r in gpu_catalog_rows if r["slug"] == "h100")
        row = GpuCatalogRow.project(h100)
        assert isinstance(row.specs, dict)
        # specs in the H100 row carries fp16_tflops, cuda_cores, etc.
        assert any(k.endswith("_tflops") for k in row.specs)


# ----------------------------------------------------------------- GpuPriceRow


class TestGpuPriceRow:
    def test_projects_real_row(self, gpu_price_rows: list[dict[str, Any]]) -> None:
        raw = gpu_price_rows[0]
        row = GpuPriceRow.project(raw)
        assert row.provider == raw["provider"]
        assert row.gpu_slug == raw["gpu_slug"]
        assert row.price_per_hour_usd == raw["price_per_hour_usd"]
        assert row.pricing_type in {"on_demand", "reserved", "spot"}

    def test_commitment_months_optional(self, gpu_price_rows: list[dict[str, Any]]) -> None:
        # Fixture has many rows with commitment_months=None; one must project cleanly.
        raw = next(r for r in gpu_price_rows if r["commitment_months"] is None)
        row = GpuPriceRow.project(raw)
        assert row.commitment_months is None

    def test_all_fixture_rows_project(self, gpu_price_rows: list[dict[str, Any]]) -> None:
        rows = [GpuPriceRow.project(r) for r in gpu_price_rows]
        assert len(rows) == 1000
        # Every pricing_type encountered must be in our Literal.
        observed = {r.pricing_type for r in rows}
        assert observed <= {"on_demand", "reserved", "spot"}, observed

    def test_unknown_pricing_type_rejected(self, gpu_price_rows: list[dict[str, Any]]) -> None:
        # If CP introduces a new pricing_type, we want to know — strict on enums.
        bad = {**gpu_price_rows[0], "pricing_type": "unicorn"}
        with pytest.raises(ValidationError):
            GpuPriceRow.project(bad)


# ---------------------------------------------------------------- LlmCatalogRow


class TestLlmCatalogRow:
    def test_projects_real_row(self, llm_catalog_rows: list[dict[str, Any]]) -> None:
        raw = llm_catalog_rows[0]
        row = LlmCatalogRow.project(raw)
        assert row.slug == raw["slug"]
        assert row.name == raw["name"]
        assert row.modalities == raw["modalities"]

    def test_optional_fields_tolerate_null(self, llm_catalog_rows: list[dict[str, Any]]) -> None:
        # Spec confirmed: 16% of rows have null family + context_window.
        raw = next(r for r in llm_catalog_rows if r["context_window"] is None)
        row = LlmCatalogRow.project(raw)
        assert row.context_window is None

    def test_all_fixture_rows_project(self, llm_catalog_rows: list[dict[str, Any]]) -> None:
        rows = [LlmCatalogRow.project(r) for r in llm_catalog_rows]
        assert len(rows) == 214


# ----------------------------------------------------------------- LlmPriceRow


class TestLlmPriceRow:
    def test_projects_real_row(self, llm_price_rows: list[dict[str, Any]]) -> None:
        raw = llm_price_rows[0]
        row = LlmPriceRow.project(raw)
        assert row.provider == raw["provider"]
        assert row.model_slug == raw["model_slug"]
        assert row.price_per_1m_input_usd == raw["price_per_1m_input_usd"]
        assert row.price_per_1m_output_usd == raw["price_per_1m_output_usd"]
        assert row.pricing_type in {"standard", "batch"}

    def test_cached_input_price_optional(self, llm_price_rows: list[dict[str, Any]]) -> None:
        raw = llm_price_rows[0]
        row = LlmPriceRow.project(raw)
        # New field for providers that support prompt caching; None elsewhere.
        assert row.price_per_1m_cached_input_usd == raw.get("price_per_1m_cached_input_usd")

    def test_all_fixture_rows_project(self, llm_price_rows: list[dict[str, Any]]) -> None:
        rows = [LlmPriceRow.project(r) for r in llm_price_rows]
        assert len(rows) == 498
