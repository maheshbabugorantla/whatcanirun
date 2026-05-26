"""End-to-end schema-evolution tests for ComputePricesClient.

ADR-015: upstream-data clients must tolerate new fields without
breaking validation. The CI workflow's dedicated `schema-evolution`
job collects tests carrying `@pytest.mark.schema_evolution` and
fails the build if none are collected.

These tests inject synthetic future fields at two levels and assert
they survive the full HTTP -> projection -> cache path:
  - new top-level field on a CP row
  - new nested key inside the evolving `specs` blob

Both paths must surface the new value via `row.raw` so a later code
change can project it without re-deploying.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing.computeprices import ComputePricesClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_BASE = "https://www.computeprices.com/api/v1"


def _payload(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_top_level_field_survives_end_to_end(cache_dir: Path) -> None:
    """A future CP release adds a new top-level field on the GPU
    catalog. The client returns rows unchanged and the new field is
    queryable through `row.raw`."""
    payload = _payload("cp_gpus_2026-05-26.json")
    payload["data"][0] = {
        **payload["data"][0],
        "thermal_class": "B+",  # field we don't currently model
    }
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_gpu_catalog()

    assert rows[0].raw["thermal_class"] == "B+"


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_nested_specs_key_survives_end_to_end(cache_dir: Path) -> None:
    """The evolving `specs` blob gains a new nested key (CP has done
    this multiple times — added `cuda_cores`, `memory_bandwidth_gb_s`,
    `fp16_tflops`, etc. without notice). The new key MUST survive in
    both `row.specs` (typed dict, still loose per ADR-015) AND `row.raw`.
    """
    payload = _payload("cp_gpus_2026-05-26.json")
    h100 = next(r for r in payload["data"] if r["slug"] == "h100")
    h100["specs"] = {**h100["specs"], "fp4_tflops_dense": 9999.0}
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_gpu_catalog()

    h100_row = next(r for r in rows if r.slug == "h100")
    assert h100_row.specs["fp4_tflops_dense"] == 9999.0
    assert h100_row.raw["specs"]["fp4_tflops_dense"] == 9999.0


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_field_survives_cache_round_trip(cache_dir: Path) -> None:
    """A future-field response must round-trip through the on-disk
    cache without losing the new field on the second (cached) call."""
    payload = _payload("cp_gpu_prices_2026-05-26.json")
    payload["data"][0] = {
        **payload["data"][0],
        "discount_window_hours": 4,  # field we don't currently model
    }
    respx.get(f"{_BASE}/gpu-prices").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows_first = await client.get_gpu_prices()
    rows_cached = await client.get_gpu_prices()

    assert rows_first[0].raw["discount_window_hours"] == 4
    assert rows_cached[0].raw["discount_window_hours"] == 4
