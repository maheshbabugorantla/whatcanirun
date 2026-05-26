"""HTTP client tests for ComputePrices (Slice B).

Covers the happy path for all four endpoints using respx to mock
upstream. Cache (Slice C), snapshots (D), fallback (E), pruning (F),
and schema-evolution (G) ship in subsequent slices.

Per ADR-013 + ADR-015 the client must:
  - read from the real-fixture payload (captured via
    `scripts/capture_cp_gpus_fixture.py`) so tests stay aligned with
    upstream's actual schema
  - return typed projections (`GpuCatalogRow` etc.) that already carry
    the verbatim payload in `raw`
  - never touch the live network in tests (CI runs with
    COMPUTEPRICES_API_KEY="" and the respx fixture catches accidental
    bypass attempts as ConnectError)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing.computeprices import ComputePricesClient
from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmCatalogRow,
    LlmPriceRow,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_BASE = "https://www.computeprices.com/api/v1"


def _payload(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


# ----------------------------------------------------------------- happy paths


@pytest.mark.asyncio
@respx.mock
async def test_get_gpu_catalog_returns_projected_rows(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_gpu_catalog()

    assert len(rows) == 66
    assert all(isinstance(r, GpuCatalogRow) for r in rows)
    assert {r.slug for r in rows[:3]} <= {r["slug"] for r in payload["data"][:3]}


@pytest.mark.asyncio
@respx.mock
async def test_get_gpu_prices_returns_projected_rows(cache_dir: Path) -> None:
    payload = _payload("cp_gpu_prices_2026-05-26.json")
    respx.get(f"{_BASE}/gpu-prices").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_gpu_prices()

    assert len(rows) == 1000
    assert all(isinstance(r, GpuPriceRow) for r in rows)


@pytest.mark.asyncio
@respx.mock
async def test_get_llm_catalog_returns_projected_rows(cache_dir: Path) -> None:
    payload = _payload("cp_llm_models_2026-05-26.json")
    respx.get(f"{_BASE}/llm-models").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_llm_catalog()

    assert len(rows) == 214
    assert all(isinstance(r, LlmCatalogRow) for r in rows)


@pytest.mark.asyncio
@respx.mock
async def test_get_llm_prices_returns_projected_rows(cache_dir: Path) -> None:
    payload = _payload("cp_llm_prices_2026-05-26.json")
    respx.get(f"{_BASE}/llm-prices").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_llm_prices()

    assert len(rows) == 498
    assert all(isinstance(r, LlmPriceRow) for r in rows)


# --------------------------------------------------------------- auth + tier


@pytest.mark.asyncio
@respx.mock
async def test_sends_bearer_token_when_api_key_present(cache_dir: Path) -> None:
    """When an API key is configured, every request carries it as a
    bearer token so CP attributes the call to the 5k/hr quota tier."""
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir, api_key="cp_live_test123")
    await client.get_gpu_catalog()

    assert route.called
    sent_auth = route.calls.last.request.headers.get("authorization")
    assert sent_auth == "Bearer cp_live_test123"


@pytest.mark.asyncio
@respx.mock
async def test_omits_authorization_when_anonymous(cache_dir: Path) -> None:
    """No API key → no Authorization header → CP serves from the
    anonymous 60/hr tier."""
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir, api_key=None)
    await client.get_gpu_catalog()

    assert route.called
    assert "authorization" not in {h.lower() for h in route.calls.last.request.headers}


# ---------------------------------------------------------------- raw passthrough


@pytest.mark.asyncio
@respx.mock
async def test_row_raw_carries_unknown_future_field(cache_dir: Path) -> None:
    """Future CP releases adding fields must not break the client —
    unknown keys survive in `row.raw` per ADR-015."""
    base = _payload("cp_gpus_2026-05-26.json")
    # Mutate just the first row to add an undeclared future field.
    base["data"][0] = {**base["data"][0], "future_metric": 12345}
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=base))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows = await client.get_gpu_catalog()

    assert rows[0].raw["future_metric"] == 12345
