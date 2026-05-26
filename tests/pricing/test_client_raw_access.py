"""Public raw-access tests for ComputePricesClient.

`get_raw_response(endpoint)` returns the full unparsed CP payload —
including the top-level `meta` block that the typed projections drop.
Two consumer use cases:
  - trust envelope provenance reading `meta.generated_at` for
    freshness reporting
  - sampling new upstream fields before adding them to the typed
    projection

This method must follow the same cache + retry + fallback rules as
the typed-projection methods so it doesn't drift into a separate
code path.
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


@pytest.mark.asyncio
@respx.mock
async def test_returns_full_payload_including_meta(cache_dir: Path) -> None:
    """The full upstream payload — including the top-level `meta` block
    that the typed methods drop — must round-trip through
    `get_raw_response`."""
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    raw = await client.get_raw_response("gpus")

    assert raw == payload
    assert "meta" in raw
    assert raw["meta"]["generated_at"] == payload["meta"]["generated_at"]


@pytest.mark.asyncio
@respx.mock
async def test_uses_cache_like_typed_methods(cache_dir: Path) -> None:
    """Second call within TTL must serve from cache without hitting upstream."""
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_raw_response("gpus")
    await client.get_raw_response("gpus")

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_share_cache_with_typed_method(cache_dir: Path) -> None:
    """`get_raw_response('gpus')` and `get_gpu_catalog()` must share
    one cache file — they're both reading the same endpoint."""
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()
    raw = await client.get_raw_response("gpus")

    assert route.call_count == 1, "second call should hit shared cache"
    assert raw == payload


@pytest.mark.asyncio
@respx.mock
async def test_rejects_unknown_endpoint(cache_dir: Path) -> None:
    """Endpoint must be one of the 4 known CP paths. An unknown one
    is a caller bug — refuse loudly rather than make an arbitrary
    upstream call."""
    client = ComputePricesClient(cache_dir=cache_dir)
    with pytest.raises(ValueError, match="unknown CP endpoint"):
        await client.get_raw_response("not-a-real-endpoint")


@pytest.mark.asyncio
@respx.mock
async def test_each_known_endpoint_is_accepted(cache_dir: Path) -> None:
    for name, fixture in [
        ("gpus", "cp_gpus_2026-05-26.json"),
        ("gpu-prices", "cp_gpu_prices_2026-05-26.json"),
        ("llm-models", "cp_llm_models_2026-05-26.json"),
        ("llm-prices", "cp_llm_prices_2026-05-26.json"),
    ]:
        payload = _payload(fixture)
        respx.get(f"{_BASE}/{name}").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    for name in ("gpus", "gpu-prices", "llm-models", "llm-prices"):
        raw = await client.get_raw_response(name)
        assert "data" in raw
