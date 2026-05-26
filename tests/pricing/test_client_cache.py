"""On-disk cache + TTL tests for ComputePricesClient (Slice C).

Cache contract (from spec/M02-computeprices-client.md):
  ~/.cache/whatcanirun/computeprices/<endpoint>.latest.json
  TTL: 1h for gpu-prices / llm-prices, 24h for gpus / llm-models
  Pitfall #4: TTL needs jitter (±60s) to avoid thundering herd

Tests use a tmp_path cache and monkeypatch `_now` + file mtime to drive
TTL behavior deterministically.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing import computeprices as cp_mod
from whatcanirun.pricing.computeprices import ComputePricesClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_BASE = "https://www.computeprices.com/api/v1"


def _payload(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


# -------------------------------------------------------------- cache-on-miss


@pytest.mark.asyncio
@respx.mock
async def test_first_call_writes_latest_json(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()

    cache_file = cache_dir / "gpus.latest.json"
    assert cache_file.exists()
    on_disk = json.loads(cache_file.read_text())
    # Verbatim payload, including the top-level `meta` block (ADR-015).
    assert on_disk == payload


@pytest.mark.asyncio
@respx.mock
async def test_second_call_within_ttl_skips_upstream(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    rows1 = await client.get_gpu_catalog()
    rows2 = await client.get_gpu_catalog()

    assert route.call_count == 1, "second call within TTL must serve from disk"
    assert [r.slug for r in rows1] == [r.slug for r in rows2]


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit_is_fast(cache_dir: Path) -> None:
    """Spec acceptance criterion: cache hit returns <5ms."""
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()  # populate cache

    start = time.perf_counter()
    await client.get_gpu_catalog()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 5ms is the spec target. Give some slack for CI: 50ms.
    assert elapsed_ms < 50, f"cache hit took {elapsed_ms:.1f}ms (target <5ms)"


# ---------------------------------------------------------------- TTL expiry


@pytest.mark.asyncio
@respx.mock
async def test_cache_expires_after_prices_ttl_1h(cache_dir: Path) -> None:
    """Prices TTL is 1h. After 2h the next call must hit upstream again."""
    payload = _payload("cp_gpu_prices_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpu-prices").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_prices()
    assert route.call_count == 1

    # Backdate the cache file to 2h ago.
    cache_file = cache_dir / "gpu-prices.latest.json"
    two_hours_ago = time.time() - 2 * 3600
    os.utime(cache_file, (two_hours_ago, two_hours_ago))

    await client.get_gpu_prices()
    assert route.call_count == 2, "expired cache must refetch from upstream"


@pytest.mark.asyncio
@respx.mock
async def test_cache_holds_for_catalogs_24h(cache_dir: Path) -> None:
    """Catalog TTL is 24h. A cache file 6h old must NOT trigger a refetch."""
    payload = _payload("cp_llm_models_2026-05-26.json")
    route = respx.get(f"{_BASE}/llm-models").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_llm_catalog()

    cache_file = cache_dir / "llm-models.latest.json"
    six_hours_ago = time.time() - 6 * 3600
    os.utime(cache_file, (six_hours_ago, six_hours_ago))

    await client.get_llm_catalog()
    assert route.call_count == 1, "catalog cache <24h old must not refetch"


# -------------------------------------------------------- separate endpoints


@pytest.mark.asyncio
@respx.mock
async def test_each_endpoint_caches_independently(cache_dir: Path) -> None:
    """The 4 endpoints share a cache dir but distinct files."""
    g = _payload("cp_gpus_2026-05-26.json")
    p = _payload("cp_gpu_prices_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=g))
    respx.get(f"{_BASE}/gpu-prices").mock(return_value=httpx.Response(200, json=p))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()
    await client.get_gpu_prices()

    assert (cache_dir / "gpus.latest.json").exists()
    assert (cache_dir / "gpu-prices.latest.json").exists()


# ----------------------------------------------------------- monkeypatchable now


@pytest.mark.asyncio
@respx.mock
async def test_now_function_is_monkeypatchable_for_tests(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client must expose its time-of-now as a swappable callable so
    TTL behavior is testable without sleeping."""
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()
    assert route.call_count == 1

    # Pretend we're 25h in the future. Catalog TTL is 24h -> must refetch.
    future = dt.datetime.now(dt.UTC) + dt.timedelta(hours=25)
    monkeypatch.setattr(cp_mod, "_now", lambda: future)

    await client.get_gpu_catalog()
    assert route.call_count == 2
