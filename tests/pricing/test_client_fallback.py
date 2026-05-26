"""Upstream-down fallback tests for ComputePricesClient (Slice E).

ADR-013: when upstream is unreachable after retries are exhausted, the
client serves the last-good `<endpoint>.latest.json` so tool calls
never fail outright. The trust envelope (M08) is the channel for
communicating staleness.

Hierarchy of behavior tested here:
  1. Transient 500 followed by success -> retry transparently recovers
  2. All retries fail + cache present -> serve cache, no exception
  3. All retries fail + cache empty   -> raise ComputePricesUnavailable

Retry waits are zeroed in tests so the suite doesn't burn ~7 seconds
per fallback test on real exponential backoff sleeps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing.computeprices import (
    ComputePricesClient,
    ComputePricesUnavailable,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_BASE = "https://www.computeprices.com/api/v1"


def _payload(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.fixture
def fast_client(cache_dir: Path) -> ComputePricesClient:
    """Client with retries enabled but zero wait — exercises the retry
    code path without the 7s exponential-backoff slog."""
    return ComputePricesClient(
        cache_dir=cache_dir,
        retry_attempts=4,
        retry_wait_min_s=0.0,
        retry_wait_max_s=0.0,
    )


# -------------------------------------------------------------- transient 500


@pytest.mark.asyncio
@respx.mock
async def test_transient_500_then_success_recovers(
    fast_client: ComputePricesClient,
) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(500, text="still boom"),
            httpx.Response(200, json=payload),
        ]
    )

    rows = await fast_client.get_gpu_catalog()

    assert route.call_count == 3
    assert len(rows) == 66


@pytest.mark.asyncio
@respx.mock
async def test_connect_error_then_success_recovers(
    fast_client: ComputePricesClient,
) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json=payload),
        ]
    )

    rows = await fast_client.get_gpu_catalog()

    assert route.call_count == 2
    assert len(rows) == 66


# ---------------------------------------------------------- fallback to cache


@pytest.mark.asyncio
@respx.mock
async def test_persistent_5xx_falls_back_to_last_good_cache(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """All retries fail BUT we have a stale latest.json -> serve cache."""
    payload = _payload("cp_gpus_2026-05-26.json")

    # First: a successful fetch that populates the cache.
    success_route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))
    await fast_client.get_gpu_catalog()
    assert success_route.called

    # Backdate the cache so TTL forces a refetch attempt next.
    import os
    import time as time_mod

    cache_file = cache_dir / "gpus.latest.json"
    os.utime(cache_file, (time_mod.time() - 99 * 3600, time_mod.time() - 99 * 3600))

    # Now: upstream is down for the entire retry window.
    respx.reset()
    fail_route = respx.get(f"{_BASE}/gpus").mock(
        return_value=httpx.Response(503, text="service unavailable")
    )

    rows = await fast_client.get_gpu_catalog()
    # Fell back to the cached payload (66 rows) — no exception.
    assert len(rows) == 66
    # Hit upstream the full retry budget before giving up.
    assert fail_route.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_persistent_5xx_with_empty_cache_raises(
    fast_client: ComputePricesClient,
) -> None:
    fail_route = respx.get(f"{_BASE}/gpus").mock(
        return_value=httpx.Response(503, text="service unavailable")
    )

    with pytest.raises(ComputePricesUnavailable) as exc_info:
        await fast_client.get_gpu_catalog()

    assert "gpus" in str(exc_info.value)
    assert fail_route.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_connect_error_with_empty_cache_raises(
    fast_client: ComputePricesClient,
) -> None:
    respx.get(f"{_BASE}/gpus").mock(side_effect=httpx.ConnectError("DNS fail"))

    with pytest.raises(ComputePricesUnavailable):
        await fast_client.get_gpu_catalog()


# ------------------------------------------------------------- retry counts


@pytest.mark.asyncio
@respx.mock
async def test_2xx_does_not_retry(fast_client: ComputePricesClient) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    await fast_client.get_gpu_catalog()
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_4xx_client_error_is_not_retried(
    fast_client: ComputePricesClient,
) -> None:
    """A 4xx is a client bug (bad path, bad auth) — retrying is just
    burning quota. Should raise immediately without retries and without
    falling back to cache (cache could be valid; the request itself
    being wrong is the user's problem to fix)."""
    route = respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(401, text="unauthorized"))

    with pytest.raises(httpx.HTTPStatusError):
        await fast_client.get_gpu_catalog()

    assert route.call_count == 1
