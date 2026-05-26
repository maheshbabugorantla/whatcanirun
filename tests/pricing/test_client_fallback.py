"""Upstream-down fallback tests for ComputePricesClient.

ADR-013: when upstream is unreachable after retries are exhausted,
the client serves the last-good `<endpoint>.latest.json` (or, if
that's missing or corrupt, walks `<endpoint>.snapshots/` newest-first
for a valid snapshot) so tool calls never fail outright. The trust
envelope is the channel for communicating staleness to users.

Hierarchy of behavior tested here:
  1. Transient 500 followed by success -> retry transparently recovers
  2. All retries fail + cache present  -> serve cache, no exception
  3. All retries fail + cache missing or corrupt + snapshot present
                                       -> serve snapshot
  4. All retries fail + nothing usable -> raise ComputePricesUnavailable

Retry waits are zeroed in tests so the suite doesn't burn ~7 seconds
per fallback test on real exponential backoff sleeps.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
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


def _seed_cache_file(cache_dir: Path, endpoint: str, contents: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{endpoint}.latest.json").write_text(contents)


def _seed_snapshot(
    cache_dir: Path,
    endpoint: str,
    ts: dt.datetime,
    payload: dict[str, Any],
) -> Path:
    """Write a valid gzipped snapshot with the given mtime."""
    snapshots = cache_dir / f"{endpoint}.snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    name = ts.strftime("%Y-%m-%dT%H-%M-%SZ") + ".json.gz"
    path = snapshots / name
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)
    os.utime(path, (ts.timestamp(), ts.timestamp()))
    return path


# --------------------------------------------------- snapshot walk-fallback


@pytest.mark.asyncio
@respx.mock
async def test_outage_falls_back_to_snapshot_when_latest_missing(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """ADR-013 promises a 30-day rolling snapshot history. When
    upstream is down and `latest.json` doesn't exist (e.g. the cache
    dir was rebuilt but snapshots were preserved), the client must
    walk the snapshot directory and serve the most-recent one rather
    than raising ComputePricesUnavailable."""
    payload = _payload("cp_gpus_2026-05-26.json")
    ts = dt.datetime(2026, 5, 24, 0, 0, 0, tzinfo=dt.UTC)
    _seed_snapshot(cache_dir, "gpus", ts, payload)

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

    rows = await fast_client.get_gpu_catalog()
    assert len(rows) == 66


@pytest.mark.asyncio
@respx.mock
async def test_outage_falls_back_to_snapshot_when_latest_corrupt(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """If latest.json is unreadable AND a valid snapshot exists, the
    snapshot is what we serve. The snapshot is the more durable
    record."""
    payload = _payload("cp_gpus_2026-05-26.json")
    _seed_cache_file(cache_dir, "gpus", "{not valid json")
    _seed_snapshot(cache_dir, "gpus", dt.datetime(2026, 5, 24, 0, 0, 0, tzinfo=dt.UTC), payload)

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

    rows = await fast_client.get_gpu_catalog()
    assert len(rows) == 66


@pytest.mark.asyncio
@respx.mock
async def test_outage_picks_most_recent_snapshot(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """Multiple snapshots present — newest valid one wins."""
    fresh_payload = _payload("cp_gpus_2026-05-26.json")
    # Older snapshot carries a different fingerprint so we can tell
    # which one the client picked.
    older_payload = {
        "data": [{**fresh_payload["data"][0], "slug": "OLD-MARKER"}],
        "meta": fresh_payload.get("meta", {}),
    }

    _seed_snapshot(
        cache_dir,
        "gpus",
        dt.datetime(2026, 5, 20, 0, 0, 0, tzinfo=dt.UTC),
        older_payload,
    )
    _seed_snapshot(
        cache_dir,
        "gpus",
        dt.datetime(2026, 5, 24, 0, 0, 0, tzinfo=dt.UTC),
        fresh_payload,
    )

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

    rows = await fast_client.get_gpu_catalog()
    assert "OLD-MARKER" not in {r.slug for r in rows}
    assert len(rows) == 66


@pytest.mark.asyncio
@respx.mock
async def test_outage_skips_corrupt_snapshot_to_next_oldest(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """A corrupt newest snapshot must not block fallback — the walk
    keeps going to the next-oldest valid snapshot."""
    payload = _payload("cp_gpus_2026-05-26.json")
    # Newer snapshot is corrupt (gzipped garbage); next one is valid.
    snapshots = cache_dir / "gpus.snapshots"
    snapshots.mkdir(parents=True)
    bad_ts = dt.datetime(2026, 5, 25, 0, 0, 0, tzinfo=dt.UTC)
    bad_path = snapshots / f"{bad_ts.strftime('%Y-%m-%dT%H-%M-%SZ')}.json.gz"
    bad_path.write_bytes(b"not actually gzipped json")
    os.utime(bad_path, (bad_ts.timestamp(), bad_ts.timestamp()))

    _seed_snapshot(cache_dir, "gpus", dt.datetime(2026, 5, 20, 0, 0, 0, tzinfo=dt.UTC), payload)

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

    rows = await fast_client.get_gpu_catalog()
    assert len(rows) == 66


@pytest.mark.asyncio
@respx.mock
async def test_corrupt_cache_during_outage_raises_unavailable(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    """If upstream is down AND latest.json is corrupt AND no snapshots
    exist, the client must raise ComputePricesUnavailable — not surface
    a raw JSONDecodeError or shape ValueError to callers."""
    _seed_cache_file(cache_dir, "gpus", "{not valid json")

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

    with pytest.raises(ComputePricesUnavailable):
        await fast_client.get_gpu_catalog()


@pytest.mark.asyncio
@respx.mock
async def test_outage_with_only_corrupt_snapshots_raises_unavailable(
    cache_dir: Path, fast_client: ComputePricesClient
) -> None:
    snapshots = cache_dir / "gpus.snapshots"
    snapshots.mkdir(parents=True)
    for ts in (
        dt.datetime(2026, 5, 25, 0, 0, 0, tzinfo=dt.UTC),
        dt.datetime(2026, 5, 24, 0, 0, 0, tzinfo=dt.UTC),
    ):
        p = snapshots / f"{ts.strftime('%Y-%m-%dT%H-%M-%SZ')}.json.gz"
        p.write_bytes(b"corrupt")
        os.utime(p, (ts.timestamp(), ts.timestamp()))

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(503, text="down"))

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
