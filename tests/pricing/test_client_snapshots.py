"""Snapshot persistence tests for ComputePricesClient (Slice D).

After each successful fetch, write a timestamped, gzipped copy of the
payload to `<cache_dir>/<endpoint>.snapshots/<UTC-ISO>.json.gz` so the
30-day rolling history (per ADR-013) can serve as fallback when
upstream is unreachable and as audit trail for trust envelope
freshness claims.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
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


@pytest.mark.asyncio
@respx.mock
async def test_first_fetch_writes_snapshot(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()

    snapshots_dir = cache_dir / "gpus.snapshots"
    assert snapshots_dir.is_dir()
    files = list(snapshots_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffixes == [".json", ".gz"]


@pytest.mark.asyncio
@respx.mock
async def test_snapshot_filename_is_utc_iso_safe(cache_dir: Path) -> None:
    """Snapshot filename is an ISO-8601 UTC timestamp with colons
    replaced by dashes so the filename works on every OS we support."""
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    fixed_now = dt.datetime(2026, 5, 26, 12, 34, 56, tzinfo=dt.UTC)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp_mod, "_now", lambda: fixed_now)
        client = ComputePricesClient(cache_dir=cache_dir)
        await client.get_gpu_catalog()

    snapshot = next((cache_dir / "gpus.snapshots").iterdir())
    assert snapshot.name == "2026-05-26T12-34-56Z.json.gz"


@pytest.mark.asyncio
@respx.mock
async def test_snapshot_contents_match_upstream_payload(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()

    snapshot = next((cache_dir / "gpus.snapshots").iterdir())
    with gzip.open(snapshot, "rt") as f:
        on_disk = json.load(f)
    assert on_disk == payload


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit_does_not_write_snapshot(cache_dir: Path) -> None:
    """Snapshots only fire on live fetches — cache hits MUST NOT
    create a duplicate snapshot of the same payload."""
    payload = _payload("cp_gpus_2026-05-26.json")
    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    client = ComputePricesClient(cache_dir=cache_dir)
    await client.get_gpu_catalog()  # live fetch + snapshot #1
    await client.get_gpu_catalog()  # cache hit  -> no new snapshot

    snapshots = list((cache_dir / "gpus.snapshots").iterdir())
    assert len(snapshots) == 1


@pytest.mark.asyncio
@respx.mock
async def test_subsequent_fetches_accumulate_snapshots(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each independent live fetch (cache miss) adds a fresh snapshot."""
    payload = _payload("cp_gpu_prices_2026-05-26.json")
    respx.get(f"{_BASE}/gpu-prices").mock(return_value=httpx.Response(200, json=payload))

    # Three fetches at three distinct simulated times, each past TTL.
    times = [
        dt.datetime(2026, 5, 26, 0, 0, 0, tzinfo=dt.UTC),
        dt.datetime(2026, 5, 26, 2, 0, 0, tzinfo=dt.UTC),
        dt.datetime(2026, 5, 26, 4, 0, 0, tzinfo=dt.UTC),
    ]
    client = ComputePricesClient(cache_dir=cache_dir)
    for t in times:
        # Each fetch needs both: simulated `now` AND backdated cache mtime
        # so TTL recognises the cache as stale on the next iteration.
        monkeypatch.setattr(cp_mod, "_now", lambda t=t: t)
        await client.get_gpu_prices()
        cache_file = cache_dir / "gpu-prices.latest.json"
        # Reset mtime so the next iteration's _now() sees it as expired.
        import os

        os.utime(cache_file, (0, 0))

    snapshots = sorted((cache_dir / "gpu-prices.snapshots").iterdir())
    assert len(snapshots) == 3
    # Filenames are sortable by UTC time.
    assert snapshots[0].name < snapshots[1].name < snapshots[2].name
