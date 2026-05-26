"""Snapshot pruning tests for ComputePricesClient.

Snapshots older than the retention window (default 30 days) are
deleted on every fetch so the on-disk cache doesn't grow without
bound. Pruning age is a constructor kwarg so tests can shorten the
window instead of fast-forwarding mtime across weeks.
"""

from __future__ import annotations

import datetime as dt
import gzip
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


def _seed_snapshot(snapshots_dir: Path, ts: dt.datetime, payload: dict[str, Any]) -> Path:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    name = ts.strftime("%Y-%m-%dT%H-%M-%SZ") + ".json.gz"
    path = snapshots_dir / name
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)
    os.utime(path, (ts.timestamp(), ts.timestamp()))
    return path


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


# ------------------------------------------------------------ default 30-day


@pytest.mark.asyncio
@respx.mock
async def test_fetch_prunes_snapshots_older_than_30_days(cache_dir: Path) -> None:
    payload = _payload("cp_gpus_2026-05-26.json")
    snapshots_dir = cache_dir / "gpus.snapshots"

    now = dt.datetime(2026, 5, 26, 12, 0, 0, tzinfo=dt.UTC)
    fresh_ts = now - dt.timedelta(days=10)  # keep
    old_ts = now - dt.timedelta(days=40)  # prune
    edge_ts = now - dt.timedelta(days=29, hours=23)  # keep
    way_old = now - dt.timedelta(days=120)  # prune

    fresh = _seed_snapshot(snapshots_dir, fresh_ts, payload)
    old = _seed_snapshot(snapshots_dir, old_ts, payload)
    edge = _seed_snapshot(snapshots_dir, edge_ts, payload)
    way = _seed_snapshot(snapshots_dir, way_old, payload)

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp_mod, "_now", lambda: now)
        client = ComputePricesClient(cache_dir=cache_dir)
        await client.get_gpu_catalog()

    survivors = sorted(p.name for p in snapshots_dir.iterdir())
    assert fresh.exists()
    assert edge.exists()
    assert not old.exists()
    assert not way.exists()
    # 2 surviving old snapshots + the new one this fetch just wrote.
    assert len(survivors) == 3


# ------------------------------------------------------ configurable retention


@pytest.mark.asyncio
@respx.mock
async def test_retention_age_is_overridable(cache_dir: Path) -> None:
    """Caller can shorten retention without subclassing — useful for
    smaller deployments or CI hygiene."""
    payload = _payload("cp_gpus_2026-05-26.json")
    snapshots_dir = cache_dir / "gpus.snapshots"

    now = dt.datetime(2026, 5, 26, 12, 0, 0, tzinfo=dt.UTC)
    keep_ts = now - dt.timedelta(days=3)
    prune_ts = now - dt.timedelta(days=10)
    keep = _seed_snapshot(snapshots_dir, keep_ts, payload)
    prune = _seed_snapshot(snapshots_dir, prune_ts, payload)

    respx.get(f"{_BASE}/gpus").mock(return_value=httpx.Response(200, json=payload))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp_mod, "_now", lambda: now)
        client = ComputePricesClient(cache_dir=cache_dir, snapshot_retention=dt.timedelta(days=7))
        await client.get_gpu_catalog()

    assert keep.exists()
    assert not prune.exists()


# ---------------------------------------------------- direct method available


@pytest.mark.asyncio
async def test_prune_snapshots_method_callable_directly(cache_dir: Path) -> None:
    """Spec acceptance criterion: `prune_snapshots(older_than=...)` is
    a public method, not only an implicit post-fetch step. Useful for
    a periodic cleanup job that doesn't want to trigger a fetch."""
    payload = _payload("cp_gpus_2026-05-26.json")
    snapshots_dir = cache_dir / "gpus.snapshots"

    now = dt.datetime(2026, 5, 26, 12, 0, 0, tzinfo=dt.UTC)
    keep = _seed_snapshot(snapshots_dir, now - dt.timedelta(days=5), payload)
    prune = _seed_snapshot(snapshots_dir, now - dt.timedelta(days=40), payload)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp_mod, "_now", lambda: now)
        client = ComputePricesClient(cache_dir=cache_dir)
        deleted = client.prune_snapshots(older_than=dt.timedelta(days=30))

    assert keep.exists()
    assert not prune.exists()
    assert deleted == 1


# ----------------------------------------------- safe when nothing to prune


@pytest.mark.asyncio
async def test_prune_returns_zero_when_no_snapshots(cache_dir: Path) -> None:
    client = ComputePricesClient(cache_dir=cache_dir)
    assert client.prune_snapshots(older_than=dt.timedelta(days=30)) == 0


@pytest.mark.asyncio
async def test_prune_skips_non_snapshot_files(cache_dir: Path) -> None:
    """A stray .json.gz outside `<endpoint>.snapshots/` MUST NOT be
    nuked — only files under the snapshot directories are in scope."""
    snapshots_dir = cache_dir / "gpus.snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    # Decoy: not under any snapshots dir.
    decoy = cache_dir / "random.json.gz"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(b"deliberately not a snapshot")
    very_old = time.time() - 365 * 24 * 3600
    os.utime(decoy, (very_old, very_old))

    client = ComputePricesClient(cache_dir=cache_dir)
    client.prune_snapshots(older_than=dt.timedelta(days=30))

    assert decoy.exists()
