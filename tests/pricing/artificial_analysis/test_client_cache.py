"""Slice D: 6-hour cache + snapshot persistence.

AA's free-tier budget is 1k/day; ~4 refreshes/day at 6h TTL leaves
plenty of headroom and we're never more than ~3h + jitter stale. The
cache file is `models.latest.json` (raw upstream bytes per ADR-015);
each successful refresh ALSO drops a gzipped snapshot under
`models.snapshots/` for the 30-day audit window M02 established.

Cache hits short-circuit the HTTP call entirely — second call within
the TTL window returns the same payload without touching upstream.

Tests monkeypatch `_jitter_seconds` and `_now` to assert TTL
boundaries exactly; in production both float by their natural amount
so a fleet of clients doesn't refresh in lockstep at the hour
boundary (same desync trick M02 uses).
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

from whatcanirun.pricing.artificial_analysis import (
    ArtificialAnalysisClient,
)
from whatcanirun.pricing.artificial_analysis import client as aa_mod

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"

_AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


@pytest.fixture(scope="module")
def aa_payload_bytes() -> bytes:
    return (_FIXTURES / "aa_models_2026-05-27.json").read_bytes()


@pytest.fixture(scope="module")
def aa_payload(aa_payload_bytes: bytes) -> dict[str, Any]:
    return json.loads(aa_payload_bytes)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def no_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin jitter to 0 so the TTL boundary tests don't randomly miss
    the cutoff. Production keeps the random ±60s window."""
    monkeypatch.setattr(aa_mod, "_jitter_seconds", lambda: 0.0)


# ----------------------------------------------------- happy path: cache write


@pytest.mark.asyncio
@respx.mock
async def test_first_call_writes_raw_bytes_to_cache(
    cache_dir: Path, aa_payload: dict[str, Any], aa_payload_bytes: bytes
) -> None:
    """ADR-015: cache file holds the raw upstream bytes, not a
    json.dumps reserialization of the parsed dict. A future schema-
    evolution audit needs to compare HF/AA bytes against the
    documented schema — not against our normalized rewrite."""
    respx.get(_AA_URL).mock(
        return_value=httpx.Response(
            200,
            content=aa_payload_bytes,
            headers={"content-type": "application/json"},
        )
    )

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    await client.get_models()

    cache_file = cache_dir / "artificial_analysis" / "models.latest.json"
    assert cache_file.exists()
    # Byte-for-byte equality — would fail under text-mode write +
    # locale roundtrip for non-ASCII payloads.
    assert cache_file.read_bytes() == aa_payload_bytes


@pytest.mark.asyncio
@respx.mock
async def test_first_call_writes_dated_snapshot(cache_dir: Path, aa_payload_bytes: bytes) -> None:
    """Every successful refresh drops a `.json.gz` snapshot under
    `models.snapshots/` so the 30-day audit window stays populated.
    Snapshot is the gzipped raw bytes — readable by the standard
    `gzip` module, byte-identical when decompressed."""
    respx.get(_AA_URL).mock(
        return_value=httpx.Response(
            200,
            content=aa_payload_bytes,
            headers={"content-type": "application/json"},
        )
    )

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    await client.get_models()

    snapshots = list((cache_dir / "artificial_analysis" / "models.snapshots").glob("*.json.gz"))
    assert len(snapshots) == 1
    with gzip.open(snapshots[0], "rb") as f:
        decompressed = f.read()
    assert decompressed == aa_payload_bytes


# ------------------------------------------------ cache hit within TTL


@pytest.mark.asyncio
@respx.mock
async def test_second_call_within_ttl_skips_network(
    cache_dir: Path, aa_payload_bytes: bytes, no_jitter: None
) -> None:
    """Second call within 6h returns the cached payload without
    hitting AA. Saves quota AND respects the AA free-tier 1k/day
    budget under heavy CI use."""
    route = respx.get(_AA_URL).mock(
        return_value=httpx.Response(
            200,
            content=aa_payload_bytes,
            headers={"content-type": "application/json"},
        )
    )

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows_first = await client.get_models()
    rows_second = await client.get_models()

    assert len(rows_first) == len(rows_second) == 525
    # Second call MUST NOT have hit the network.
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_call_after_ttl_expiry_refetches(
    cache_dir: Path,
    aa_payload_bytes: bytes,
    no_jitter: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the 6h TTL passes, the next call refetches and rewrites
    the cache + drops a new snapshot.

    Clock-mocking note: `os.path.getmtime` returns real wall-clock
    seconds; the cache mtime is written at REAL time, so monkey-
    patching `_now` to an arbitrary fake timestamp produces a stale
    cache-vs-fake-now skew unless we anchor fake_now to the actual
    mtime of the file we just wrote. Read it back, then walk
    forward 6h + 1s in fake time."""
    route = respx.get(_AA_URL).mock(
        return_value=httpx.Response(
            200,
            content=aa_payload_bytes,
            headers={"content-type": "application/json"},
        )
    )

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    await client.get_models()
    assert route.call_count == 1

    cache_file = cache_dir / "artificial_analysis" / "models.latest.json"
    real_mtime = dt.datetime.fromtimestamp(cache_file.stat().st_mtime, tz=dt.UTC)
    later = real_mtime + dt.timedelta(hours=6, seconds=1)
    monkeypatch.setattr(aa_mod, "_now", lambda: later)

    await client.get_models()
    assert route.call_count == 2

    snapshots = list((cache_dir / "artificial_analysis" / "models.snapshots").glob("*.json.gz"))
    assert len(snapshots) == 2


@pytest.mark.asyncio
@respx.mock
async def test_call_just_before_ttl_expiry_uses_cache(
    cache_dir: Path,
    aa_payload_bytes: bytes,
    no_jitter: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary: at TTL - 1s, cache is still valid. Without
    `no_jitter` this would be flaky."""
    route = respx.get(_AA_URL).mock(
        return_value=httpx.Response(
            200,
            content=aa_payload_bytes,
            headers={"content-type": "application/json"},
        )
    )

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    await client.get_models()

    cache_file = cache_dir / "artificial_analysis" / "models.latest.json"
    real_mtime = dt.datetime.fromtimestamp(cache_file.stat().st_mtime, tz=dt.UTC)
    just_inside = real_mtime + dt.timedelta(hours=6) - dt.timedelta(seconds=1)
    monkeypatch.setattr(aa_mod, "_now", lambda: just_inside)

    await client.get_models()
    assert route.call_count == 1  # cache hit, no refetch
