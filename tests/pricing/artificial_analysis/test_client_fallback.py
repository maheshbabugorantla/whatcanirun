"""Slice H: AA upstream failure must NOT fail the parent tool call.

Per spec/M04 § Acceptance criteria: "If AA returns 401/429/500: falls
back gracefully without failing parent tool call." AA is OPTIONAL
enrichment; M07 Tier 2 routes around it. When the upstream is
unreachable AND no cached snapshot exists, the client returns an
empty list and logs a warning — silently empty would be dishonest
(downstream might mistake it for "AA tracks nothing"), but raising
to the caller would propagate AA's failure into the user-visible
budget-to-plan response, defeating the optional-by-design contract.

This is the ADR-013 "never fail tool calls outright" pattern with an
AA-specific twist: the fallback to empty is only allowed because AA
is optional. CP/HF are required upstreams and raise
`*Unavailable` exceptions when their cache fallback also fails.
"""

from __future__ import annotations

import datetime as dt
import gzip
import logging
from pathlib import Path

import httpx
import pytest
import respx

from whatcanirun.pricing.artificial_analysis import (
    AA_MODELS_URL,
    ArtificialAnalysisClient,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def fast_client(cache_dir: Path) -> ArtificialAnalysisClient:
    return ArtificialAnalysisClient(
        cache_dir=cache_dir,
        api_key="k",
        retry_attempts=2,
        retry_wait_min_s=0.0,
        retry_wait_max_s=0.0,
    )


# -------------------------------------------- 5xx with valid cache → cache served


@pytest.mark.asyncio
@respx.mock
async def test_5xx_with_recent_cache_serves_cache_silently(
    fast_client: ArtificialAnalysisClient, cache_dir: Path, tmp_path: Path
) -> None:
    """A 5xx during refresh while the cache is still within TTL
    means the existing TTL gate already short-circuits the fetch —
    AA never sees the request. No fallback logic needed for this
    case (the cache hit IS the fallback). Test pins that behavior."""
    # Pre-warm the cache by writing a valid payload directly.
    aa_dir = cache_dir / "artificial_analysis"
    aa_dir.mkdir(parents=True)
    (aa_dir / "models.latest.json").write_bytes(
        b'{"status": 200, "data": [{"id": "x", "slug": "x", "name": "x", '
        b'"model_creator": {"id": "y", "name": "v", "slug": "v"}, '
        b'"release_date": "2026-01-01", "pricing": {}, "evaluations": {}, '
        b'"median_output_tokens_per_second": 100.0, '
        b'"median_time_to_first_token_seconds": 0.5, '
        b'"median_time_to_first_answer_token": 1.0}]}'
    )
    # Even if we mock 5xx, the cache age check short-circuits the
    # fetch — route below should never be called.
    route = respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(500))

    rows = await fast_client.get_models()
    assert len(rows) == 1
    assert route.call_count == 0


# -------------------------------------- 5xx with stale cache → stale cache served


@pytest.mark.asyncio
@respx.mock
async def test_schema_breaking_response_falls_back_to_cache_not_raises(
    fast_client: ArtificialAnalysisClient,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If AA ships a payload restructuring (e.g. renames `data` to
    `models`), the shape-validation `ValueError` MUST NOT propagate
    to the parent tool call. The fallback machinery should recover
    from the existing cache. This is the same trust-contract path
    HTTP failures take — the failure mode (upstream broken,
    parent must keep working) is identical, the cause just differs.

    Pre-warm a valid cache, walk time past TTL so a refresh fires,
    then serve a 200 with a missing-`data` payload. Caller sees the
    cached rows, not a crash.

    Uses `monkeypatch.setattr` (not bare `aa_mod._now = ...`) so the
    patched lambdas get restored on teardown — without that, a
    follow-up test in the same session would inherit a frozen
    12h-in-the-future clock and see spurious cache-miss/hit failures
    depending on test order.
    """
    from whatcanirun.pricing.artificial_analysis import client as aa_mod

    aa_dir = cache_dir / "artificial_analysis"
    aa_dir.mkdir(parents=True)
    cache_file = aa_dir / "models.latest.json"
    cache_file.write_bytes(
        b'{"status": 200, "data": [{"id": "stale-row", "slug": "stale", '
        b'"name": "stale", "model_creator": {"id": "y", "name": "v", '
        b'"slug": "v"}, "release_date": "2026-01-01", "pricing": {}, '
        b'"evaluations": {}, "median_output_tokens_per_second": 100.0, '
        b'"median_time_to_first_token_seconds": 0.5, '
        b'"median_time_to_first_answer_token": 1.0}]}'
    )
    # Make the existing cache stale so a refresh fires.
    import datetime as dt_mod

    real_mtime = dt_mod.datetime.fromtimestamp(cache_file.stat().st_mtime, tz=dt_mod.UTC)
    monkeypatch.setattr(aa_mod, "_jitter_seconds", lambda: 0.0)
    monkeypatch.setattr(aa_mod, "_now", lambda: real_mtime + dt_mod.timedelta(hours=12))

    # AA returns 200 but with the `data` array renamed to `models`
    # — a breaking schema change. Without the ValueError-as-
    # fallback fix, this raises and crashes the parent tool call.
    respx.get(AA_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"status": 200, "models": []})
    )

    rows = await fast_client.get_models()
    assert len(rows) == 1
    assert rows[0].slug == "stale"


@pytest.mark.asyncio
@respx.mock
async def test_5xx_with_stale_cache_serves_stale_silently(
    fast_client: ArtificialAnalysisClient,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cache is past TTL AND AA is down (5xx), the
    ADR-013 fallback returns the stale cache rather than failing.
    M09's trust envelope will mark `freshness["artificial_analysis"]`
    accordingly — staleness is the user's signal, not silent
    degradation."""
    from whatcanirun.pricing.artificial_analysis import client as aa_mod

    monkeypatch.setattr(aa_mod, "_jitter_seconds", lambda: 0.0)
    aa_dir = cache_dir / "artificial_analysis"
    aa_dir.mkdir(parents=True)
    cache_file = aa_dir / "models.latest.json"
    cache_file.write_bytes(
        b'{"status": 200, "data": [{"id": "stale-row", "slug": "stale", '
        b'"name": "stale", "model_creator": {"id": "y", "name": "v", '
        b'"slug": "v"}, "release_date": "2026-01-01", "pricing": {}, '
        b'"evaluations": {}, "median_output_tokens_per_second": 100.0, '
        b'"median_time_to_first_token_seconds": 0.5, '
        b'"median_time_to_first_answer_token": 1.0}]}'
    )

    # Walk time forward past TTL.
    real_mtime = dt.datetime.fromtimestamp(cache_file.stat().st_mtime, tz=dt.UTC)
    monkeypatch.setattr(aa_mod, "_now", lambda: real_mtime + dt.timedelta(hours=12))
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(500, text="boom"))

    rows = await fast_client.get_models()
    assert len(rows) == 1
    assert rows[0].slug == "stale"


# ----------------------------------------- 5xx with no cache → empty list + log


@pytest.mark.asyncio
@respx.mock
async def test_5xx_without_cache_returns_empty_list_and_logs_warning(
    fast_client: ArtificialAnalysisClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The hardest case: AA is down on the very first sync, no
    cache exists. The optional-by-design contract means we MUST NOT
    raise to the caller — M07's Tier 2 routing depends on AA
    failures degrading silently to "no aggregate available" (which
    is functionally identical to "AA doesn't track this slug").

    The warning gets logged so an operator can investigate; the
    caller sees an empty list and falls through to Tier 3/4."""
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(500, text="boom"))

    with caplog.at_level(logging.WARNING):
        rows = await fast_client.get_models()

    assert rows == []
    assert any("artificial analysis" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_connect_error_without_cache_returns_empty_list(
    fast_client: ArtificialAnalysisClient,
) -> None:
    """Same behavior for connection-layer errors — DNS failure,
    TLS handshake timeout, whatever. AA optional means the parent
    tool call never fails because of AA."""
    respx.get(AA_MODELS_URL).mock(side_effect=httpx.ConnectError("dns down"))

    rows = await fast_client.get_models()
    assert rows == []


# ------------------------------------------- 401 with no cache → empty + log


@pytest.mark.asyncio
@respx.mock
async def test_401_without_cache_returns_empty_list_and_logs(
    fast_client: ArtificialAnalysisClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """401 = bad API key. The operator should fix the key, but
    while they investigate, the parent tool call must keep
    working. Empty list + logged warning."""
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(401, text="bad key"))

    with caplog.at_level(logging.WARNING):
        rows = await fast_client.get_models()

    assert rows == []
    assert any(("401" in rec.message or "key" in rec.message.lower()) for rec in caplog.records)


# --------------------------------------- snapshot-only recovery (no live cache)


@pytest.mark.asyncio
@respx.mock
async def test_snapshot_recovery_when_latest_missing(
    fast_client: ArtificialAnalysisClient,
    cache_dir: Path,
) -> None:
    """If `models.latest.json` is missing but a snapshot exists
    (e.g. someone manually deleted the latest file, or a prior
    write was interrupted), AND AA is down on this attempt, the
    fallback walks the snapshots directory and serves the most
    recent valid one. Mirrors M02's snapshot-fallback pattern."""
    aa_dir = cache_dir / "artificial_analysis"
    snapshots_dir = aa_dir / "models.snapshots"
    snapshots_dir.mkdir(parents=True)
    snap_payload = (
        b'{"status": 200, "data": [{"id": "snap", "slug": "from-snap", '
        b'"name": "snap", "model_creator": {"id": "y", "name": "v", '
        b'"slug": "v"}, "release_date": "2026-01-01", "pricing": {}, '
        b'"evaluations": {}, "median_output_tokens_per_second": 100.0, '
        b'"median_time_to_first_token_seconds": 0.5, '
        b'"median_time_to_first_answer_token": 1.0}]}'
    )
    snap_path = snapshots_dir / "2026-05-26T12-00-00Z.json.gz"
    with gzip.open(snap_path, "wb") as f:
        f.write(snap_payload)

    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(500))

    rows = await fast_client.get_models()
    assert len(rows) == 1
    assert rows[0].slug == "from-snap"
