"""Async ComputePrices client (M02 Slices B + C).

  - GET /api/v1/gpus         -> list[GpuCatalogRow]    (24h cache)
  - GET /api/v1/gpu-prices   -> list[GpuPriceRow]      (1h cache)
  - GET /api/v1/llm-models   -> list[LlmCatalogRow]    (24h cache)
  - GET /api/v1/llm-prices   -> list[LlmPriceRow]      (1h cache)

Snapshot persistence (D), upstream-down fallback (E), pruning (F), and
schema-evolution test + CI shim revert (G) land in later slices on
this branch.

Auth: optional bearer token from `COMPUTEPRICES_API_KEY`. When unset,
requests go through CP's anonymous 60/hr-per-IP tier. ADR-001.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmCatalogRow,
    LlmPriceRow,
    _CpRow,
)


class ComputePricesUnavailable(Exception):  # noqa: N818  (name fixed by spec/M02-computeprices-client.md)
    """Raised when ComputePrices is unreachable AND no cached fallback
    exists. The caller is expected to surface this to the user with a
    trust envelope explaining why no number can be returned (ADR-013).
    """


def _is_retryable_http_error(exc: BaseException) -> bool:
    """True for transient errors worth retrying — 429, 5xx, or any
    connection-layer error. 4xx (other than 429) is a client bug and
    retrying it would just burn quota.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


CP_BASE_URL = "https://www.computeprices.com/api/v1"
DEFAULT_TIMEOUT_S = 30.0

# Per-endpoint cache TTL. Prices change hourly; catalogs change rarely.
# Effective TTL is `_TTL_SECONDS[endpoint] + _jitter_seconds()` so the
# fleet doesn't refresh in lockstep at the top of every hour (pitfall
# #4 in spec/M02). Tests monkeypatch `_jitter_seconds` to 0 when they
# need to assert TTL boundaries exactly.
_TTL_SECONDS: dict[str, int] = {
    "gpus": 24 * 3600,
    "llm-models": 24 * 3600,
    "gpu-prices": 1 * 3600,
    "llm-prices": 1 * 3600,
}


def _now() -> dt.datetime:
    """Module-level clock so tests can monkeypatch TTL behavior without sleeping."""
    return dt.datetime.now(dt.UTC)


_JITTER_RANGE_S = 60.0


def _jitter_seconds() -> float:
    """Random offset added to the TTL cutoff on every cache-age check.

    The point is to desynchronize fleet-wide refreshes — without it,
    every client started at roughly the same time would expire its
    cache simultaneously and hammer upstream in lockstep when prices
    rolled over at the top of the hour (pitfall #4 in spec/M02).

    Sampled per check rather than baked into the cached file's mtime
    so multiple processes sharing one cache directory each get an
    independent jitter window. Tests can monkeypatch this to a fixed
    value (often 0.0) so TTL boundaries assert exactly.
    """
    return random.uniform(-_JITTER_RANGE_S, _JITTER_RANGE_S)


class ComputePricesClient:
    """Async ComputePrices `/api/v1/*` client.

    M02 Slice B surface: just the four fetch methods. Cache & fallback
    behavior is added in Slices C-F; the `cache_dir` argument is
    threaded through now so callers don't need to re-instantiate later.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None = None,
        *,
        base_url: str = CP_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_attempts: int = 4,
        retry_wait_min_s: float = 1.0,
        retry_wait_max_s: float = 4.0,
        snapshot_retention: dt.timedelta = dt.timedelta(days=30),
    ) -> None:
        self.cache_dir = cache_dir
        # When the caller doesn't pass an explicit api_key, fall back to
        # COMPUTEPRICES_API_KEY from the environment. An empty string is
        # treated as "no key" so CI's deliberate `COMPUTEPRICES_API_KEY=""`
        # safeguard keeps test runs anonymous instead of sending a
        # malformed `Authorization: Bearer ` header.
        if api_key is None:
            env_key = os.environ.get("COMPUTEPRICES_API_KEY", "").strip()
            api_key = env_key or None
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        # Retries: total attempts incl. the first one (spec: initial + 3
        # retries). Tests pass wait_*_s=0 so the suite doesn't sleep ~7s
        # for every fallback path exercised.
        self._retry_attempts = retry_attempts
        self._retry_wait_min_s = retry_wait_min_s
        self._retry_wait_max_s = retry_wait_max_s
        self._snapshot_retention = snapshot_retention

    # ---------------------------------------------------------------- public API

    async def get_gpu_catalog(self) -> list[GpuCatalogRow]:
        return await self._fetch_and_project("gpus", GpuCatalogRow)

    async def get_gpu_prices(self) -> list[GpuPriceRow]:
        return await self._fetch_and_project("gpu-prices", GpuPriceRow)

    async def get_llm_catalog(self) -> list[LlmCatalogRow]:
        return await self._fetch_and_project("llm-models", LlmCatalogRow)

    async def get_llm_prices(self) -> list[LlmPriceRow]:
        return await self._fetch_and_project("llm-prices", LlmPriceRow)

    async def get_raw_response(self, endpoint: str) -> dict[str, Any]:
        """Return the full unparsed CP payload for `endpoint`, including
        the top-level `meta` block that the typed-projection methods
        drop. Shares the cache + retry + fallback path with those
        methods, so callers don't burn extra quota by reaching for raw
        data.

        Intended uses (per spec/M02 § Public surface):
          - TrustEnvelope.freshness consumers (M08) reading
            `meta.generated_at`
          - sampling new upstream fields before projecting them

        Raises ValueError if `endpoint` isn't one of the four known CP
        paths; arbitrary upstream URLs are deliberately not callable
        through this method.
        """
        if endpoint not in _TTL_SECONDS:
            raise ValueError(
                f"unknown CP endpoint {endpoint!r}; expected one of {sorted(_TTL_SECONDS)}"
            )
        return await self._fetch_cached_or_live(endpoint)

    # --------------------------------------------------------------- internals

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _fetch_raw(self, endpoint: str) -> dict[str, Any]:
        """Single live GET. No retry, no cache. Retry wrapper is
        `_fetch_raw_with_retry`; cache lookups happen in
        `_fetch_and_project`.
        """
        url = f"{self._base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(f"ComputePrices {endpoint!r}: response missing top-level `data` array")
        if not isinstance(payload["data"], list):
            raise ValueError(
                f"ComputePrices {endpoint!r}: top-level `data` is "
                f"{type(payload['data']).__name__}, expected list"
            )
        return payload

    async def _fetch_raw_with_retry(self, endpoint: str) -> dict[str, Any]:
        """Live GET with tenacity retry on 429/5xx/connection errors.

        4xx other than 429 is treated as a client bug and bubbles
        immediately so the caller sees the real error rather than the
        timed-out retry budget.
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(min=self._retry_wait_min_s, max=self._retry_wait_max_s),
            retry=retry_if_exception(_is_retryable_http_error),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._fetch_raw(endpoint)
        # Unreachable: `reraise=True` makes the loop either return or
        # re-raise the last exception. Present to satisfy mypy's
        # exhaustiveness check on the loop body.
        raise AssertionError(
            "AsyncRetrying with reraise=True exhausted without raising"
        )  # pragma: no cover

    # ---------------------------------------------------------------- cache

    def _cache_path(self, endpoint: str) -> Path:
        return self.cache_dir / f"{endpoint}.latest.json"

    def _cache_age_within_ttl(self, endpoint: str) -> bool:
        path = self._cache_path(endpoint)
        if not path.exists():
            return False
        ttl = _TTL_SECONDS.get(endpoint)
        if ttl is None:
            return False
        cached_at = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
        age_s = (_now() - cached_at).total_seconds()
        return age_s < ttl + _jitter_seconds()

    def _read_cache(self, endpoint: str) -> dict[str, Any]:
        path = self._cache_path(endpoint)
        try:
            decoded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"ComputePrices {endpoint!r}: cache file at {path} unreadable: {exc}"
            ) from exc
        # Cache files can be corrupted (truncated mid-write, hand-edited,
        # disk corruption). Validate the same shape contract `_fetch_raw`
        # enforces on live responses so callers never iterate something
        # that isn't actually a row list.
        if not isinstance(decoded, dict) or "data" not in decoded:
            raise ValueError(
                f"ComputePrices {endpoint!r}: cache file at {path} missing top-level `data`"
            )
        if not isinstance(decoded["data"], list):
            raise ValueError(
                f"ComputePrices {endpoint!r}: cache file at {path} has `data` of type "
                f"{type(decoded['data']).__name__}, expected list"
            )
        return decoded

    def _write_cache(self, endpoint: str, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(endpoint)
        # Atomic write: tmp file + rename so a crash mid-write can't leave a
        # half-written cache file that would later fail json.loads.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    # ----------------------------------------------------------- snapshots

    def _snapshots_dir(self, endpoint: str) -> Path:
        return self.cache_dir / f"{endpoint}.snapshots"

    def _read_most_recent_valid_snapshot(self, endpoint: str) -> dict[str, Any] | None:
        """Walk `<endpoint>.snapshots/` newest-first, return the first
        snapshot whose contents validate against the same shape check
        used for `_read_cache`. Returns None if no snapshots exist or
        every snapshot is unreadable.

        Used by the upstream-down fallback path so the 30-day rolling
        snapshot history (ADR-013) actually delivers on its promised
        role as fallback storage — not just an audit artifact.
        """
        snapshots = self._snapshots_dir(endpoint)
        if not snapshots.is_dir():
            return None
        # Files are named with sortable ISO timestamps; lexicographic
        # reverse sort == newest-first.
        for path in sorted(snapshots.iterdir(), reverse=True):
            try:
                with gzip.open(path, "rt") as f:
                    decoded = json.load(f)
            except (OSError, EOFError, json.JSONDecodeError):
                # Corrupt snapshot — try the next-oldest.
                continue
            if (
                isinstance(decoded, dict)
                and "data" in decoded
                and isinstance(decoded["data"], list)
            ):
                return decoded
        return None

    def _write_snapshot(self, endpoint: str, payload: dict[str, Any]) -> Path:
        """Persist a gzipped snapshot per fetch.

        Filename is the UTC ISO timestamp with `:` replaced by `-` so the
        name is valid on every supported filesystem (Windows in particular).
        Used by Slice E for upstream-down fallback and by Slice F for the
        30-day pruning policy.
        """
        snapshots = self._snapshots_dir(endpoint)
        snapshots.mkdir(parents=True, exist_ok=True)
        ts = _now().strftime("%Y-%m-%dT%H-%M-%SZ")
        path = snapshots / f"{ts}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(payload, f)
        return path

    def prune_snapshots(self, older_than: dt.timedelta) -> int:
        """Delete snapshot files older than `older_than` from every
        endpoint's snapshot directory. Returns the count of deleted
        files. Safe to call when `cache_dir` doesn't exist yet.

        Only files under `<cache_dir>/*.snapshots/` are eligible —
        stray .json.gz files elsewhere in the cache dir are
        untouched (a stray file there is more likely user data than
        leftover snapshot debris).
        """
        if not self.cache_dir.exists():
            return 0
        cutoff = (_now() - older_than).timestamp()
        deleted = 0
        for snapshots_dir in self.cache_dir.glob("*.snapshots"):
            if not snapshots_dir.is_dir():
                continue
            for path in snapshots_dir.iterdir():
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
        return deleted

    async def _fetch_cached_or_live(self, endpoint: str) -> dict[str, Any]:
        """Shared cache + retry + fallback path used by both the typed
        projection methods and the raw-access escape hatch."""
        payload: dict[str, Any] | None = None
        if self._cache_age_within_ttl(endpoint):
            try:
                payload = self._read_cache(endpoint)
            except ValueError:
                # Cache is corrupt or malformed — refetch as if the file
                # never existed. Logging happens at the M08 layer where
                # the trust envelope makes it actionable.
                payload = None
        if payload is None:
            try:
                payload = await self._fetch_raw_with_retry(endpoint)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                # Per ADR-013: serve last-good cache rather than fail outright.
                # If even that is missing OR corrupt, the caller deserves
                # an explicit signal (no silent empty result, no leaked
                # JSONDecodeError).
                if not _is_retryable_http_error(exc):
                    # 4xx etc. — surface the real error to the caller.
                    raise
                # Fallback order per ADR-013:
                #   1. latest.json if present and valid
                #   2. most-recent valid snapshot (rolling history)
                #   3. ComputePricesUnavailable
                if self._cache_path(endpoint).exists():
                    try:
                        payload = self._read_cache(endpoint)
                    except ValueError:
                        # latest.json is unreadable; fall through to snapshots.
                        payload = self._read_most_recent_valid_snapshot(endpoint)
                else:
                    payload = self._read_most_recent_valid_snapshot(endpoint)
                if payload is None:
                    raise ComputePricesUnavailable(
                        f"ComputePrices {endpoint!r} unreachable after "
                        f"{self._retry_attempts} attempts and no usable "
                        f"latest.json or snapshot exists under "
                        f"{self._cache_path(endpoint).parent}"
                    ) from exc
            else:
                self._write_cache(endpoint, payload)
                self._write_snapshot(endpoint, payload)
                # Opportunistic prune: bounded I/O cost per live fetch
                # keeps the cache dir from growing without bound on
                # long-running deployments.
                self.prune_snapshots(older_than=self._snapshot_retention)
        return payload

    async def _fetch_and_project[Row: _CpRow](
        self, endpoint: str, row_model: type[Row]
    ) -> list[Row]:
        payload = await self._fetch_cached_or_live(endpoint)
        return [row_model.project(item) for item in payload["data"]]
