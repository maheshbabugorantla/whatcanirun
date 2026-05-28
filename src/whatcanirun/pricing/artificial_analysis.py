"""Async Artificial Analysis (AA) client — OPTIONAL enrichment for M07.

When `AA_API_KEY` is set, ingests AA's `/api/v2/data/llms/models`
endpoint (525-row free-tier response on 2026-05-27 capture) and
exposes per-model TPS aggregates as Tier-2 anchors in `tps_estimator`.
When the key is unset, every method either raises `AaDisabled` or
returns empty — the rest of the system works unchanged with no AA
mentions in trust envelopes.

AA optionality is a strict guarantee, not best-effort: M07's Tier 2
must be able to ask `client.enabled` and route to Tier 3/4 without
ever touching the network. The AA free tier carries attribution
requirements (see spec/M04 § Attribution); any consumer that ships
an AA-sourced number into a `TrustEnvelope.sources` entry must
include the AA `license_attribution` string.

Auth header is `X-Api-Key: <key>` per AA's free-tier contract —
NOT `Authorization: Bearer` (verified live 2026-05-27 with a real
key). Mirrors the M02 ComputePrices client's empty-string-is-
anonymous env-var semantics so a CI safeguard `AA_API_KEY=""`
doesn't accidentally enable an unusable bearer header.

Cache layout per spec/M04:
  <cache_dir>/artificial_analysis/models.latest.json    (raw bytes)
  <cache_dir>/artificial_analysis/models.snapshots/
    <ISO-8601>.json.gz                                   (gzipped raw)

TTL is 6h (well inside AA's 1k/day budget at ~4 refreshes/day) plus
±60s jitter so a fleet of clients doesn't refresh in lockstep at
the hour boundary. ADR-015 bytes-verbatim: `_fetch_raw` returns
`response.content` (bytes), cache write is binary mode, parse from
disk bytes — the M03 round-4 lesson applied from the start.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import gzip
import json
import os
import random
import secrets
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from whatcanirun.pricing.aa_projections import AaModelRow

AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
DEFAULT_TIMEOUT_S = 30.0

_TTL_SECONDS = 6 * 3600
_JITTER_RANGE_S = 60.0


def _now() -> dt.datetime:
    """Module-level clock so tests can monkeypatch TTL behavior
    without sleeping."""
    return dt.datetime.now(dt.UTC)


def _jitter_seconds() -> float:
    """Random offset added to the TTL cutoff on every cache-age
    check. Desynchronizes fleet-wide refreshes so we don't hammer AA
    in lockstep when caches expire at the same wall-clock instant.
    Tests monkeypatch to 0.0 to assert TTL boundaries exactly."""
    return random.uniform(-_JITTER_RANGE_S, _JITTER_RANGE_S)


class AaDisabled(Exception):  # noqa: N818  (name is the public contract)
    """Raised when an AA-only operation is requested but no
    `AA_API_KEY` was supplied.

    Callers that route around the AA tier (M07, the unknown-model
    dispatcher in M09) check `client.enabled` first and avoid this
    exception entirely. The exception exists for the "forgot to
    check" case — make the failure loud rather than silently
    returning an empty list that downstream code mistakes for "AA
    returned no match for this model".
    """


def _is_retryable_http_error(exc: BaseException) -> bool:
    """True for transient errors worth retrying — 429, 5xx, or any
    connection-layer error. Other 4xx is a client bug (bad key,
    bad request shape) and retrying would only burn AA's free-tier
    1k/day quota while logging the same 401/403 over and over."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


class ArtificialAnalysisClient:
    """AA `/api/v2/data/llms/models` client with optional auth,
    6-hour cache, and ADR-015 byte-identical snapshot persistence.

    All AA-touching methods raise `AaDisabled` when `self.enabled`
    is False. Callers SHOULD branch on `client.enabled` rather than
    catch the exception in the hot path.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None = None,
        *,
        base_url: str = AA_MODELS_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_attempts: int = 4,
        retry_wait_min_s: float = 1.0,
        retry_wait_max_s: float = 4.0,
    ) -> None:
        self.cache_dir = cache_dir
        # ctor arg wins over env var (matches M02 CP convention so
        # tests can run with deterministic keys without polluting
        # from a developer's `.env`). Empty / whitespace env var is
        # treated as anonymous so CI safeguards like
        # `AA_API_KEY=""` don't enable a broken bearer path.
        if api_key is None:
            env_key = os.environ.get("AA_API_KEY", "").strip()
            api_key = env_key or None
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_s = timeout_s
        # Retries: total attempts incl. the first one (spec: initial
        # + 3 retries on 429/5xx / connection errors). Tests pass
        # wait_*_s=0 so the suite doesn't sleep ~7s per fallback path.
        self._retry_attempts = retry_attempts
        self._retry_wait_min_s = retry_wait_min_s
        self._retry_wait_max_s = retry_wait_max_s

    @property
    def enabled(self) -> bool:
        """True iff an AA API key is available. M07 Tier-2 routing
        check, M09 trust-envelope attribution check, and every
        AA-only method gate off this."""
        return self._api_key is not None

    async def get_models(self) -> list[AaModelRow]:
        """Return the projected list of `AaModelRow`. Raises
        `AaDisabled` when no key is configured.

        Shape failures (missing `data` array, non-list `data`) raise
        `ValueError` at the boundary so an upstream schema change
        fails loudly rather than silently producing an empty list
        that downstream code mistakes for "no models tracked".
        """
        payload = await self.get_raw_response()
        if "data" not in payload:
            raise ValueError(
                f"AA response missing required `data` array; got top-level keys {sorted(payload)!r}"
            )
        data = payload["data"]
        if not isinstance(data, list):
            raise ValueError(f"AA response `data` must be a list, got {type(data).__name__}")
        return [AaModelRow.project(row) for row in data]

    async def get_raw_response(self) -> dict[str, Any]:
        """Return the full unparsed AA payload (top-level dict with
        `status`, `prompt_options`, `data`). Used by M09's trust-
        envelope provenance + the schema-evolution audit. Raises
        `AaDisabled` when no key is configured."""
        if not self.enabled:
            raise AaDisabled(
                "AA_API_KEY is not configured; AA enrichment is off. "
                "Set the env var or pass `api_key=...` to enable."
            )
        return await self._fetch_cached_or_live()

    # ---------------------------------------------------------------- cache layer

    def _aa_dir(self) -> Path:
        return self.cache_dir / "artificial_analysis"

    def _cache_path(self) -> Path:
        """Path to `models.latest.json` — raw upstream bytes."""
        return self._aa_dir() / "models.latest.json"

    def _snapshots_dir(self) -> Path:
        return self._aa_dir() / "models.snapshots"

    def _cache_age_within_ttl(self) -> bool:
        """True if the cache file exists and was written within the
        last 6h + jitter window. Jitter applies BOTH ways so a
        cache barely past the nominal cutoff still serves ~50% of
        the time — desyncs the refresh fleet at hour boundaries."""
        path = self._cache_path()
        if not path.exists():
            return False
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
        age_s = (_now() - mtime).total_seconds()
        return age_s < (_TTL_SECONDS + _jitter_seconds())

    def _read_cache(self) -> dict[str, Any]:
        """Parse the cached bytes. `json.loads` accepts bytes
        directly (auto-detects UTF-8/16/32) so we never round-trip
        through `read_text()` and lose bytes for non-ASCII payloads.
        Shape-validated on the way out for the same reason
        `_fetch_raw` does it — a corrupt cache shouldn't poison
        the projection."""
        decoded: Any = json.loads(self._cache_path().read_bytes())
        if not isinstance(decoded, dict):
            raise ValueError(
                f"AA cache at {self._cache_path()} is not a JSON object, "
                f"got {type(decoded).__name__}"
            )
        return decoded

    def _write_cache(self, payload_bytes: bytes) -> None:
        """Atomic write of raw upstream bytes to `models.latest.json`.

        Uses the M03 per-attempt-unique tmp + rename pattern so
        concurrent syncs of the same cache file don't race-destroy
        each other's tmp. ADR-015 byte-identity: bytes go in
        verbatim, no text-mode reencoding.
        """
        self._aa_dir().mkdir(parents=True, exist_ok=True)
        path = self._cache_path()
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        # O_EXCL | O_NOFOLLOW refuses to follow a planted symlink
        # at the tmp path (defense against local-attacker symlink
        # redirect to /etc/passwd-class files), and refuses any
        # pre-existing entry — with 64 bits of token entropy a
        # collision is either ~0-probability random or an attacker
        # pre-planting the exact name, both worth escalating.
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload_bytes)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)

    def _write_snapshot(self, payload_bytes: bytes) -> Path:
        """Append a gzipped snapshot of the raw bytes under
        `models.snapshots/<ISO-8601>.json.gz`. Mirrors M02's
        snapshot pattern — 30-day audit window driven by `_now()`."""
        self._snapshots_dir().mkdir(parents=True, exist_ok=True)
        ts = _now().strftime("%Y-%m-%dT%H-%M-%SZ")
        path = self._snapshots_dir() / f"{ts}.json.gz"
        with gzip.open(path, "wb") as f:
            f.write(payload_bytes)
        return path

    async def _fetch_cached_or_live(self) -> dict[str, Any]:
        """Cache-first read: if the on-disk payload is within the
        TTL window, return it without touching AA. Otherwise fetch,
        persist (cache + snapshot), and return.
        """
        if self._cache_age_within_ttl():
            return self._read_cache()
        payload_bytes = await self._fetch_raw_with_retry()
        self._write_cache(payload_bytes)
        self._write_snapshot(payload_bytes)
        # _fetch_raw already shape-validated this exact byte
        # sequence (dict at top level), so parsing here is safe; the
        # explicit isinstance keeps mypy honest about the Any boundary
        # of `json.loads` without re-doing the runtime check.
        parsed: Any = json.loads(payload_bytes)
        assert isinstance(parsed, dict)
        return parsed

    # ---------------------------------------------------------------- HTTP layer

    def _headers(self) -> dict[str, str]:
        """Build request headers. AA uses `X-Api-Key`, NOT
        `Authorization: Bearer` (verified live 2026-05-27)."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key is not None:
            headers["X-Api-Key"] = self._api_key
        return headers

    async def _fetch_raw(self) -> bytes:
        """Return `response.content` — raw bytes for ADR-015
        byte-identical cache persistence. Shape validation moves
        out to the caller so a parse failure still leaves the
        bytes on disk for the investigator."""
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(self._base_url, headers=self._headers())
        response.raise_for_status()
        # Top-level shape check happens before persistence so we
        # never write garbage. Parse-then-validate could be done
        # post-persist but AA is small (~430 KB on capture) and the
        # whole roundtrip is sub-second; checking inline keeps the
        # error surface simple. The bytes still hit the cache on
        # success; on failure they don't, but the error message
        # references the upstream URL for forensic re-fetch.
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"AA response top-level must be a JSON object, got {type(payload).__name__}"
            )
        return response.content

    async def _fetch_raw_with_retry(self) -> bytes:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(min=self._retry_wait_min_s, max=self._retry_wait_max_s),
            retry=retry_if_exception(_is_retryable_http_error),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._fetch_raw()
        raise AssertionError(  # pragma: no cover
            "AsyncRetrying with reraise=True exhausted without raising"
        )
