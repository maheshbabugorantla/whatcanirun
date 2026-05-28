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
"""

from __future__ import annotations

import os
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
    """AA `/api/v2/data/llms/models` client with optional auth.

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
        return await self._fetch_with_retry()

    # ---------------------------------------------------------------- internals

    def _headers(self) -> dict[str, str]:
        """Build request headers. AA uses `X-Api-Key`, NOT
        `Authorization: Bearer` (verified live 2026-05-27)."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key is not None:
            headers["X-Api-Key"] = self._api_key
        return headers

    async def _fetch(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(self._base_url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"AA response top-level must be a JSON object, got {type(payload).__name__}"
            )
        return payload

    async def _fetch_with_retry(self) -> dict[str, Any]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(min=self._retry_wait_min_s, max=self._retry_wait_max_s),
            retry=retry_if_exception(_is_retryable_http_error),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._fetch()
        raise AssertionError(  # pragma: no cover
            "AsyncRetrying with reraise=True exhausted without raising"
        )
