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
import json
from pathlib import Path
from typing import Any

import httpx

from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmCatalogRow,
    LlmPriceRow,
    _CpRow,
)

CP_BASE_URL = "https://www.computeprices.com/api/v1"
DEFAULT_TIMEOUT_S = 30.0

# Per-endpoint cache TTL. Prices change hourly; catalogs change rarely.
# Pitfall #4: real cache misses use TTL ± 60s jitter — implemented in
# `_cache_age_within_ttl` so tests can monkeypatch deterministically.
_TTL_SECONDS: dict[str, int] = {
    "gpus": 24 * 3600,
    "llm-models": 24 * 3600,
    "gpu-prices": 1 * 3600,
    "llm-prices": 1 * 3600,
}


def _now() -> dt.datetime:
    """Module-level clock so tests can monkeypatch TTL behavior without sleeping."""
    return dt.datetime.now(dt.UTC)


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
    ) -> None:
        self.cache_dir = cache_dir
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    # ---------------------------------------------------------------- public API

    async def get_gpu_catalog(self) -> list[GpuCatalogRow]:
        return await self._fetch_and_project("gpus", GpuCatalogRow)

    async def get_gpu_prices(self) -> list[GpuPriceRow]:
        return await self._fetch_and_project("gpu-prices", GpuPriceRow)

    async def get_llm_catalog(self) -> list[LlmCatalogRow]:
        return await self._fetch_and_project("llm-models", LlmCatalogRow)

    async def get_llm_prices(self) -> list[LlmPriceRow]:
        return await self._fetch_and_project("llm-prices", LlmPriceRow)

    # --------------------------------------------------------------- internals

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _fetch_raw(self, endpoint: str) -> dict[str, Any]:
        """Single live GET. No retry, no cache. Slices D-E layer those on."""
        url = f"{self._base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(f"ComputePrices {endpoint!r}: response missing top-level `data` array")
        return payload

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
        return age_s < ttl

    def _read_cache(self, endpoint: str) -> dict[str, Any]:
        path = self._cache_path(endpoint)
        try:
            return json.loads(path.read_text())  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"ComputePrices {endpoint!r}: cache file at {path} unreadable: {exc}"
            ) from exc

    def _write_cache(self, endpoint: str, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(endpoint)
        # Atomic write: tmp file + rename so a crash mid-write can't leave a
        # half-written cache file that would later fail json.loads.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    async def _fetch_and_project[Row: _CpRow](
        self, endpoint: str, row_model: type[Row]
    ) -> list[Row]:
        if self._cache_age_within_ttl(endpoint):
            payload = self._read_cache(endpoint)
        else:
            payload = await self._fetch_raw(endpoint)
            self._write_cache(endpoint, payload)
        return [row_model.project(item) for item in payload["data"]]
