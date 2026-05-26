"""Async ComputePrices client (M02 Slice B).

This slice covers the happy path for the four `/api/v1/*` endpoints:

  - GET /api/v1/gpus         -> list[GpuCatalogRow]
  - GET /api/v1/gpu-prices   -> list[GpuPriceRow]
  - GET /api/v1/llm-models   -> list[LlmCatalogRow]
  - GET /api/v1/llm-prices   -> list[LlmPriceRow]

Cache (Slice C), snapshot persistence (D), upstream-down fallback (E),
pruning (F), and schema-evolution test + CI shim revert (G) land in
later slices on this branch.

Auth: optional bearer token from `COMPUTEPRICES_API_KEY`. When unset,
requests go through CP's anonymous 60/hr-per-IP tier. ADR-001.
"""

from __future__ import annotations

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
        """Single GET, no retry, no cache. Slices C-E layer those on."""
        url = f"{self._base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(f"ComputePrices {endpoint!r}: response missing top-level `data` array")
        return payload

    async def _fetch_and_project[Row: _CpRow](
        self, endpoint: str, row_model: type[Row]
    ) -> list[Row]:
        payload = await self._fetch_raw(endpoint)
        return [row_model.project(item) for item in payload["data"]]
