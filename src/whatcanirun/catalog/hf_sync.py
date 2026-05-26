"""Async Hugging Face Hub sync for tracked models.

`HfModelSync.sync_model(repo_id, ...)` fetches the model's current
revision SHA from `https://huggingface.co/api/models/{repo_id}` and
the config.json at that revision, projects through
`Model.from_hf_config`, persists the resulting `Model` JSON to disk,
and returns it. Subsequent calls at the same revision SHA skip the
config.json fetch (the cheap info endpoint is always consulted because
that's how we learn the SHA hasn't changed).

Auth: optional bearer token from `HF_TOKEN`. Empty / whitespace env
variable is treated as anonymous — same CI safeguard pattern M02's
ComputePrices client uses for `COMPUTEPRICES_API_KEY`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import httpx

from whatcanirun.catalog.hf_model import KvCacheStrategy, Model

HF_API_BASE = "https://huggingface.co/api/models"
HF_RAW_BASE = "https://huggingface.co"
DEFAULT_TIMEOUT_S = 30.0


class HfModelSync:
    """Sync HF config.json metadata for tracked models with on-disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        hf_token: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._hf_dir = cache_dir / "huggingface"
        # Match M02's empty-string-is-anonymous semantics so a CI safeguard
        # `HF_TOKEN=""` doesn't produce a malformed `Authorization: Bearer `.
        if hf_token is None:
            env_token = os.environ.get("HF_TOKEN", "").strip()
            hf_token = env_token or None
        self._hf_token = hf_token
        self._timeout_s = timeout_s

    async def sync_model(
        self,
        *,
        repo_id: str,
        slug: str,
        display_name: str,
        total_params_b: float,
        active_params_b: float | None,
        kv_cache_strategy_override: KvCacheStrategy | None = None,
    ) -> Model:
        """Fetch + project + cache one model.

        `total_params_b` and `active_params_b` are explicit kwargs
        because the HF config.json doesn't carry them (the model card
        does, or safetensors metadata via a separate fetch). The
        `sync_all_tracked` caller passes them through from the
        tracked-models YAML row.
        """
        info = await self._fetch_info(repo_id)
        sha = str(info["sha"])

        cache_path = self._hf_dir / f"{slug}.model.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                cached = None
            if cached and cached.get("hf_revision_sha") == sha:
                return Model.model_validate(cached)

        raw_config = await self._fetch_config(repo_id, sha)
        model = Model.from_hf_config(
            slug=slug,
            hf_repo_id=repo_id,
            display_name=display_name,
            total_params_b=total_params_b,
            active_params_b=active_params_b,
            raw_config=raw_config,
            raw_safetensors_meta={},
            hf_revision_sha=sha,
            last_synced_at=dt.datetime.now(dt.UTC),
            # Pass the override through as-is. When the caller didn't supply
            # one (None), `Model.from_hf_config` auto-derives the strategy
            # from `architecture_family` (so DeepSeek-V3 configs become
            # `kv_cache_strategy="mla"` rather than the standard_gqa default).
            # Bypassing this with a literal `or "standard_gqa"` fallback
            # would silently mis-classify MLA models.
            kv_cache_strategy=kv_cache_strategy_override,
        )
        self._hf_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write per the same tmp+rename pattern M02 uses for its
        # cache files, so a crash mid-write can't leave a half-written
        # JSON that the next call would refuse to parse.
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(model.model_dump_json())
        tmp.replace(cache_path)
        return model

    # --------------------------------------------------------------- internals

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"
        return headers

    async def _fetch_info(self, repo_id: str) -> dict[str, Any]:
        url = f"{HF_API_BASE}/{repo_id}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "sha" not in payload:
            raise ValueError(f"HF model info for {repo_id!r} missing required `sha` field")
        return payload

    async def _fetch_config(self, repo_id: str, sha: str) -> dict[str, Any]:
        url = f"{HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"HF config.json for {repo_id!r}@{sha} is not a JSON object")
        return payload
