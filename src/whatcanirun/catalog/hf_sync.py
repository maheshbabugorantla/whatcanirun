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
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from whatcanirun.catalog.hf_model import (
    KvCacheStrategy,
    Model,
    UnsupportedArchitectureFamily,
    detect_architecture_family,
)

HF_API_BASE = "https://huggingface.co/api/models"
HF_RAW_BASE = "https://huggingface.co"
DEFAULT_TIMEOUT_S = 30.0


class HfModelSyncUnavailable(Exception):  # noqa: N818 (parallel to ComputePricesUnavailable; user-facing identifier)
    """Raised when Hugging Face is unreachable after retries AND no
    cached `Model` exists for the slug. Mirrors M02's
    `ComputePricesUnavailable` shape: caller is expected to surface
    this through a trust envelope explaining the gap (ADR-013)."""


def _is_retryable_http_error(exc: BaseException) -> bool:
    """True for transient errors worth retrying — 429, 5xx, or any
    connection-layer error. 4xx other than 429 is a client bug (bad
    repo_id, missing token for a gated repo); retrying it burns quota
    and masks the real problem."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


# slug becomes a cache filename under <cache_dir>/huggingface/. Restrict to
# the conservative subset that's safe across filesystems AND can't traverse
# (no `/`, no `..`, no leading dot, no shell metacharacters). Matches the
# project's existing lowercase-with-dashes-and-underscores convention for
# CP slugs.
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Hugging Face's documented repo_id grammar: <namespace>/<name>, each
# segment matching ASCII alphanumerics + `._-`. Rejecting anything else
# at the boundary prevents URL-path traversal (`foo/../bar`), query
# string injection (`?token=`), userinfo segments (`@evil.com/x`), and
# extra slashes that would target an unrelated HF endpoint with the
# user's bearer token attached.
_SAFE_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class HfModelSync:
    """Sync HF config.json metadata for tracked models with on-disk cache."""

    def __init__(
        self,
        cache_dir: Path,
        hf_token: str | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retry_attempts: int = 4,
        retry_wait_min_s: float = 1.0,
        retry_wait_max_s: float = 4.0,
    ) -> None:
        self._hf_dir = cache_dir / "huggingface"
        # Match M02's empty-string-is-anonymous semantics so a CI safeguard
        # `HF_TOKEN=""` doesn't produce a malformed `Authorization: Bearer `.
        if hf_token is None:
            env_token = os.environ.get("HF_TOKEN", "").strip()
            hf_token = env_token or None
        self._hf_token = hf_token
        self._timeout_s = timeout_s
        # Retries: total attempts incl. the first one (spec: initial + 3
        # retries on 429/5xx / connection errors). Tests pass wait_*_s=0
        # so the suite doesn't sleep ~7s on every fallback path.
        self._retry_attempts = retry_attempts
        self._retry_wait_min_s = retry_wait_min_s
        self._retry_wait_max_s = retry_wait_max_s

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

        Both `slug` and `repo_id` are validated at this boundary — `slug`
        is interpolated into the cache filename and `repo_id` into the
        HF URL path, so a malformed value here is a path-traversal /
        URL-injection vector. Invalid values raise `ValueError` BEFORE
        any HTTP call or filesystem write.
        """
        if not _SAFE_SLUG_RE.match(slug):
            raise ValueError(
                f"invalid slug {slug!r}: must match {_SAFE_SLUG_RE.pattern} "
                "(lowercase alphanumerics + `._-`, no path separators)"
            )
        if not _SAFE_REPO_ID_RE.match(repo_id):
            raise ValueError(
                f"invalid repo_id {repo_id!r}: must match {_SAFE_REPO_ID_RE.pattern} "
                "(HF's documented `<namespace>/<name>` format)"
            )

        cache_path = self._hf_dir / f"{slug}.model.json"

        try:
            info = await self._fetch_info_with_retry(repo_id)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            # 4xx other than 429 already escaped (retry policy didn't match);
            # only retryable errors land here.
            if not _is_retryable_http_error(exc):
                raise
            cached_model = self._read_any_cached_model(cache_path)
            if cached_model is not None:
                return cached_model
            raise HfModelSyncUnavailable(
                f"Hugging Face Hub unreachable for {repo_id!r} after "
                f"{self._retry_attempts} attempts and no cached Model "
                f"exists at {cache_path}"
            ) from exc

        sha = info["sha"]
        cached_model = self._read_cached_model(cache_path, sha)
        if cached_model is not None:
            return cached_model

        try:
            raw_config = await self._fetch_config_with_retry(repo_id, sha)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            if not _is_retryable_http_error(exc):
                raise
            # Info succeeded but config didn't; if we still have ANY
            # cached Model for this slug (even a stale-SHA one), prefer
            # that over a hard failure.
            cached_model = self._read_any_cached_model(cache_path)
            if cached_model is not None:
                return cached_model
            raise HfModelSyncUnavailable(
                f"Hugging Face config.json fetch for {repo_id!r}@{sha} failed "
                f"after {self._retry_attempts} attempts and no cached Model "
                f"exists at {cache_path}"
            ) from exc
        # ADR-015 invariant #2: persist the raw upstream response verbatim
        # BEFORE parsing, so a future projection bug or schema-evolution
        # check can re-read the exact bytes HF returned. The projection
        # (Model JSON) is also cached so cache-hit reads are fast. Raw
        # is persisted BEFORE the family check below, so even on the
        # unsupported-family skip path the investigator can read what
        # HF actually returned.
        self._hf_dir.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self._raw_path(slug), json.dumps(raw_config))

        # Unsupported architecture family → raise so `sync_all_tracked`
        # can log + skip + continue with the next model rather than
        # silently caching a Model with family="other" that downstream
        # M06 / M07 don't know how to interpret. Per spec/M03 § Failure
        # modes: "Unknown architecture family: skip with logged warning,
        # raw_config still cached" — both halves satisfied here.
        detected_family = detect_architecture_family(raw_config)
        if detected_family == "other":
            archs = raw_config.get("architectures") or []
            head = archs[0] if archs else "<missing architectures>"
            raise UnsupportedArchitectureFamily(
                f"unsupported architecture family for {repo_id!r} (slug={slug!r}): "
                f"architectures[0]={head!r} doesn't match any known prefix. "
                f"Raw config persisted at {self._raw_path(slug)} for inspection."
            )

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
        self._write_atomic(cache_path, model.model_dump_json())
        return model

    # --------------------------------------------------------------- internals

    def _raw_path(self, slug: str) -> Path:
        """Path to the verbatim HF config.json cache for `slug` (ADR-015)."""
        return self._hf_dir / f"{slug}.config.json"

    @staticmethod
    def _read_any_cached_model(cache_path: Path) -> Model | None:
        """Stale-tolerant cache read for the upstream-down fallback path.

        Same shape-resilience as `_read_cached_model` but does NOT
        require the cached SHA to match anything — when HF is
        unreachable we don't know the current SHA. Returning a stale
        Model is honest under ADR-013 ("never fail tool calls outright")
        because the trust envelope (M08) carries the freshness signal.
        """
        if not cache_path.exists():
            return None
        try:
            decoded = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        try:
            return Model.model_validate(decoded)
        except ValidationError:
            return None

    @staticmethod
    def _read_cached_model(cache_path: Path, expected_sha: str) -> Model | None:
        """Return the cached Model when it exists, parses as a dict with
        a matching `hf_revision_sha`, AND validates against the current
        Model schema. Any other state — file missing, malformed JSON,
        non-dict decoded value, missing/mismatched SHA, or
        ValidationError from a schema migration — returns None so the
        caller transparently refetches. The cache must NEVER make the
        callable harder to use than no cache at all.
        """
        if not cache_path.exists():
            return None
        try:
            decoded = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        if decoded.get("hf_revision_sha") != expected_sha:
            return None
        try:
            return Model.model_validate(decoded)
        except ValidationError:
            # Schema drift: a future Model field that the cached row
            # doesn't carry. Treat as cache miss; refetch produces a
            # fresh, schema-current Model.
            return None

    @staticmethod
    def _write_atomic(path: Path, contents: str) -> None:
        """Tmp + rename so a crash mid-write can't leave a half-written
        file that the next read would refuse to parse."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(contents)
        tmp.replace(path)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"
        return headers

    async def _fetch_info_with_retry(self, repo_id: str) -> dict[str, Any]:
        """`_fetch_info` wrapped in tenacity retry on 429/5xx/RequestError.

        4xx other than 429 bubbles immediately so the caller sees the
        real client-side error (bad repo_id, missing token for a gated
        repo) rather than the timed-out retry budget.
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(min=self._retry_wait_min_s, max=self._retry_wait_max_s),
            retry=retry_if_exception(_is_retryable_http_error),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._fetch_info(repo_id)
        raise AssertionError(  # pragma: no cover
            "AsyncRetrying with reraise=True exhausted without raising"
        )

    async def _fetch_config_with_retry(self, repo_id: str, sha: str) -> dict[str, Any]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retry_attempts),
            wait=wait_exponential(min=self._retry_wait_min_s, max=self._retry_wait_max_s),
            retry=retry_if_exception(_is_retryable_http_error),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self._fetch_config(repo_id, sha)
        raise AssertionError(  # pragma: no cover
            "AsyncRetrying with reraise=True exhausted without raising"
        )

    async def _fetch_info(self, repo_id: str) -> dict[str, Any]:
        url = f"{HF_API_BASE}/{repo_id}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "sha" not in payload:
            raise ValueError(f"HF model info for {repo_id!r} missing required `sha` field")
        # HF's documented contract: sha is a non-empty string (commit hash).
        # A non-string would silently coerce via str() and produce a malformed
        # URL (`.../raw/None/config.json`) — fail loudly here instead.
        if not isinstance(payload["sha"], str) or not payload["sha"]:
            raise ValueError(
                f"HF model info for {repo_id!r}: `sha` must be a non-empty "
                f"string, got {type(payload['sha']).__name__} {payload['sha']!r}"
            )
        return payload

    async def _fetch_config(self, repo_id: str, sha: str) -> dict[str, Any]:
        url = f"{HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.get(url, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"HF config.json for {repo_id!r}@{sha} is not a JSON object")
        # `architectures` is the discriminator family detection reads;
        # without it, the projection silently routes to "other" and the
        # caller can't tell whether HF dropped the field or we got an
        # unexpected payload shape. Surface the missing key at the
        # boundary so the error names the real fault.
        if "architectures" not in payload:
            raise ValueError(
                f"HF config.json for {repo_id!r}@{sha} missing required "
                "`architectures` field (cannot detect model family)"
            )
        return payload
