"""HTTP + cache tests for `HfModelSync`.

`HfModelSync.sync_model(repo_id, ...)` fetches the model's current
revision SHA from `https://huggingface.co/api/models/{repo_id}` and the
config.json at that revision, projects through `Model.from_hf_config`,
and persists the resulting Model JSON to disk so subsequent calls at
the same revision skip the network entirely.

Live network is never touched in tests — respx stubs the HF endpoints
the same way M02's CP client tests stub ComputePrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.catalog.hf_sync import HfModelSync, HfModelSyncUnavailable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"

_HF_API_BASE = "https://huggingface.co/api/models"
_HF_RAW_BASE = "https://huggingface.co"


@pytest.fixture(scope="module")
def llama_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_llama-3-3-70b_config.json").read_text())


@pytest.fixture(scope="module")
def deepseek_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_deepseek-v3_config.json").read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.fixture
def fast_sync(cache_dir: Path) -> HfModelSync:
    """Sync client with retries on but zero backoff so tests don't
    burn ~7 seconds per fallback path on real exponential sleeps."""
    return HfModelSync(
        cache_dir=cache_dir,
        retry_attempts=4,
        retry_wait_min_s=0.0,
        retry_wait_max_s=0.0,
    )


# ---------------------------------------------------- happy path: live -> cache


@pytest.mark.asyncio
@respx.mock
async def test_first_sync_fetches_from_hf_then_caches(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """First sync of a model hits HF for both the model-info endpoint
    (to get the revision SHA) and the raw config.json. Result is
    persisted under the cache dir so a follow-up call can skip HF."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc123"

    info_route = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    config_route = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert info_route.called
    assert config_route.called
    assert model.slug == "llama-3-3-70b"
    assert model.hf_revision_sha == sha
    assert model.n_kv_heads == 8

    # Projection cache file written.
    cache_file = cache_dir / "huggingface" / "llama-3-3-70b.model.json"
    assert cache_file.exists()


@pytest.mark.asyncio
@respx.mock
async def test_first_sync_persists_raw_config_per_adr_015(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """ADR-015 invariant #2: every upstream API response is persisted to
    disk verbatim BEFORE parsing. The raw config.json must exist on
    disk after sync, not just the typed projection."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc123"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    raw_cache_file = cache_dir / "huggingface" / "llama-3-3-70b.config.json"
    assert raw_cache_file.exists()
    on_disk = json.loads(raw_cache_file.read_text())
    # Verbatim — same dict the HF endpoint returned, before any projection.
    assert on_disk == llama_config


@pytest.mark.asyncio
@respx.mock
async def test_raw_config_bytes_are_byte_identical_not_reserialized(
    cache_dir: Path,
) -> None:
    """ADR-015 invariant #2 demands the *raw bytes* HF returned land
    on disk verbatim — not a parse → reserialize round-trip. A
    Python `json.dumps(parsed)` rewrites whitespace, normalizes key
    order, and drops formatting HF actually sent. That defeats the
    invariant's purpose: a future schema-evolution audit needs the
    exact bytes to compare against the documented schema, not our
    reserialization of them.

    This test uses a deliberately quirky source payload (custom key
    ordering + unusual indentation) and asserts the on-disk file
    matches the source bytes exactly. Round-trip dict equality would
    pass even under the broken reserialize-via-json.dumps behavior,
    so we compare strings here."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc123"

    # Quirky-but-valid source bytes: 4-space indent, keys in
    # non-alphabetic order, trailing newline. json.dumps(parsed) would
    # produce single-line compact output, normalizing all three.
    quirky_raw = (
        "{\n"
        '    "architectures": ["LlamaForCausalLM"],\n'
        '    "num_hidden_layers": 80,\n'
        '    "num_attention_heads": 64,\n'
        '    "num_key_value_heads": 8,\n'
        '    "hidden_size": 8192,\n'
        '    "max_position_embeddings": 131072,\n'
        '    "torch_dtype": "bfloat16"\n'
        "}\n"
    )

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(
            200,
            content=quirky_raw.encode(),
            headers={"content-type": "application/json"},
        )
    )

    sync = HfModelSync(cache_dir=cache_dir)
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )

    raw_cache_file = cache_dir / "huggingface" / "llama-3-3-70b.config.json"
    # Byte-for-byte match — would FAIL under reserialize-via-json.dumps
    # because that strips whitespace and reorders keys.
    assert raw_cache_file.read_text() == quirky_raw


@pytest.mark.asyncio
@respx.mock
async def test_raw_config_persists_even_when_shape_validation_fails(
    cache_dir: Path,
) -> None:
    """ADR-015 invariant #2 doesn't just apply on the happy path — it
    applies BEFORE any parse-or-validation step. A config.json missing
    `architectures` (the family discriminator) gets rejected with
    ValueError. But the raw bytes must still be on disk afterwards so
    the investigator can read what HF actually returned (maybe HF
    renamed the field, maybe the upstream payload was an HTML error
    page, maybe the network corrupted it).

    Pre-fix behavior: shape validation in `_fetch_config` raised
    BEFORE persistence, so the raw bytes vanished. Post-fix: persist
    raw bytes first, then parse + validate."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    missing_arch_payload = '{"num_hidden_layers": 80}'
    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(
            200,
            content=missing_arch_payload.encode(),
            headers={"content-type": "application/json"},
        )
    )

    sync = HfModelSync(cache_dir=cache_dir)
    with pytest.raises(ValueError, match="architectures"):
        await sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )

    # Raw STILL on disk despite the shape rejection — the investigator
    # can read exactly what HF returned without needing to re-request.
    raw_cache_file = cache_dir / "huggingface" / "llama-3-3-70b.config.json"
    assert raw_cache_file.exists()
    assert raw_cache_file.read_text() == missing_arch_payload


# --------------------------------------------------- cache hit on same revision


@pytest.mark.asyncio
@respx.mock
async def test_second_sync_same_sha_skips_config_fetch(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """The info endpoint is cheap and ALWAYS consulted (it returns the
    current revision SHA — that's how we know whether the cached
    config is still current). The expensive `raw/{sha}/config.json`
    fetch is what the cache prevents on a same-SHA second call."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc123"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    config_route = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert config_route.call_count == 1, "second sync at same SHA must skip the config.json fetch"


# ----------------------------------------------- cache invalidation on new sha


@pytest.mark.asyncio
@respx.mock
async def test_sync_refetches_when_revision_sha_changes(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """When HF reports a new SHA on the info endpoint, the cached
    config is stale — refetch the new revision."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        side_effect=[
            httpx.Response(200, json={"sha": "rev_a", "modelId": repo_id}),
            httpx.Response(200, json={"sha": "rev_b", "modelId": repo_id}),
        ]
    )
    config_a = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/rev_a/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )
    config_b = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/rev_b/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    first = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )
    second = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert first.hf_revision_sha == "rev_a"
    assert second.hf_revision_sha == "rev_b"
    assert config_a.call_count == 1
    assert config_b.call_count == 1


# -------------------------------------------------------------------- auth


@pytest.mark.asyncio
@respx.mock
async def test_passes_hf_token_when_provided(cache_dir: Path, llama_config: dict[str, Any]) -> None:
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"

    info_route = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir, hf_token="hf_test123")
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert info_route.calls.last.request.headers.get("authorization") == "Bearer hf_test123"


# --------------------------------------- retry + upstream-down fallback (ADR-013)


class TestRetryAndFallback:
    @pytest.mark.asyncio
    @respx.mock
    async def test_transient_500_then_success_on_info_recovers(
        self, fast_sync: HfModelSync, llama_config: dict[str, Any]
    ) -> None:
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        info = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(500, text="still boom"),
                httpx.Response(200, json={"sha": sha, "modelId": repo_id}),
            ]
        )
        respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        model = await fast_sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )

        assert info.call_count == 3
        assert model.slug == "llama-3-3-70b"

    @pytest.mark.asyncio
    @respx.mock
    async def test_connect_error_on_info_recovers(
        self, fast_sync: HfModelSync, llama_config: dict[str, Any]
    ) -> None:
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            side_effect=[
                httpx.ConnectError("dns hiccup"),
                httpx.Response(200, json={"sha": sha, "modelId": repo_id}),
            ]
        )
        respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        model = await fast_sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        assert model.slug == "llama-3-3-70b"

    @pytest.mark.asyncio
    @respx.mock
    async def test_persistent_5xx_with_cache_serves_cache(
        self, cache_dir: Path, fast_sync: HfModelSync, llama_config: dict[str, Any]
    ) -> None:
        """ADR-013: with a populated cache, persistent upstream failure
        falls back to the cached Model rather than raising. Trust
        envelope (M08) is the channel for surfacing staleness."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"

        # First sync: success → populates cache.
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
        )
        respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )
        await fast_sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )

        # Now upstream is down for the entire retry budget.
        respx.reset()
        fail = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(503, text="service unavailable")
        )

        model = await fast_sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        # Served from cache (no exception).
        assert model.slug == "llama-3-3-70b"
        assert model.hf_revision_sha == sha  # the cached one
        assert fail.call_count == 4  # full retry budget burned

    @pytest.mark.asyncio
    @respx.mock
    async def test_persistent_5xx_without_cache_raises_unavailable(
        self, fast_sync: HfModelSync
    ) -> None:
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        fail = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(503, text="down")
        )

        with pytest.raises(HfModelSyncUnavailable) as exc_info:
            await fast_sync.sync_model(
                repo_id=repo_id,
                slug="llama-3-3-70b",
                display_name="Llama",
                total_params_b=70.6,
                active_params_b=None,
            )

        assert "llama-3-3-70b" in str(exc_info.value) or repo_id in str(exc_info.value)
        assert fail.call_count == 4

    @pytest.mark.asyncio
    @respx.mock
    async def test_401_unauth_does_not_retry_and_does_not_fall_back(
        self, fast_sync: HfModelSync
    ) -> None:
        """A 4xx other than 429 is a client bug (bad path / missing
        token for a gated repo). Retrying just burns quota; serving
        cache would mask the real auth problem. Bubble the
        HTTPStatusError immediately."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        fail = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )

        with pytest.raises(httpx.HTTPStatusError):
            await fast_sync.sync_model(
                repo_id=repo_id,
                slug="llama-3-3-70b",
                display_name="Llama",
                total_params_b=70.6,
                active_params_b=None,
            )
        assert fail.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_429_is_retried(
        self, fast_sync: HfModelSync, llama_config: dict[str, Any]
    ) -> None:
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        info = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            side_effect=[
                httpx.Response(429, text="rate limit"),
                httpx.Response(200, json={"sha": sha, "modelId": repo_id}),
            ]
        )
        respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        model = await fast_sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        assert info.call_count == 2
        assert model.slug == "llama-3-3-70b"


# ----------------------------------------------- cache shape resilience


class TestCacheShapeResilience:
    """The cache file at `<slug>.model.json` can be corrupt or out of
    schema for several reasons (truncated write, a previous version of
    the projection, manual tampering). The cache-hit path must NEVER
    crash on these — it must transparently refetch."""

    @pytest.fixture
    def seed_corrupt_cache(self, cache_dir: Path):
        def _seed(slug: str, contents: str) -> None:
            (cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
            (cache_dir / "huggingface" / f"{slug}.model.json").write_text(contents)

        return _seed

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_dict_cached_payload_triggers_refetch(
        self, cache_dir: Path, llama_config: dict[str, Any], seed_corrupt_cache
    ) -> None:
        """`json.loads` of `[]`, `"oops"`, `42` returns truthy non-dict
        values; the previous code called `.get(...)` and raised
        AttributeError. Cache-hit path must isinstance-check first."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
        )
        config_route = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        seed_corrupt_cache("llama-3-3-70b", '"this is a string, not a dict"')

        sync = HfModelSync(cache_dir=cache_dir)
        model = await sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        assert model.slug == "llama-3-3-70b"
        assert config_route.called  # refetched

    @pytest.mark.asyncio
    @respx.mock
    async def test_validation_error_on_cached_payload_triggers_refetch(
        self, cache_dir: Path, llama_config: dict[str, Any], seed_corrupt_cache
    ) -> None:
        """A cached file with the right SHA but missing required Model
        fields (e.g. a future Model schema added a field) would crash
        with ValidationError. Cache-hit path must catch and refetch."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
        )
        config_route = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        # Dict with matching SHA but nothing else — fails Model.model_validate
        seed_corrupt_cache(
            "llama-3-3-70b",
            json.dumps({"hf_revision_sha": sha, "oops": "missing required fields"}),
        )

        sync = HfModelSync(cache_dir=cache_dir)
        model = await sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        assert model.slug == "llama-3-3-70b"
        assert config_route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_truncated_json_cached_payload_triggers_refetch(
        self, cache_dir: Path, llama_config: dict[str, Any], seed_corrupt_cache
    ) -> None:
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
        )
        config_route = respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json=llama_config)
        )

        seed_corrupt_cache("llama-3-3-70b", "{not valid json")

        sync = HfModelSync(cache_dir=cache_dir)
        model = await sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )
        assert model.slug == "llama-3-3-70b"
        assert config_route.called


class TestUpstreamShapeValidation:
    @pytest.mark.asyncio
    @respx.mock
    async def test_info_payload_with_non_string_sha_rejected(self, cache_dir: Path) -> None:
        """If HF returns `sha: 12345` (int) or `sha: null`, `str(payload['sha'])`
        previously produced `"12345"` or `"None"` and proceeded with a
        wrong URL. Validate at the boundary."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": 12345, "modelId": repo_id})
        )

        sync = HfModelSync(cache_dir=cache_dir)
        with pytest.raises(ValueError, match="sha"):
            await sync.sync_model(
                repo_id=repo_id,
                slug="llama-3-3-70b",
                display_name="Llama",
                total_params_b=70.6,
                active_params_b=None,
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_config_payload_missing_architectures_rejected(self, cache_dir: Path) -> None:
        """A config.json without `architectures` would silently route to
        family `"other"`. Surface the missing required key clearly."""
        repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        sha = "abc"
        respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
            return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
        )
        respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
            return_value=httpx.Response(200, json={"num_hidden_layers": 80})
        )

        sync = HfModelSync(cache_dir=cache_dir)
        with pytest.raises(ValueError, match="architectures"):
            await sync.sync_model(
                repo_id=repo_id,
                slug="llama-3-3-70b",
                display_name="Llama",
                total_params_b=70.6,
                active_params_b=None,
            )


# ------------------------------------ slug + repo_id validation at boundary


# ----------------------------- minimal sync_model invocation (M09 lazy-sync)


@pytest.mark.asyncio
@respx.mock
async def test_sync_model_with_only_slug_and_repo_id(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """M09's Case 1 lazy-sync (and Case 3 post-elicitation sync) calls
    `sync_model(slug=<user slug>, repo_id=<HF repo_id>)` with nothing
    else — the user hasn't told the MCP server anything about params or
    display name, and `tracked_models.yaml` doesn't (and shouldn't)
    carry the unknown model yet. The public lazy-sync primitive must
    support this minimum invocation."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(slug="llama-3-3-70b", repo_id=repo_id)

    assert model.slug == "llama-3-3-70b"
    assert model.hf_repo_id == repo_id
    # display_name derives from repo_id last segment when not supplied.
    assert model.display_name == "Llama-3.3-70B-Instruct"
    # total_params_b stays None when caller doesn't supply it — M07
    # treats null as requires_measurement per ADR-010.
    assert model.total_params_b is None
    assert model.active_params_b is None


class TestSlugValidation:
    @pytest.fixture
    def sync(self, cache_dir: Path) -> HfModelSync:
        return HfModelSync(cache_dir=cache_dir)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_slug",
        [
            "../../etc/passwd",
            "../escape",
            "/etc/cron.d/evil",
            "a/b/c",
            "with space",
            "with\nnewline",
            "",  # empty
            ".",
            "..",
            "weird?query=1",
            "UPPER_CASE",  # spec slugs are lowercase per existing CP convention
        ],
    )
    async def test_rejects_unsafe_slug(self, sync: HfModelSync, bad_slug: str) -> None:
        """Slug becomes a filename under the cache dir. Any value that
        could traverse out, contain shell-meaningful chars, or violate
        the project's lowercase-slug convention is rejected at the
        boundary — before any HTTP fetch or filesystem write."""
        with pytest.raises(ValueError, match="slug"):
            await sync.sync_model(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                slug=bad_slug,
                display_name="ignored",
                total_params_b=70.6,
                active_params_b=None,
            )


class TestRepoIdValidation:
    @pytest.fixture
    def sync(self, cache_dir: Path) -> HfModelSync:
        return HfModelSync(cache_dir=cache_dir)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_repo_id",
        [
            "../../admin",  # traversal
            "../admin",  # single dot-dot segment
            ".",  # bare dot
            "..",  # bare dot-dot
            "./foo",  # leading dot segment
            "foo/.",  # trailing dot segment
            "foo/..",  # trailing dot-dot segment
            ".../legit",  # all-dot segment
            "meta-llama/Llama/../../attacker/poisoned",  # cache-poisoning path
            "meta-llama",  # missing /
            "meta-llama/Llama-3.3-70B-Instruct?token=x",  # query injection
            "meta-llama/Llama-3.3-70B-Instruct#frag",  # fragment
            "meta-llama/Llama-3.3-70B-Instruct/extra",  # extra slash
            "/meta-llama/Llama-3.3-70B-Instruct",  # leading slash
            "@evil.com/x",  # userinfo
            "meta-llama/",  # trailing slash
            "",  # empty
            " meta-llama/Llama-3.3-70B-Instruct",  # leading space
        ],
    )
    async def test_rejects_unsafe_repo_id(self, sync: HfModelSync, bad_repo_id: str) -> None:
        """repo_id is interpolated directly into the HF URL. Anything
        that could traverse paths on huggingface.co, inject a query
        string, or change host semantics is rejected at the boundary.
        HF's documented format is `<namespace>/<name>` with each segment
        matching `[A-Za-z0-9_.-]+`."""
        with pytest.raises(ValueError, match="repo_id"):
            await sync.sync_model(
                repo_id=bad_repo_id,
                slug="some-slug",
                display_name="ignored",
                total_params_b=70.6,
                active_params_b=None,
            )


# ---------------------------- end-to-end family auto-detection via HfModelSync


@pytest.mark.asyncio
@respx.mock
async def test_sync_deepseek_yields_mla_kv_cache_strategy(
    cache_dir: Path, deepseek_config: dict[str, Any]
) -> None:
    """The MLA family auto-detection (Slice D) must flow through the
    HfModelSync code path — not just direct `Model.from_hf_config` calls.
    Without this end-to-end coverage, the production caller could
    bypass auto-detection by passing a non-None default and the test
    suite would not catch it."""
    repo_id = "deepseek-ai/DeepSeek-V3"
    sha = "abc"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=deepseek_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="deepseek-v3",
        display_name="DeepSeek-V3",
        total_params_b=671.0,
        active_params_b=37.0,
    )

    assert model.architecture_family == "deepseek_v3"
    assert model.kv_cache_strategy == "mla"


@pytest.mark.asyncio
@respx.mock
async def test_omits_authorization_when_no_token(
    cache_dir: Path, llama_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty HF_TOKEN env (CI safeguard) MUST NOT produce a malformed
    `Authorization: Bearer ` header — same contract as M02's CP client."""
    monkeypatch.setenv("HF_TOKEN", "")
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"

    info_route = respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama 3.3 70B Instruct",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert "authorization" not in {h.lower() for h in info_route.calls.last.request.headers}
