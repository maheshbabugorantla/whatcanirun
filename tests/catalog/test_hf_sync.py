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

from whatcanirun.catalog.hf_sync import HfModelSync

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
