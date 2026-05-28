"""Tests for `HfModelSync.sync_all_tracked()`.

Loads `seeds/tracked_models.yaml` (+ optional user_models.yaml
extension per PR #3's contract), syncs each row via `sync_model`,
catches per-row failures so one bad model doesn't abort the whole
catalog sync.

Conflict policy on slug collisions: project seeds win. User entries
that share a slug with a seed entry are dropped with a logged
warning — asymmetric on purpose (users extend, can't redirect).
"""

from __future__ import annotations

import json
import logging
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


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.fixture
def fast_sync(cache_dir: Path) -> HfModelSync:
    return HfModelSync(
        cache_dir=cache_dir,
        retry_attempts=2,
        retry_wait_min_s=0.0,
        retry_wait_max_s=0.0,
    )


def _llama_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_llama-3-3-70b_config.json").read_text())


def _deepseek_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_deepseek-v3_config.json").read_text())


def _mixtral_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_mixtral-8x22b_config.json").read_text())


def _write_yaml(path: Path, rows: list[dict[str, Any]]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows, sort_keys=False))


def _stub_hf(repo_id: str, sha: str, config: dict[str, Any]) -> None:
    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=config)
    )


# ---------------------------------------------------------------- happy path


@pytest.mark.asyncio
@respx.mock
async def test_syncs_all_rows_in_yaml(fast_sync: HfModelSync, tmp_path: Path) -> None:
    yaml_path = tmp_path / "tracked.yaml"
    _write_yaml(
        yaml_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
            {
                "slug": "deepseek-v3",
                "hf_repo_id": "deepseek-ai/DeepSeek-V3",
                "display_name": "DeepSeek-V3",
                "total_params_b": 671.0,
                "active_params_b": 37.0,
            },
            {
                "slug": "mixtral-8x22b",
                "hf_repo_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "display_name": "Mixtral 8x22B Instruct v0.1",
                "total_params_b": 141.0,
                "active_params_b": 39.0,
            },
        ],
    )

    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())
    _stub_hf("deepseek-ai/DeepSeek-V3", "shaB", _deepseek_config())
    _stub_hf("mistralai/Mixtral-8x22B-Instruct-v0.1", "shaC", _mixtral_config())

    models = await fast_sync.sync_all_tracked(yaml_path)

    assert len(models) == 3
    by_slug = {m.slug: m for m in models}
    assert by_slug["deepseek-v3"].kv_cache_strategy == "mla"
    assert by_slug["llama-3-3-70b"].kv_cache_strategy == "standard_gqa"
    assert by_slug["mixtral-8x22b"].active_params_b == 39.0


# --------------------------------------------------------- per-row skip-on-fail


@pytest.mark.asyncio
@respx.mock
async def test_unsupported_family_row_skipped_others_continue(
    fast_sync: HfModelSync, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One bad model in the YAML does NOT abort the whole sync — the
    other rows still go through, and the bad row is logged."""
    yaml_path = tmp_path / "tracked.yaml"
    _write_yaml(
        yaml_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
            {
                "slug": "brandnew-7b",
                "hf_repo_id": "vendor/BrandNew-7B",
                "display_name": "BrandNew 7B",
                "total_params_b": 7.0,
            },
        ],
    )

    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())
    _stub_hf(
        "vendor/BrandNew-7B",
        "shaB",
        {
            "architectures": ["BrandNewForCausalLM"],  # unknown family
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
    )

    with caplog.at_level(logging.WARNING):
        models = await fast_sync.sync_all_tracked(yaml_path)

    assert len(models) == 1
    assert models[0].slug == "llama-3-3-70b"
    assert any("brandnew-7b" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_hf_404_row_skipped_others_continue(
    fast_sync: HfModelSync, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 4xx from HF (e.g. 404 deleted, 401 gated) is logged + skipped
    by sync_all_tracked. 4xx escaped the retry wrapper at the
    sync_model layer; sync_all_tracked is where we choose to be
    lenient about it (one deleted repo shouldn't abort the catalog)."""
    yaml_path = tmp_path / "tracked.yaml"
    _write_yaml(
        yaml_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
            {
                "slug": "deleted-model",
                "hf_repo_id": "vendor/DeletedModel-7B",
                "display_name": "deleted",
                "total_params_b": 7.0,
            },
        ],
    )

    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())
    respx.get(f"{_HF_API_BASE}/vendor/DeletedModel-7B").mock(
        return_value=httpx.Response(404, text="not found")
    )

    with caplog.at_level(logging.WARNING):
        models = await fast_sync.sync_all_tracked(yaml_path)

    assert len(models) == 1
    assert models[0].slug == "llama-3-3-70b"
    assert any("deleted-model" in r.message for r in caplog.records)


# --------------------------------------------------------- user file merge


@pytest.mark.asyncio
@respx.mock
async def test_bad_yaml_row_value_error_skips_not_aborts(
    fast_sync: HfModelSync, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A YAML row that passes loader validation but trips
    sync_model's slug/repo_id boundary check (e.g. via M09's
    user_models.yaml where a user supplied an unsafe value) raises
    ValueError. sync_all_tracked must skip + log, not abort the
    whole batch."""
    yaml_path = tmp_path / "tracked.yaml"
    _write_yaml(
        yaml_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
            {
                # Loader accepts this (TrackedModelRow doesn't validate
                # repo_id format), then sync_model's regex rejects it.
                "slug": "weird-slug",
                "hf_repo_id": "no-slash-here",
                "display_name": "Bad",
                "total_params_b": 1.0,
            },
        ],
    )
    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())

    with caplog.at_level(logging.WARNING):
        models = await fast_sync.sync_all_tracked(yaml_path)

    assert len(models) == 1
    assert models[0].slug == "llama-3-3-70b"
    assert any("weird-slug" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_user_file_extends_project_seeds(fast_sync: HfModelSync, tmp_path: Path) -> None:
    """User-supplied entries with slugs the project seeds don't carry
    are added to the merged set and synced."""
    seed_path = tmp_path / "tracked.yaml"
    user_path = tmp_path / "user.yaml"

    _write_yaml(
        seed_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
        ],
    )
    _write_yaml(
        user_path,
        [
            {
                "slug": "mixtral-8x22b",
                "hf_repo_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "display_name": "Mixtral",
                "total_params_b": 141.0,
                "active_params_b": 39.0,
            },
        ],
    )

    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())
    _stub_hf("mistralai/Mixtral-8x22B-Instruct-v0.1", "shaB", _mixtral_config())

    models = await fast_sync.sync_all_tracked(seed_path, user_yaml_path=user_path)

    assert {m.slug for m in models} == {"llama-3-3-70b", "mixtral-8x22b"}


@pytest.mark.asyncio
@respx.mock
async def test_seed_wins_on_slug_conflict_with_user_file(
    fast_sync: HfModelSync, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When both files declare the same slug, the project seed's
    hf_repo_id is what gets synced. User entry is dropped with a
    logged warning so an attacker who can write user_models.yaml
    can't silently redirect a tracked model."""
    seed_path = tmp_path / "tracked.yaml"
    user_path = tmp_path / "user.yaml"

    _write_yaml(
        seed_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct (project)",
                "total_params_b": 70.6,
            },
        ],
    )
    _write_yaml(
        user_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "evil/Hijacked-Model",
                "display_name": "Should not be used",
                "total_params_b": 999.0,
            },
        ],
    )

    # Only stub the SEED's repo_id. The user's evil repo_id should never
    # be called.
    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())

    with caplog.at_level(logging.WARNING):
        models = await fast_sync.sync_all_tracked(seed_path, user_yaml_path=user_path)

    assert len(models) == 1
    assert models[0].hf_repo_id == "meta-llama/Llama-3.3-70B-Instruct"
    assert models[0].total_params_b == 70.6  # the project value, not 999
    # Warning identifies BOTH the seed value (what's being used) AND
    # the user's attempted value (so an investigator can see the
    # attempted redirect, not just the safe outcome).
    relevant = [
        r.message
        for r in caplog.records
        if "llama-3-3-70b" in r.message and "project seeds win" in r.message
    ]
    assert relevant, "expected the seed-wins warning"
    joined = " ".join(relevant)
    assert "evil/Hijacked-Model" in joined, "user's attempted hf_repo_id must appear in the log"
    assert "meta-llama/Llama-3.3-70B-Instruct" in joined


@pytest.mark.asyncio
@respx.mock
async def test_missing_user_file_is_fine(fast_sync: HfModelSync, tmp_path: Path) -> None:
    seed_path = tmp_path / "tracked.yaml"
    user_path = tmp_path / "does_not_exist.yaml"

    _write_yaml(
        seed_path,
        [
            {
                "slug": "llama-3-3-70b",
                "hf_repo_id": "meta-llama/Llama-3.3-70B-Instruct",
                "display_name": "Llama 3.3 70B Instruct",
                "total_params_b": 70.6,
            },
        ],
    )
    _stub_hf("meta-llama/Llama-3.3-70B-Instruct", "shaA", _llama_config())

    models = await fast_sync.sync_all_tracked(seed_path, user_yaml_path=user_path)
    assert len(models) == 1


# ----------------------------------------- actual seeds/tracked_models.yaml


def test_repo_tracked_models_yaml_loads_cleanly() -> None:
    """The committed seeds/tracked_models.yaml at the repo root must
    parse cleanly via the loader — catches typos that would otherwise
    only surface at first sync."""
    from whatcanirun.catalog.loaders import load_tracked_models

    seed_path = _REPO_ROOT / "seeds" / "tracked_models.yaml"
    rows = load_tracked_models(seed_path)
    assert len(rows) > 0  # at least the 3 we ship with fixtures
    slugs = {row.slug for row in rows}
    # We have fixtures for these three; they MUST be present so the
    # offline test suite can sync them via respx stubs.
    assert {"llama-3-3-70b", "deepseek-v3", "mixtral-8x22b"} <= slugs
