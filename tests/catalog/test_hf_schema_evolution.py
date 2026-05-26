"""End-to-end schema-evolution tests for HF model sync (M03).

ADR-015: upstream-data clients must tolerate new fields without
breaking validation. The CI workflow's dedicated `schema-evolution`
job collects tests carrying `@pytest.mark.schema_evolution`.

These tests inject synthetic future fields at two depths and assert
they survive the full HfModelSync -> projection -> cache path:

  - new top-level field on the HF config.json
  - new nested key inside the evolving `rope_scaling` blob

Both paths must surface the new value via `model.raw_config` so a
later code change can project it without re-deploying. Also covers
the cache round-trip: the unknown field must still be present after
a sync + cache-hit re-read.
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


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.fixture(scope="module")
def llama_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_llama-3-3-70b_config.json").read_text())


def _stub_hf(repo_id: str, sha: str, config: dict[str, Any]) -> None:
    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=config)
    )


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_top_level_config_field_survives_end_to_end(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """A future HF release adds a new top-level field on config.json.
    The sync returns successfully and the unknown field is queryable
    through `model.raw_config`."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    future_config = {**llama_config, "experimental_attention_variant": "fused_qkv_v2"}
    _stub_hf(repo_id, sha, future_config)

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert model.raw_config["experimental_attention_variant"] == "fused_qkv_v2"


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_nested_rope_scaling_key_survives_end_to_end(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """The `rope_scaling` blob is the canonical evolving-nested-object
    case in HF configs; new keys land regularly (Llama-3.x added
    `original_max_position_embeddings`, `low_freq_factor`,
    `high_freq_factor`). A novel nested key must survive end-to-end
    without us declaring it in the projection."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    future_config = {
        **llama_config,
        "rope_scaling": {
            **llama_config["rope_scaling"],
            "experimental_rope_variant": "yarn_v3",
        },
    }
    _stub_hf(repo_id, sha, future_config)

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert model.raw_config["rope_scaling"]["experimental_rope_variant"] == "yarn_v3"


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_field_survives_cache_round_trip(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """A future-field response must round-trip through both the raw
    config cache (`<slug>.config.json`) AND the projected Model cache
    (`<slug>.model.json`) without losing the new field on the second
    (cached) call. The cache-hit path returns the cached Model, which
    `Model.model_validate` re-hydrates from disk — so `raw_config`
    must persist as a Pydantic field, not be reconstructed at sync
    time only."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    future_config = {**llama_config, "discount_window_hours": 4}
    _stub_hf(repo_id, sha, future_config)

    sync = HfModelSync(cache_dir=cache_dir)
    first = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )
    second = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )

    assert first.raw_config["discount_window_hours"] == 4
    assert second.raw_config["discount_window_hours"] == 4

    # Raw cache file also carries the unknown field byte-for-byte.
    raw_path = cache_dir / "huggingface" / "llama-3-3-70b.config.json"
    on_disk = json.loads(raw_path.read_text())
    assert on_disk["discount_window_hours"] == 4
