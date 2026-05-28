"""Unknown architecture-family handling via `HfModelSync.sync_model`.

When HF returns a config whose `architectures[0]` doesn't match any
known prefix in `_FAMILY_PREFIX_MAP`, family auto-detects to
`"other"`. The sync_model contract per spec/M03 § Failure modes is:

  raise `UnsupportedArchitectureFamily`; log warning; raw_config IS
  still cached so the next investigator can see what HF returned.

The exception lets the upcoming `sync_all_tracked` skip the
offending model and continue with the rest of the tracked set
without aborting the entire sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.catalog.hf_model import UnsupportedArchitectureFamily
from whatcanirun.catalog.hf_sync import HfModelSync

_HF_API_BASE = "https://huggingface.co/api/models"
_HF_RAW_BASE = "https://huggingface.co"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


def _unknown_family_config() -> dict[str, Any]:
    """A config with all required fields but an `architectures` string
    that doesn't match any family in `_FAMILY_PREFIX_MAP` — drops to
    `"other"`."""
    return {
        "architectures": ["BrandNewExperimentalForCausalLM"],
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    }


@pytest.mark.asyncio
@respx.mock
async def test_unknown_family_raises_unsupported_architecture_family(
    cache_dir: Path,
) -> None:
    """sync_model on a config whose architecture doesn't match any
    family raises UnsupportedArchitectureFamily, so the caller can
    skip-with-warning instead of trusting an "other"-family Model."""
    repo_id = "vendor/BrandNewExperimental-7B"
    sha = "abc"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=_unknown_family_config())
    )

    sync = HfModelSync(cache_dir=cache_dir)
    with pytest.raises(UnsupportedArchitectureFamily) as exc_info:
        await sync.sync_model(
            repo_id=repo_id,
            slug="brandnew-7b",
            display_name="BrandNew 7B",
            total_params_b=7.0,
            active_params_b=None,
        )

    assert "BrandNewExperimentalForCausalLM" in str(exc_info.value) or (
        "brandnew-7b" in str(exc_info.value) or repo_id in str(exc_info.value)
    )


@pytest.mark.asyncio
@respx.mock
async def test_unknown_family_still_persists_raw_config(cache_dir: Path) -> None:
    """ADR-015 invariant #2: the raw HF config.json is persisted to
    disk BEFORE projection. Even on the unknown-family path where
    projection refuses, the raw bytes survive so the next
    investigator can read what HF actually returned."""
    repo_id = "vendor/BrandNewExperimental-7B"
    sha = "abc"

    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    config = _unknown_family_config()
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    with pytest.raises(UnsupportedArchitectureFamily):
        await sync.sync_model(
            repo_id=repo_id,
            slug="brandnew-7b",
            display_name="BrandNew 7B",
            total_params_b=7.0,
            active_params_b=None,
        )

    raw_path = cache_dir / "huggingface" / "brandnew-7b.config.json"
    assert raw_path.exists(), "raw config must be persisted even on unsupported-family"
    assert json.loads(raw_path.read_text()) == config


@pytest.mark.asyncio
@respx.mock
async def test_known_family_does_not_raise(cache_dir: Path) -> None:
    """Sanity: an explicitly-named family from the prefix table
    proceeds normally (Cohere → command, not unknown)."""
    repo_id = "CohereForAI/c4ai-command-r-plus"
    sha = "abc"

    cohere_config = {
        "architectures": ["CohereForCausalLM"],
        "num_hidden_layers": 64,
        "num_attention_heads": 96,
        "num_key_value_heads": 8,
        "hidden_size": 12288,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    }
    respx.get(f"{_HF_API_BASE}/{repo_id}").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"{_HF_RAW_BASE}/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=cohere_config)
    )

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="command-r-plus",
        display_name="Command R+",
        total_params_b=104.0,
        active_params_b=None,
    )

    assert model.architecture_family == "command"
