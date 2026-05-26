"""DeepSeek MLA extractor tests.

DeepSeek-V3 uses Multi-head Latent Attention (MLA), not the standard
GQA pattern most families share. Per ADR-015, the projection's typed
fields stay flexible while MLA-specific keys (`kv_lora_rank`,
`qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`) live in
`raw_config` for M06 / M07 to consume when they need them.

The key Slice D contract: `kv_cache_strategy` must be auto-derived
from `architecture_family` so a DeepSeek model doesn't silently
report `standard_gqa` (which would mislead M07's KV-cache size math).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from whatcanirun.catalog.hf_model import (
    Model,
    detect_kv_cache_strategy,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def deepseek_v3_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_deepseek-v3_config.json").read_text())


# ---------------------------------------------- detect_kv_cache_strategy


class TestDetectKvCacheStrategy:
    @pytest.mark.parametrize(
        "family, expected",
        [
            ("deepseek_v3", "mla"),
            ("llama", "standard_gqa"),
            ("qwen", "standard_gqa"),
            ("qwen3", "standard_gqa"),
            ("mistral", "standard_gqa"),
            ("mixtral", "standard_gqa"),
            ("phi", "standard_gqa"),
            ("gemma", "standard_gqa"),
            ("gpt_oss", "standard_gqa"),
            ("command", "standard_gqa"),
            ("other", "standard_gqa"),
        ],
    )
    def test_maps_each_family(self, family: str, expected: str) -> None:
        assert detect_kv_cache_strategy(family) == expected


# ---------------------------------------------- from_hf_config integration


class TestDeepSeekV3Projection:
    def test_auto_detects_mla_kv_cache_strategy(self, deepseek_v3_config: dict[str, Any]) -> None:
        """The DeepSeek-V3 config has `num_key_value_heads=128` which would
        let a naive standard-GQA extractor compute a 128-head KV cache.
        But DeepSeek uses MLA — the actual KV cache is the compressed
        kv_lora_rank=512 tensor. The strategy field auto-flips so M07
        doesn't multiply the standard-GQA assumption."""
        model = Model.from_hf_config(
            slug="deepseek-v3",
            hf_repo_id="deepseek-ai/DeepSeek-V3",
            display_name="DeepSeek-V3",
            total_params_b=671.0,
            active_params_b=37.0,
            raw_config=deepseek_v3_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.architecture_family == "deepseek_v3"
        assert model.kv_cache_strategy == "mla"
        assert model.active_params_b == 37.0
        assert model.total_params_b == 671.0

    def test_mla_specific_fields_preserved_in_raw_config(
        self, deepseek_v3_config: dict[str, Any]
    ) -> None:
        """The MLA-specific keys (kv_lora_rank, qk_rope_head_dim,
        qk_nope_head_dim, v_head_dim) MUST round-trip through raw_config
        so M07's MLA-aware throughput math has them available."""
        model = Model.from_hf_config(
            slug="deepseek-v3",
            hf_repo_id="deepseek-ai/DeepSeek-V3",
            display_name="DeepSeek-V3",
            total_params_b=671.0,
            active_params_b=37.0,
            raw_config=deepseek_v3_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.raw_config["kv_lora_rank"] == 512
        assert model.raw_config["qk_rope_head_dim"] == 64
        assert model.raw_config["qk_nope_head_dim"] == 128
        assert model.raw_config["v_head_dim"] == 128

    def test_moe_specific_fields_preserved_in_raw_config(
        self, deepseek_v3_config: dict[str, Any]
    ) -> None:
        """DeepSeek-V3 is also MoE — 256 routed experts, 8 active per
        token. M07 will need these for compute-bound throughput on the
        active path; they live in raw_config until then."""
        model = Model.from_hf_config(
            slug="deepseek-v3",
            hf_repo_id="deepseek-ai/DeepSeek-V3",
            display_name="DeepSeek-V3",
            total_params_b=671.0,
            active_params_b=37.0,
            raw_config=deepseek_v3_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.raw_config["n_routed_experts"] == 256
        assert model.raw_config["num_experts_per_tok"] == 8


# --------------------------------------------------- override behavior


class TestKvCacheStrategyOverride:
    def test_explicit_strategy_beats_family_default(
        self, deepseek_v3_config: dict[str, Any]
    ) -> None:
        """A user override (e.g. via tracked_models.yaml's
        kv_cache_strategy_override) wins over the family-derived default.
        Symmetric with how architecture_family overrides work."""
        model = Model.from_hf_config(
            slug="deepseek-v3-experimental",
            hf_repo_id="vendor/SomeForkOfDeepseek",
            display_name="Deepseek fork",
            total_params_b=671.0,
            active_params_b=37.0,
            raw_config=deepseek_v3_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
            kv_cache_strategy="standard_gqa",  # override even though family is deepseek_v3
        )
        assert model.architecture_family == "deepseek_v3"
        assert model.kv_cache_strategy == "standard_gqa"
