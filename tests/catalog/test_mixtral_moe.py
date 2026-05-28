"""Mixtral MoE extractor tests.

Mixtral uses standard GQA for attention (`num_key_value_heads=8` with
48 attention heads → 6x grouping) but sparse Mixture-of-Experts for
FFN. Mixtral 8x22B has 8 experts per layer, 2 active per token
(`num_experts_per_tok=2`). Total params ~141B; active params ~39B.

Per ADR-015 the MoE-specific keys (`num_local_experts`,
`num_experts_per_tok`, `intermediate_size`) live in `raw_config` for
M07's MoE-aware compute-bound throughput math to consume directly.
The typed `Model` carries total + active params as explicit kwargs
from the tracked-models YAML row (config.json doesn't carry them).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from whatcanirun.catalog.hf_model import Model

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def mixtral_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_mixtral-8x22b_config.json").read_text())


class TestMixtral8x22BProjection:
    def test_auto_detects_mixtral_family_and_standard_gqa(
        self, mixtral_config: dict[str, Any]
    ) -> None:
        """Mixtral's `architectures=["MixtralForCausalLM"]` flips
        architecture_family to "mixtral". kv_cache_strategy stays
        "standard_gqa" — MoE applies to FFN, not attention; KV-cache
        math is identical to GQA models."""
        model = Model.from_hf_config(
            slug="mixtral-8x22b",
            hf_repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
            display_name="Mixtral 8x22B Instruct v0.1",
            total_params_b=141.0,
            active_params_b=39.0,
            raw_config=mixtral_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.architecture_family == "mixtral"
        assert model.kv_cache_strategy == "standard_gqa"
        assert model.total_params_b == 141.0
        assert model.active_params_b == 39.0

    def test_gqa_dimensions_match_config(self, mixtral_config: dict[str, Any]) -> None:
        """Mixtral 8x22B: 56 layers, 48 attention heads, 8 KV heads
        (6x GQA grouping), hidden_size 6144, 64k context."""
        model = Model.from_hf_config(
            slug="mixtral-8x22b",
            hf_repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
            display_name="Mixtral 8x22B Instruct v0.1",
            total_params_b=141.0,
            active_params_b=39.0,
            raw_config=mixtral_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.n_layers == 56
        assert model.n_attention_heads == 48
        assert model.n_kv_heads == 8
        assert model.hidden_size == 6144
        assert model.max_position_embeddings == 65536
        assert model.native_dtype == "bfloat16"

    def test_moe_keys_preserved_in_raw_config(self, mixtral_config: dict[str, Any]) -> None:
        """M07's MoE-aware throughput math will read `num_local_experts`
        and `num_experts_per_tok` from raw_config to derive the
        compute-bound active path. These MUST round-trip."""
        model = Model.from_hf_config(
            slug="mixtral-8x22b",
            hf_repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
            display_name="Mixtral 8x22B Instruct v0.1",
            total_params_b=141.0,
            active_params_b=39.0,
            raw_config=mixtral_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.raw_config["num_local_experts"] == 8
        assert model.raw_config["num_experts_per_tok"] == 2
        assert model.raw_config["intermediate_size"] == 16384
