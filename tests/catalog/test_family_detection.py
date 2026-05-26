"""Tests for HF architecture-family auto-detection.

`raw_config["architectures"]` is the conventional discriminator in HF
configs (e.g. `["LlamaForCausalLM"]`, `["Qwen3ForCausalLM"]`,
`["DeepseekV3ForCausalLM"]`). `detect_architecture_family` maps these
strings to the `ArchitectureFamily` Literal so `from_hf_config`
doesn't require callers to know the family ahead of time.

Explicit overrides (passed via `from_hf_config(architecture_family=...)`)
beat auto-detection so `seeds/tracked_models.yaml` can pin family
choices when the architecture string is ambiguous (e.g. a fine-tune
that didn't update its config).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from whatcanirun.catalog.hf_model import Model, detect_architecture_family

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def llama_3_3_70b_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_llama-3-3-70b_config.json").read_text())


# ----------------------------------------------------- detect_architecture_family


class TestDetectArchitectureFamily:
    @pytest.mark.parametrize(
        "arch_string, expected",
        [
            ("LlamaForCausalLM", "llama"),
            ("MistralForCausalLM", "mistral"),
            ("MixtralForCausalLM", "mixtral"),
            ("Qwen2ForCausalLM", "qwen"),
            ("Qwen3ForCausalLM", "qwen3"),
            ("Qwen3MoeForCausalLM", "qwen3"),
            ("DeepseekV3ForCausalLM", "deepseek_v3"),
            ("Phi3ForCausalLM", "phi"),
            ("Phi4ForCausalLM", "phi"),
            ("GemmaForCausalLM", "gemma"),
            ("Gemma2ForCausalLM", "gemma"),
            ("Gemma3ForCausalLM", "gemma"),
            ("GptOssForCausalLM", "gpt_oss"),
            ("CohereForCausalLM", "command"),
        ],
    )
    def test_known_architectures_map_to_expected_family(
        self, arch_string: str, expected: str
    ) -> None:
        assert detect_architecture_family({"architectures": [arch_string]}) == expected

    def test_unknown_architecture_falls_back_to_other(self) -> None:
        assert detect_architecture_family({"architectures": ["NovelXyzForCausalLM"]}) == "other"

    def test_missing_architectures_key_falls_back_to_other(self) -> None:
        assert detect_architecture_family({}) == "other"

    def test_empty_architectures_list_falls_back_to_other(self) -> None:
        assert detect_architecture_family({"architectures": []}) == "other"


# ----------------------------------------------- from_hf_config wires detection


class TestFromHfConfigAutoDetects:
    def test_llama_3_3_70b_auto_detects_llama_family(
        self, llama_3_3_70b_config: dict[str, Any]
    ) -> None:
        """Caller doesn't pass architecture_family; the factory reads it
        from raw_config["architectures"][0]."""
        model = Model.from_hf_config(
            slug="llama-3-3-70b",
            hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
            display_name="Llama 3.3 70B Instruct",
            total_params_b=70.6,
            active_params_b=None,
            raw_config=llama_3_3_70b_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.architecture_family == "llama"

    def test_explicit_override_beats_detection(self, llama_3_3_70b_config: dict[str, Any]) -> None:
        """A fine-tune that didn't update its config.json may carry the
        wrong `architectures` string; the seeds yaml override pins
        the family explicitly."""
        model = Model.from_hf_config(
            slug="some-fine-tune",
            hf_repo_id="vendor/SomeFineTune",
            display_name="Some Fine Tune",
            total_params_b=8.0,
            active_params_b=None,
            raw_config=llama_3_3_70b_config,  # says "LlamaForCausalLM"
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
            architecture_family="qwen3",  # explicit override
        )
        assert model.architecture_family == "qwen3"

    def test_unknown_architecture_yields_other_family(self) -> None:
        config = {
            "architectures": ["SomeBrandNewForCausalLM"],
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        }
        model = Model.from_hf_config(
            slug="new-model",
            hf_repo_id="vendor/NewModel",
            display_name="New Model",
            total_params_b=8.0,
            active_params_b=None,
            raw_config=config,
            raw_safetensors_meta={},
            hf_revision_sha="cafe",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.architecture_family == "other"
