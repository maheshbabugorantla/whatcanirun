"""Tests for the `Model` Pydantic schema and its `from_hf_config` factory.

Per ADR-015, the full HF `config.json` payload is preserved verbatim in
`raw_config`. The typed fields (`n_layers`, `n_kv_heads`, `head_dim`,
etc.) are projections of the subset M06 / M07 consume today; anything
not projected lives in `raw_config` and stays queryable.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from whatcanirun.catalog.hf_model import Model

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def llama_3_3_70b_config() -> dict[str, Any]:
    return json.loads((_FIXTURES / "hf_llama-3-3-70b_config.json").read_text())


# ----------------------------------------------------------------- raw config


class TestRawConfigPreservation:
    def test_raw_config_field_carries_full_payload(
        self, llama_3_3_70b_config: dict[str, Any]
    ) -> None:
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
        assert model.raw_config == llama_3_3_70b_config

    def test_unknown_config_field_survives(self, llama_3_3_70b_config: dict[str, Any]) -> None:
        """Future HF releases will add new config keys (rope_scaling
        variants, MLA-specific params, etc.). The model must carry
        them through unchanged."""
        future_config = {
            **llama_3_3_70b_config,
            "experimental_attention_variant": "fused_qkv_v2",
        }
        model = Model.from_hf_config(
            slug="llama-3-3-70b",
            hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
            display_name="Llama 3.3 70B Instruct",
            total_params_b=70.6,
            active_params_b=None,
            raw_config=future_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.raw_config["experimental_attention_variant"] == "fused_qkv_v2"

    def test_nested_rope_scaling_dict_preserved(self, llama_3_3_70b_config: dict[str, Any]) -> None:
        """`rope_scaling` is a nested object whose schema varies per
        model family; it must round-trip without our schema declaring
        its shape."""
        assert isinstance(llama_3_3_70b_config["rope_scaling"], dict)
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
        assert model.raw_config["rope_scaling"] == llama_3_3_70b_config["rope_scaling"]


# ------------------------------------------------------ projected field extraction


class TestProjectedFields:
    def test_llama_3_3_70b_projects_standard_gqa_fields(
        self, llama_3_3_70b_config: dict[str, Any]
    ) -> None:
        """Standard Llama config: GQA-style architecture, num_key_value_heads
        is the projected n_kv_heads."""
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
        assert model.n_layers == 80
        assert model.n_attention_heads == 64
        assert model.n_kv_heads == 8  # 64/8 = 8x GQA grouping
        assert model.head_dim == 128
        assert model.hidden_size == 8192
        assert model.max_position_embeddings == 131072
        assert model.native_dtype == "bfloat16"

    def test_head_dim_derived_when_absent(self) -> None:
        """Some older configs omit head_dim; it derives as
        hidden_size // num_attention_heads."""
        config_no_head_dim = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
            # NB: no head_dim
        }
        model = Model.from_hf_config(
            slug="some-model",
            hf_repo_id="vendor/SomeModel",
            display_name="Some Model",
            total_params_b=8.0,
            active_params_b=None,
            raw_config=config_no_head_dim,
            raw_safetensors_meta={},
            hf_revision_sha="cafef00d",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
        )
        assert model.head_dim == 128  # 4096 // 32


# ----------------------------------------------------------------- validation


class TestFromHfConfigMissingKeys:
    """`Model.from_hf_config` reads required keys via `raw_config[...]`
    direct lookup. Missing keys must raise `ValueError` (with the
    missing field name in the message), NOT bare `KeyError`. Reason:
    `HfModelSync.sync_all_tracked` catches `(ValueError,
    ValidationError, UnsupportedArchitectureFamily, ...)` for per-row
    skip-and-continue, but does NOT catch `KeyError`. A malformed
    config with `architectures` set but `num_attention_heads` missing
    would crash the whole batch sync — silently dropping every model
    that came after the bad one — instead of being skipped as a
    soft per-row failure per spec/M03 § Failure modes."""

    _BASE_KWARGS: ClassVar[dict[str, Any]] = {
        "slug": "x",
        "hf_repo_id": "vendor/x",
        "display_name": "X",
        "total_params_b": 1.0,
        "active_params_b": None,
        "raw_safetensors_meta": {},
        "hf_revision_sha": "abc",
        "last_synced_at": dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
    }

    _COMPLETE_RAW: ClassVar[dict[str, Any]] = {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    }

    @pytest.mark.parametrize(
        "missing_key",
        [
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "max_position_embeddings",
            # `torch_dtype` is intentionally NOT in this list — it
            # became optional in M07 when natively-quantized models
            # (gpt-oss MXFP4, future INT4 releases) started shipping
            # configs that omit it because the weight dtype is
            # governed by `quantization_config.quant_method`.
        ],
    )
    def test_missing_required_key_raises_value_error_naming_the_field(
        self, missing_key: str
    ) -> None:
        bad_raw = dict(self._COMPLETE_RAW)
        del bad_raw[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            Model.from_hf_config(raw_config=bad_raw, **self._BASE_KWARGS)

    def test_torch_dtype_absent_falls_back_to_quantization_method(self) -> None:
        """gpt-oss-120b and gpt-oss-20b ship MXFP4-quantized; their
        HF config.json files have NO `torch_dtype` field — instead
        carrying `quantization_config.quant_method` = 'mxfp4'.
        M07's tracked_models expansion includes both, so
        from_hf_config must handle the absence.

        Fallback order:
          1. `torch_dtype` (the historical key)
          2. `dtype` (newer HF convention for some releases)
          3. `quantization_config.quant_method` (natively-quantized)
          4. "unknown" (last resort — keeps the projection alive
             rather than refusing the row outright; M07's math
             doesn't read native_dtype anyway).
        """
        raw_no_dtype = dict(self._COMPLETE_RAW)
        del raw_no_dtype["torch_dtype"]
        raw_no_dtype["quantization_config"] = {"quant_method": "mxfp4"}
        model = Model.from_hf_config(raw_config=raw_no_dtype, **self._BASE_KWARGS)
        assert model.native_dtype == "mxfp4"

    def test_torch_dtype_absent_with_no_fallback_signal_uses_unknown(self) -> None:
        """Defense in depth: a config with neither torch_dtype nor
        a quantization hint still constructs a Model — we just
        record native_dtype='unknown' rather than refusing the
        whole row. The math layer doesn't read native_dtype, so
        this isn't a trust contract violation, but a future
        consumer that does care will see the honest 'unknown'."""
        raw_no_dtype = dict(self._COMPLETE_RAW)
        del raw_no_dtype["torch_dtype"]
        model = Model.from_hf_config(raw_config=raw_no_dtype, **self._BASE_KWARGS)
        assert model.native_dtype == "unknown"


class TestValidation:
    def test_rejects_missing_required_field(self) -> None:
        """`slug` is mandatory — Pydantic rejects construction without it."""
        with pytest.raises(ValidationError):
            Model(  # type: ignore[call-arg]
                hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
                display_name="Llama 3.3 70B Instruct",
                total_params_b=70.6,
                active_params_b=None,
                n_layers=80,
                n_attention_heads=64,
                n_kv_heads=8,
                head_dim=128,
                hidden_size=8192,
                max_position_embeddings=131072,
                native_dtype="bfloat16",
                architecture_family="llama",
                kv_cache_strategy="standard_gqa",
                raw_config={},
                raw_safetensors_meta={},
                hf_revision_sha="x",
                last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
            )

    def test_extra_field_at_top_level_is_ignored(
        self, llama_3_3_70b_config: dict[str, Any]
    ) -> None:
        """Per ADR-015, future fields we don't model yet must not break
        validation. They're dropped from the projection (they live in
        `raw_config`) rather than rejected."""
        model = Model(
            slug="llama-3-3-70b",
            hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
            display_name="Llama 3.3 70B Instruct",
            total_params_b=70.6,
            active_params_b=None,
            n_layers=80,
            n_attention_heads=64,
            n_kv_heads=8,
            head_dim=128,
            hidden_size=8192,
            max_position_embeddings=131072,
            native_dtype="bfloat16",
            architecture_family="llama",
            kv_cache_strategy="standard_gqa",
            raw_config=llama_3_3_70b_config,
            raw_safetensors_meta={},
            hf_revision_sha="deadbeef",
            last_synced_at=dt.datetime(2026, 5, 26, tzinfo=dt.UTC),
            future_top_level_field="ignored",  # type: ignore[call-arg]
        )
        assert model.slug == "llama-3-3-70b"
