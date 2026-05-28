"""Pydantic schema for HF-synced models, plus a `from_hf_config` factory.

Per ADR-015 the full `config.json` payload is preserved verbatim in
`raw_config`. The typed fields below are the subset M06 / M07 consume
today; nested or evolving objects (`rope_scaling`, MLA-specific keys,
MoE routing tables, attention-bias flags, ...) stay inside
`raw_config` and remain queryable as new milestones add accessors.

`from_hf_config` is the single factory for every family we sync. It
auto-detects `architecture_family` from `raw_config["architectures"]`
via `detect_architecture_family()`, then derives
`kv_cache_strategy` from the family via `detect_kv_cache_strategy()`
— so DeepSeek-V3 configs become `kv_cache_strategy="mla"` while
everything else falls through to `"standard_gqa"`. MoE specifics
(Mixtral's `num_local_experts`, DeepSeek's `n_routed_experts`) don't
need a separate factory; they live in `raw_config` and the M07 KV /
throughput math reads them directly when an `mla` / MoE branch
applies. Callers can override either `architecture_family` or
`kv_cache_strategy` for fine-tunes whose `architectures` string
didn't get updated — that's the
`seeds/tracked_models.yaml::kv_cache_strategy_override` path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ArchitectureFamily = Literal[
    "llama",
    "qwen",
    "qwen3",
    "deepseek_v3",
    "mistral",
    "mixtral",
    "phi",
    "gemma",
    "gpt_oss",
    "command",
    "other",
]

KvCacheStrategy = Literal["standard_gqa", "mla", "sliding_window"]


class UnsupportedArchitectureFamily(ValueError):  # noqa: N818 (matches spec's exception name)
    """Raised when a model's `architectures[0]` doesn't match any
    family in `_FAMILY_PREFIX_MAP` (i.e. detection returns `"other"`).

    The caller — typically `HfModelSync.sync_model` — re-raises this
    after persisting the raw config.json verbatim, so `sync_all_tracked`
    can log + skip the offending model and continue with the rest of
    the catalog. Wrapping `ValueError` so existing
    `except ValueError:` branches treat unknown-family as a soft error
    rather than a crash.
    """


# Suffix mapping for `raw_config["architectures"][0]`. A startswith check
# is unambiguous for the families we track today because HF arch strings
# are descriptive enough that no prefix collides. When CP / HF introduces
# a new family, add a row here AND extend the `ArchitectureFamily` Literal.
# Order matters only for prefix collisions (e.g. "Qwen3" must come before
# "Qwen" so Qwen3MoeForCausalLM doesn't get classified as plain qwen).
_FAMILY_PREFIX_MAP: dict[str, ArchitectureFamily] = {
    "Llama": "llama",
    "Mistral": "mistral",
    "Mixtral": "mixtral",
    "Qwen3": "qwen3",
    "Qwen2": "qwen",
    "Qwen": "qwen",
    "DeepseekV3": "deepseek_v3",
    "DeepSeekV3": "deepseek_v3",
    "Phi": "phi",
    "Gemma": "gemma",
    "GptOss": "gpt_oss",
    "GPTOss": "gpt_oss",
    "Cohere": "command",
}


def detect_architecture_family(raw_config: dict[str, Any]) -> ArchitectureFamily:
    """Map an HF `config.json`'s `architectures[0]` string to one of the
    `ArchitectureFamily` Literal values.

    Returns `"other"` when the config lacks an `architectures` list, the
    list is empty, or the first entry doesn't match any known family
    prefix. The `"other"` case is what triggers M03's
    `UnsupportedArchitectureFamily` skip-with-warning behavior in later
    slices.
    """
    archs = raw_config.get("architectures") or []
    if not archs:
        return "other"
    head = str(archs[0])
    for prefix, family in _FAMILY_PREFIX_MAP.items():
        if head.startswith(prefix):
            return family
    return "other"


# Per-family KV cache strategy default. Most families use standard GQA;
# DeepSeek-V3 uses Multi-head Latent Attention (MLA) — distinct enough
# that M07's KV-cache size math has to take a different branch. New
# families with non-GQA attention get a row here AND extend the
# `KvCacheStrategy` Literal.
_DEFAULT_KV_CACHE_STRATEGY: dict[ArchitectureFamily, KvCacheStrategy] = {
    "deepseek_v3": "mla",
    # All other families fall through to "standard_gqa".
}


def detect_kv_cache_strategy(architecture_family: ArchitectureFamily) -> KvCacheStrategy:
    """Derive the KV cache strategy from the model's architecture family.

    Returns `"standard_gqa"` for every family except those that need a
    different KV-cache shape (currently just `deepseek_v3` → `"mla"`).
    Tracked-models YAML rows can override via `kv_cache_strategy_override`
    when an explicit pin is safer than auto-detection.
    """
    return _DEFAULT_KV_CACHE_STRATEGY.get(architecture_family, "standard_gqa")


class Model(BaseModel):
    """One row of the HF-synced model catalog."""

    model_config = ConfigDict(extra="ignore")

    # Identifiers
    slug: str
    hf_repo_id: str
    display_name: str

    # Parameter counts. total is memory-driving; active_params_b is
    # compute-driving for MoE (None for dense). Both Optional because
    # neither lives in HF `config.json` — they come from the model
    # card or safetensors index. tracked_models.yaml carries them for
    # project-controlled rows; M09's lazy-sync flow (Case 1 / Case 3
    # of unknown-model handling) leaves them None and M07 then routes
    # affected cells to `requires_measurement` per ADR-010.
    total_params_b: float | None = None
    active_params_b: float | None = None

    # Architecture dimensions projected for M06 / M07.
    n_layers: int
    n_attention_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    native_dtype: str
    architecture_family: ArchitectureFamily = "llama"
    kv_cache_strategy: KvCacheStrategy = "standard_gqa"

    # Raw upstream payloads.
    raw_config: dict[str, Any] = Field(default_factory=dict)
    raw_safetensors_meta: dict[str, Any] = Field(default_factory=dict)

    # Provenance
    last_synced_at: datetime
    hf_revision_sha: str

    @classmethod
    def from_hf_config(
        cls,
        *,
        slug: str,
        hf_repo_id: str,
        raw_config: dict[str, Any],
        raw_safetensors_meta: dict[str, Any],
        hf_revision_sha: str,
        last_synced_at: datetime,
        display_name: str | None = None,
        total_params_b: float | None = None,
        active_params_b: float | None = None,
        architecture_family: ArchitectureFamily | None = None,
        kv_cache_strategy: KvCacheStrategy | None = None,
    ) -> Model:
        """Build a `Model` from an HF `config.json`.

        `architecture_family` defaults to `None` and is auto-detected via
        `detect_architecture_family(raw_config)`. `kv_cache_strategy`
        defaults to `None` and is auto-derived from the detected family
        via `detect_kv_cache_strategy()` — so `deepseek_v3` configs
        produce `kv_cache_strategy="mla"` while everything else falls
        through to `"standard_gqa"`. Pass either kwarg explicitly to
        override the detection (e.g. for a fine-tune that didn't update
        its `architectures` string, or via `seeds/tracked_models.yaml`'s
        `kv_cache_strategy_override` workflow).

        Family specifics (DeepSeek MLA, Mixtral / DeepSeek MoE) need
        no separate factory — their auto-detected family +
        kv_cache_strategy plus the `raw_config` carrying MLA-specific
        keys (`kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`,
        `v_head_dim`) and MoE-specific keys (`n_routed_experts`,
        `num_experts_per_tok`, `num_local_experts`) is enough for
        M07's KV / throughput math to take the right branch.

        Required raw_config keys: `num_hidden_layers`, `num_attention_heads`,
        `num_key_value_heads`, `hidden_size`, `max_position_embeddings`,
        `torch_dtype`. `head_dim` is read if present, else derived as
        `hidden_size // num_attention_heads` (the convention older Llama
        configs follow). MLA configs (DeepSeek) keep their specialized
        keys (`kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`,
        `v_head_dim`) in `raw_config`; M07's MLA branch reads from
        there directly.
        """
        if architecture_family is None:
            architecture_family = detect_architecture_family(raw_config)
        if kv_cache_strategy is None:
            kv_cache_strategy = detect_kv_cache_strategy(architecture_family)
        # display_name falls back to the HF repo_id's last segment so
        # M09's lazy-sync flow (slug + repo_id only, no metadata) still
        # gets a human-readable label.
        if display_name is None:
            display_name = hf_repo_id.rsplit("/", 1)[-1]
        # Explicit upfront check on the required-key set, raising
        # ValueError (NOT bare KeyError) with the missing field name.
        # Reason: `HfModelSync.sync_all_tracked` catches
        # `(ValueError, ValidationError, UnsupportedArchitectureFamily,
        # ...)` for per-row skip-and-continue, but does NOT catch
        # `KeyError`. A bare `raw_config["num_attention_heads"]`
        # KeyError from a malformed upstream config would crash the
        # whole batch sync instead of letting the offending row be
        # skipped per spec/M03 § Failure modes. Listing the keys here
        # also gives the caller a clearer error than a stray KeyError
        # mid-projection.
        _required_keys = (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "max_position_embeddings",
            "torch_dtype",
        )
        missing = [key for key in _required_keys if key not in raw_config]
        if missing:
            raise ValueError(
                f"HF config for {hf_repo_id!r} (slug={slug!r}) missing required "
                f"key(s) {missing!r}; raw_config retained for inspection."
            )
        n_attention_heads = int(raw_config["num_attention_heads"])
        hidden_size = int(raw_config["hidden_size"])
        head_dim = int(raw_config.get("head_dim") or (hidden_size // n_attention_heads))

        return cls(
            slug=slug,
            hf_repo_id=hf_repo_id,
            display_name=display_name,
            total_params_b=total_params_b,
            active_params_b=active_params_b,
            n_layers=int(raw_config["num_hidden_layers"]),
            n_attention_heads=n_attention_heads,
            n_kv_heads=int(raw_config["num_key_value_heads"]),
            head_dim=head_dim,
            hidden_size=hidden_size,
            max_position_embeddings=int(raw_config["max_position_embeddings"]),
            native_dtype=str(raw_config["torch_dtype"]),
            architecture_family=architecture_family,
            kv_cache_strategy=kv_cache_strategy,
            raw_config=dict(raw_config),
            raw_safetensors_meta=dict(raw_safetensors_meta),
            hf_revision_sha=hf_revision_sha,
            last_synced_at=last_synced_at,
        )
