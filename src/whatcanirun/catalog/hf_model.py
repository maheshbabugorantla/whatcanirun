"""Pydantic schema for HF-synced models, plus a `from_hf_config` factory.

Per ADR-015 the full `config.json` payload is preserved verbatim in
`raw_config`. The typed fields below are the subset M06 / M07 consume
today; nested or evolving objects (`rope_scaling`, MLA-specific keys,
attention-bias flags, ...) stay inside `raw_config` and remain
queryable as new milestones add accessors.

`from_hf_config` is the standard-GQA factory. Family-specific variants
(DeepSeek MLA, Mixtral MoE, GPT-OSS MoE) land in `catalog/families/`
as separate factories that subclass or delegate to this one.
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


class Model(BaseModel):
    """One row of the HF-synced model catalog."""

    model_config = ConfigDict(extra="ignore")

    # Identifiers
    slug: str
    hf_repo_id: str
    display_name: str

    # Parameter counts. total is memory-driving; active_params_b is
    # compute-driving for MoE (None for dense).
    total_params_b: float
    active_params_b: float | None

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
        display_name: str,
        total_params_b: float,
        active_params_b: float | None,
        raw_config: dict[str, Any],
        raw_safetensors_meta: dict[str, Any],
        hf_revision_sha: str,
        last_synced_at: datetime,
        architecture_family: ArchitectureFamily | None = None,
        kv_cache_strategy: KvCacheStrategy = "standard_gqa",
    ) -> Model:
        """Build a `Model` from a standard-GQA HF `config.json`.

        `architecture_family` defaults to `None` and is auto-detected
        via `detect_architecture_family(raw_config)`. Pass an explicit
        value to override the detection — useful for fine-tunes that
        didn't update their `architectures` string, or to pin family
        via `seeds/tracked_models.yaml`'s override workflow.

        Family-specific extractors (DeepSeek MLA, Mixtral MoE) live in
        `catalog/families/` and call into this with the appropriate
        family + strategy overrides.

        Required raw_config keys: `num_hidden_layers`, `num_attention_heads`,
        `num_key_value_heads`, `hidden_size`, `max_position_embeddings`,
        `torch_dtype`. `head_dim` is read if present, else derived as
        `hidden_size // num_attention_heads` (the convention older Llama
        configs follow).
        """
        if architecture_family is None:
            architecture_family = detect_architecture_family(raw_config)
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
