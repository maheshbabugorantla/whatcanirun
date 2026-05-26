# M03 — Hugging Face Model Sync

**Status:** ⬜ Not started
**Effort:** 4h (8h realistic)
**Dependencies:** M00
**Unblocks:** M06 (fit_check needs architecture fields), M07 (tps_estimator needs total_params_b)

> Read [`SHARED.md`](SHARED.md) first. ADR-002, ADR-015 are load-bearing.

---

## Goal

Sync architectural metadata from Hugging Face `config.json` and safetensors for the 30 models we track. Persist the full raw config alongside a projection that fit_check and tps_estimator query. When a new model family ships next year with new MLA variants, the raw config already has the fields — we just add a projection accessor.

---

## Scope

### Public surface (`src/whatcanirun/catalog/hf_sync.py`)

```python
class HfModelSync:
    """Sync HF config.json + safetensors metadata for tracked models."""

    def __init__(self, cache_dir: Path, hf_token: str | None = None): ...

    async def sync_model(self, repo_id: str) -> Model: ...
    async def sync_all_tracked(self, tracked_yaml_path: Path) -> list[Model]: ...
```

### Pydantic projection (ADR-015 — `raw_config` preserves everything)

```python
class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Identifiers (stable)
    slug: str                          # joins ComputePrices llm-models
    hf_repo_id: str
    display_name: str

    # Projected fields — currently used by fit_check / tps_estimator
    total_params_b: float
    active_params_b: float | None      # MoE; None for dense
    n_layers: int
    n_attention_heads: int
    n_kv_heads: int                    # GQA-aware
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    native_dtype: str                  # "bfloat16", "float16", "float8_e4m3fn"
    architecture_family: Literal["llama", "qwen", "qwen3", "deepseek_v3",
                                 "mistral", "mixtral", "phi", "gemma",
                                 "gpt_oss", "command", "other"]
    kv_cache_strategy: Literal["standard_gqa", "mla", "sliding_window"]

    # Raw payload — config.json varies per family and gains fields constantly.
    # rope_scaling sub-objects, attention_bias flags, MLA-specific params...
    # Anything we don't project today lives here, queryable, never lost.
    raw_config: dict[str, Any]
    raw_safetensors_meta: dict[str, Any]

    # Provenance
    last_synced_at: datetime
    hf_revision_sha: str               # so we know when configs change
```

### `seeds/tracked_models.yaml` (30 rows)

Maps ComputePrices slug → HF repo_id:

```yaml
# Llama family
- slug: llama-3-3-70b
  hf_repo_id: "meta-llama/Llama-3.3-70B-Instruct"
- slug: llama-3-1-70b
  hf_repo_id: "meta-llama/Llama-3.1-70B-Instruct"
- slug: llama-3-1-8b
  hf_repo_id: "meta-llama/Llama-3.1-8B-Instruct"
- slug: llama-3-1-405b
  hf_repo_id: "meta-llama/Llama-3.1-405B-Instruct"

# Qwen family
- slug: qwen3-32b
  hf_repo_id: "Qwen/Qwen3-32B"
- slug: qwen-3-coder-30b
  hf_repo_id: "Qwen/Qwen3-Coder-30B-A3B-Instruct"
- slug: qwen3-coder-flash
  hf_repo_id: "Qwen/Qwen3-Coder-Flash"
# ... etc

# DeepSeek (MLA — special handling required)
- slug: deepseek-v3
  hf_repo_id: "deepseek-ai/DeepSeek-V3"
  kv_cache_strategy_override: mla   # config.json's architecture string is informative; explicit override is safer

- slug: deepseek-v3-1
  hf_repo_id: "deepseek-ai/DeepSeek-V3.1"
  kv_cache_strategy_override: mla

# GPT-OSS (MoE)
- slug: gpt-oss-120b
  hf_repo_id: "openai/gpt-oss-120b"
- slug: gpt-oss-20b
  hf_repo_id: "openai/gpt-oss-20b"

# Mistral / Mixtral / Phi / Gemma — additional 15 rows
```

### Family-specific extraction

Each family has its own quirk. Extraction logic in `src/whatcanirun/catalog/families/`:

| Family | Quirk | Implementation |
|---|---|---|
| **llama, qwen, qwen3, mistral, phi, gemma** | Standard GQA | `num_key_value_heads` direct from config |
| **deepseek_v3** | MLA — `kv_lora_rank` + `qk_rope_head_dim` instead of standard KV heads | Read both fields from raw_config; n_kv_heads is informational |
| **mixtral** | MoE — sparse experts | `total_params_b` from safetensors total; `active_params_b = num_experts_per_tok × per_expert_params_b` |
| **gpt_oss** | MoE — similar to Mixtral | Same approach; verify against HF model card |
| **unknown** | Raise `UnsupportedArchitectureFamily` | Log warning; M03 skips this model with `raw_config` still preserved |

### Cache layout

```
~/.cache/whatcanirun/huggingface/
├── llama-3-3-70b.config.json       # raw config from HF
├── llama-3-3-70b.safetensors.json  # safetensors metadata
├── llama-3-3-70b.model.json        # projected Model object as JSON
├── deepseek-v3.config.json
└── ...
```

### Failure modes

- HF rate-limited (429): `tenacity` retry with exponential backoff
- Repo not found (404): skip with logged warning, do NOT fail entire sync
- Unknown architecture family: skip with logged warning, raw_config still cached
- Config schema unparseable: skip with logged error, capture raw bytes for debugging

---

## Out of scope

- Auto-detecting new architecture families. New families get added to `architecture_family` Literal and `families/` module on demand.
- Downloading model weights — we read metadata only via `huggingface_hub.hf_hub_download(filename="config.json")` and the `/api/models/{repo_id}` JSON endpoint.

---

## Vertical slices

1. **Slice A: Model Pydantic with `raw_config`** — TDD: loading a real Llama config produces a populated Model; raw_config contains keys we don't project.
2. **Slice B: Llama family extractor** — TDD: Llama-3.3-70B yields `total_params_b ≈ 70.0`, `n_kv_heads=8`, `kv_cache_strategy=standard_gqa`.
3. **Slice C: HF httpx client + cache** — failing test: second sync of same model hits cache, not network.
4. **Slice D: DeepSeek MLA extractor** — TDD: DeepSeek-V3 yields `kv_cache_strategy=mla`, `active_params_b ≈ 37.0`.
5. **Slice E: Mixtral MoE extractor** — TDD: Mixtral 8x22B yields `active_params_b ≈ 39.0` (2 experts × ~20B).
6. **Slice F: Tracked models loader** — `sync_all_tracked()` over the 30-row YAML, with stub HF responses.
7. **Slice G: Unknown family handling** — TDD: passing a Cohere Command R config (architecture="cohere") logs warning, raw_config cached, Model is None.
8. **Slice H: Schema-evolution test** — failing test: a `config.json` with `experimental_new_field: 42` is loaded successfully; the field is preserved in `raw_config` and queryable.

---

## Acceptance criteria

- [ ] `sync_all_tracked()` populates 30 Model rows from real (or fixtured) HF configs.
- [ ] DeepSeek-V3 row has `kv_cache_strategy="mla"` and `active_params_b ≈ 37.0`.
- [ ] GPT-OSS-120B row has `active_params_b ≈ 5.1` (5 experts × ~1B from 128 total).
- [ ] Mixtral 8x22B row has `active_params_b ≈ 39.0` (2 active experts).
- [ ] Caching prevents re-fetch on same `hf_revision_sha`.
- [ ] Schema-evolution test (`@pytest.mark.schema_evolution`) passes — unknown config field preserved.
- [ ] Unknown family logs warning, does not crash sync of other models.
- [ ] No live HF network calls in CI (use stubbed `huggingface_hub` client + fixture configs).

---

## Common pitfalls

- **`num_key_value_heads` vs `num_attention_heads`.** Llama-3.3-70B has 64 attention heads but 8 KV heads (GQA). Don't conflate. The KV cache math uses `n_kv_heads`.
- **MoE active params confusion.** For fit_check: use `total_params_b` (memory). For tps_estimator: use `active_params_b` (compute). Different domains, different fields.
- **HF API rate limiting for anonymous tier.** ~100 requests/hour. Use `HF_TOKEN` env var if you exceed. Cache aggressively.
- **`head_dim` derivation.** Some configs omit `head_dim` and derive as `hidden_size // num_attention_heads`. Handle both cases.

---

## When done

Commit message:
> `M03: Hugging Face model sync — 30 tracked models + family-aware extraction`

Mark M03 as ✓ in `spec/INDEX.md`. Move to M04 (parallel) or M06 (critical path).
