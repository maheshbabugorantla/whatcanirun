# ADR-002 — Hugging Face Hub is canonical for model architecture

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

Hugging Face Hub is the canonical source for model architecture
fields (`config.json` + safetensors metadata) used by `fit_check`
and the bandwidth heuristic.

## Context

The fit-check math needs `n_layers`, `n_kv_heads`, `head_dim`,
`hidden_size`, `vocab_size`, `intermediate_size`, plus the
quantized weight footprint from safetensors. No pricing aggregator
publishes these. The HF Hub config files are:

- Free, unlimited for public configs (anonymous read).
- Pinned by revision SHA so we can cache deterministically.
- Authoritative — every open-weight provider publishes the same
  config.json the model author wrote.

## Consequences

- HF is fetched per `(repo_id, revision)`; the projection is cached
  to disk keyed by revision SHA. ADR-015's raw + projection pattern
  applies — raw `config.json` saved verbatim; projection narrows
  only the load-bearing fields.
- The `config.json` schema varies per family (DeepSeek-MLA adds
  `q_lora_rank`, `kv_lora_rank`; Mixtral adds `num_local_experts`).
  Nested-object fields are typed `dict[str, Any]` per ADR-015.
- The `model_architecture` confidence domain decays slowly —
  `config.json` rarely changes after release.
- `HF_TOKEN` env var is optional; only needed for private/gated
  repos. The tracked-model list is all public.

## Alternatives considered

- **TheBloke / quantized-fork forks as canonical.** Forks reorder
  metadata fields and sometimes mis-set `torch_dtype` for natively
  quantized models. Stuck with original-author repos.
- **Curated YAML.** Same issue as ADR-001 — doesn't scale to the
  long tail.

## References

- ADR-015 (raw + projection storage; nested objects typed loosely)
- ADR-001 (CP for pricing; HF for architecture is the natural
  split)
