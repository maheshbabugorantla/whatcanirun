# ADR-001 — ComputePrices is the canonical pricing + GPU catalog source

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

ComputePrices (https://www.computeprices.com) is the canonical
source for GPU $/hr rental pricing, LLM API $/M-token pricing,
and the GPU base catalog (VRAM, memory bandwidth, fp16_tflops).

## Context

Self-hosted v1 needs one upstream that covers both GPU rentals
*and* hosted LLM APIs in a single uniform schema. The two endpoints
serving the project are:

- `/api/v1/gpus` — 66 GPUs with bandwidth, VRAM, architecture
- `/api/v1/gpu-prices` — 71 providers' on-demand $/hr
- `/api/v1/llm-models` — 214 hosted models with pricing tiers

Verified live in May 2026. Hourly refresh, free 5k/hr anonymous,
higher tier with an email-requested key.

## Consequences

- A single client (`whatcanirun.pricing.computeprices`) is enough
  to source pricing for both deployment modes.
- The hourly refresh cadence drives the `pricing` and `freshness`
  confidence domains (see [`../TRUST.md`](../TRUST.md)).
- Gaps in CP's schema must be supplemented elsewhere: `fp8_tflops`,
  KV-cache flags, form factor, MLA-vs-GQA family. See
  [ADR-005](ADR-005-gpu-supplement-yaml.md).
- CP data-quality issues we have to work around (CP reports FP16
  dense values under `fp8_tflops` on several SKUs; B300 missing
  entirely) are documented in
  [`../CP-DATA-QUALITY.md`](../CP-DATA-QUALITY.md).

## Alternatives considered

- **vast.ai + RunPod + Lambda APIs separately.** Three clients,
  three rate limits, three license terms. Rejected as integration
  cost.
- **Hand-curated YAML.** Doesn't scale and goes stale silently.
- **Scrape provider pages directly.** ToS risk, brittle, and CP
  already does this work and exposes it under stable terms.

## References

- ADR-013 (snapshot fallback when CP unreachable)
- ADR-015 (raw + projection storage pattern applies to CP responses)
