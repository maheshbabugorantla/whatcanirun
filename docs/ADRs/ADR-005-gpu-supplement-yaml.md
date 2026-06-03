# ADR-005 — GPU supplement YAML for fields CP doesn't carry

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

A small project-controlled YAML file
(`seeds/gpus_supplement.yaml`, ~12 rows) carries the GPU facts
ComputePrices does not expose: `fp8_tflops_dense` (correctly
labeled), KV-cache feature flags, form factor, MLA-vs-GQA family
markers.

## Context

ComputePrices' `/api/v1/gpus` is the GPU base catalog (ADR-001) but
its schema is missing fields the fit-check + bandwidth heuristic
need. Some fields CP exposes are mislabeled — CP's `fp8_tflops`
column reports FP16 dense values on H100/H200/B200, which would
silently halve the throughput estimates if we used it as-is.
Cross-validated against Kiely 2026 §3.2 GPU spec tables; see
[`../CP-DATA-QUALITY.md`](../CP-DATA-QUALITY.md) for the full
mismatch log.

A 12-row YAML is the smallest unit that closes the gap without
introducing schema-management overhead.

## Consequences

- Supplement YAML rows merge into the GPU catalog at load time;
  CP's row provides the base, supplement YAML overrides + adds
  fields.
- `gpu_specs` confidence domain reads 0.99 for supplement-backed
  fields (manufacturer datasheet facts don't decay) and inherits
  ComputePrices' freshness curve for CP-only fields.
- Adding a GPU SKU is a YAML edit, not a code change. Adding a
  *field* is a Pydantic-schema change (rare, batched into milestone
  work).

## Alternatives considered

- **Wait for CP to fix `fp8_tflops`.** No ETA, and several
  releases of CP have shipped without fixing it. Filed informally.
- **Use a third-party datasheet aggregator.** None cover all the
  fields we need; introducing one is more dependency surface for
  less reliability than 12 hand-curated rows.
- **Inline the fields as Python constants.** Harder to audit, harder
  to PR-review.

## References

- [`../CP-DATA-QUALITY.md`](../CP-DATA-QUALITY.md) — CP mismatches the
  YAML closes.
- ADR-001 (CP catalog is the base layer this supplements).
