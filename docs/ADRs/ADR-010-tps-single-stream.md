# ADR-010 — TPS heuristic restricted to single-stream (batch=1)

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

The bandwidth-based TPS heuristic
(`bandwidth_gbps / weights_bytes_per_token * 0.75`) is restricted
to single-stream (batch=1) inference. For batch > 1, the estimator
returns `requires_measurement` unless a measured benchmark cell
exists.

## Context

The bandwidth heuristic models LLM decode as memory-bound, which
holds for low-to-medium batch sizes per Kiely 2026
*Inference Engineering* §2.4.2. At batch=1 the heuristic is
within ±50% of measured anchors — good enough for the Tier 3
confidence value of 0.60.

Scaling the heuristic linearly with batch size — `peak_tps * B` —
was verified to overestimate by **~6×** at batch=128 in
compute-bound regimes. That's not a calibration issue; it's a
model-of-physics issue. Linear batch scaling is dishonest in v1.

## Consequences

- The estimator's Tier 3 branch hard-stops at batch=1; batch > 1
  paths return Tier 4 (`requires_measurement`, confidence 0.00).
- Cells with measured throughput (Tier 1a `own_measured` in v2,
  Tier 1b `public_benchmark_anchor` in v2) can carry batch > 1
  numbers — the measurement is real, the heuristic isn't.
- The `KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75` constant has a
  Kiely 2026 §2.4.2 citation in source. Reviewers can audit the
  number against the book.

## Alternatives considered

- **Linear scaling with batch.** Verified 6× wrong. Rejected.
- **Roofline-model curve fit.** v1 doesn't have the calibration
  data to commit to a curve; would be a fake number with extra
  steps.
- **No heuristic at all.** Drops 60%+ of v1 cells to Tier 4 with
  no number. Net trust loss given how decent batch=1 estimates
  are.

## References

- ADR-004 (the throughput confidence domain on the envelope
  reflects this tier's value)
- [`../TRUST.md`](../TRUST.md) — 4-tier TPS provenance ladder.
