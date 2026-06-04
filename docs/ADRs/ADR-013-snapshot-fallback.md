# ADR-013 — Snapshot fallback when ComputePrices is unreachable

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

When ComputePrices is unreachable (network down, rate-limited,
upstream incident), the server serves the last-good local
snapshot. Tool calls never fail outright on CP failure.

## Context

ComputePrices is the single upstream pricing dependency (ADR-001).
A hard dependency on its availability would mean any CP incident
takes the whole server down — which is the opposite of the
"self-hosted, always works" promise. Mitigation: a 30-day rolling
local cache of CP responses, served with the appropriate freshness
penalty.

## Consequences

- The CP client catches network + HTTP-error class exceptions
  uniformly and routes to the cached projection. Cache lookup is
  the recovery path, not a retry loop.
- `freshness.computeprices` on the response reflects the actual
  cache age, so the `freshness` confidence domain decays exactly
  as it would for a stale-but-served CP response (see decay curve
  in [`../TRUST.md`](../TRUST.md)).
- Resource handlers (`cost-cells://current`) catch any escape
  from the runtime-deps loader and degrade to an empty-but-
  well-formed parquet rather than failing the read.
- A cold-cache, network-down combination *does* render zero rows
  — the server still responds, just with empty results and the
  appropriate caveats.

## Alternatives considered

- **Fail loudly when CP is down.** Breaks the always-works promise.
- **In-memory retry with exponential backoff.** Adds latency to
  the failure mode; doesn't add capability the snapshot fallback
  doesn't already provide.
- **Auto-fetch from a backup mirror.** No suitable backup mirror
  exists; CP is the canonical source.

## References

- ADR-001 (CP as canonical pricing source)
- ADR-015 (raw + projection storage; snapshot is the raw layer)
