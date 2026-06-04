# ADR-006 — Benchmark cells published as Parquet on Hugging Face Datasets

**Status:** Locked. Tier 1b (`public_benchmark_anchor`) **removed
from v1** in the M10 audit. v2's M17 is the unlock for Tier 1a
(`own_measured`) via GuideLLM-measured cells — not Tier 1b.
**Date:** 2026-05 (v2.1 lock-in); revised 2026-05-31

## Decision

The benchmark-cell artifact ships as a Parquet file on Hugging Face
Datasets under CC-BY-4.0. Schema is the `BenchmarkCell` Pydantic
shape.

In v1, the seed parquet is **empty** but schema-validated and
Tier 1b (`public_benchmark_anchor`) is removed from the
estimator's tier ladder. The `BenchmarkCell` validator still
rejects `own_measured` rows at construction so v1 cannot
accidentally serve any benchmark-cell-derived throughput value.
See [`../../spec/M10-benchmark-seeds.md`](../../spec/M10-benchmark-seeds.md)
deferral preamble for the M10 audit findings.

## Context

The strategic-moat artifact of this project is the cell shape +
the dataset. v1 was scoped to seed 20–30 cells from public sources
(blog posts, vendor release notes, academic papers). The M10 audit
discovered:

- Public benchmark sources do not publish per-stream steady-state
  decode-TPS in the cell shape (`tokens_per_second` at a given
  `(batch_size, context_length)` op-point with disclosed
  methodology).
- Curated cells rot — the URL test added in M10 caught multiple
  404s within weeks of curation.
- The richest source (Kiely 2026) is methodology, not
  measurements.

v2's M17 introduces own-measured cells via GuideLLM runs, which
revives **Tier 1a (`own_measured`, confidence 0.95)** in the TPS
provenance ladder and gives the dataset a defensible methodology.
Tier 1b is not tied to M17 — reviving it would need a separate
decision once a viable public-source landscape exists.

## Consequences

- v1 ships with a zero-row parquet. Schema validation, source-URL
  reachability test, and the loader-level `public_benchmark_anchor`
  guard remain in place so the v2 unlock is a flip, not a rewrite.
- The `bench_cells` parameter on `estimate_tps` and the cost-cells
  join layer is optional in v1 (defaults to `None`/`[]`).
- `BenchmarkCell.source` field validator rejects `own_measured`
  rows at construction in v1 — v2 reverses this in a separate
  ADR update.

## Alternatives considered

- **Curate cells from blog posts only.** Tried; failed the
  source-rot test in audit.
- **Skip the artifact entirely.** Drops Tier 1b and weakens the
  TPS ladder permanently. Rejected.
- **Hand-write benchmark cells from Kiely.** The book is
  methodology, not measurements. Rejected on data-quality
  grounds.

## References

- [`../../spec/M10-benchmark-seeds.md`](../../spec/M10-benchmark-seeds.md) — M10 spec + deferral preamble.
- ADR-010 (TPS ladder this tier was meant to anchor).
