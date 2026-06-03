# ADR-003 — Artificial Analysis is optional enrichment

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

Artificial Analysis (https://artificialanalysis.ai) is treated as
an *optional* enrichment, never a required input. The server must
function fully without an `AA_API_KEY`.

## Context

AA publishes provider-anchored throughput data
(`median_output_tokens_per_second`) that maps cleanly onto the
`provider_anchor` tier of the TPS provenance ladder (confidence
0.70). Verified live: AA's free 1k/day tier covers 14/16 of the
project's tracked open-weight models. AA reasoning models add a
`-low` / `-medium` / `-high` effort dimension.

But: AA is a single point of failure if treated as required. The
server's promise is honest output for self-hosted users with no
accounts. So AA's role is to *improve* the throughput estimate
when its key is present, not gate functionality.

## Consequences

- The TPS estimator falls back to Tier 3
  (`bandwidth_heuristic_single_stream`, 0.60) when AA is absent.
  This is the v1 default path.
- Anonymous read attempts are not used; AA's anonymous tier was
  not committed to. The client either has a key or routes around
  the AA provider entirely.
- AA enrichment is gated behind a non-empty `AA_API_KEY` env var
  with the same anonymous-on-empty semantics as
  `COMPUTEPRICES_API_KEY` and `HF_TOKEN`.
- AA's `evaluations` nested object has 16+ fields where docs showed
  10 — typed as `dict[str, float | None]` per ADR-015.

## Alternatives considered

- **Make AA mandatory.** Breaks the self-hosted promise; AA's
  Free Tier ToS doesn't survive being a load-bearing dependency.
- **Use AA as the only throughput source.** Drops every model AA
  doesn't cover (notably Llama-3.3-70B as of May 2026).

## References

- [`../TRUST.md`](../TRUST.md) — 4-tier TPS ladder; AA = Tier 2.
- ADR-015 (raw + projection; `evaluations` typed loosely)
