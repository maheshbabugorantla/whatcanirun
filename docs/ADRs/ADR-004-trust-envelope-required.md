# ADR-004 — TrustEnvelope on every numerical response

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

Every tool that returns a number must wrap that number in a
`TrustEnvelope`. No exceptions. Tools that return only catalog
facts or status (`list_catalog`, `resolve_model`) are
envelope-exempt; tools that synthesize a number
(`fit_check`, `find_cheapest_deployment`,
`compare_deployment_modes`, `budget_to_plan`) are not.

## Context

The product spans personas from hobbyist to power user. Without a
structured trust contract, every persona handler would re-invent
its own "how sure are we?" annotation — or worse, omit it entirely.
A single envelope schema makes the trust contract enforceable in
code (`mypy`, schema-evolution tests) instead of a documentation
convention nobody reads.

The envelope carries `sources`, `confidence` (weakest-link),
`confidence_breakdown` per domain, `assumptions`, `caveats`,
`freshness`, and `verify_links`. Full schema in
[`../TRUST.md`](../TRUST.md).

## Consequences

- Every numerical builder in `src/whatcanirun/trust/builders.py`
  constructs an envelope; the tool can't skip it.
- The `confidence` rollup is `min(confidence_breakdown.values())`
  by code, not by convention. ADR has teeth.
- Tools that take a workload assumption surface
  `workload_assumption` in the breakdown; tools that don't, omit
  the key. Code-enforced.
- The LLM client decides how much of the envelope to relay to the
  user — the server never decides what to hide.

## Alternatives considered

- **Free-text confidence string.** Bluff-tolerant; rejected.
- **Confidence as a single float.** Loses per-domain visibility
  the weakest-link rollup needs to be useful.
- **Envelope only on `budget_to_plan`.** Other tools would re-invent
  the wheel inconsistently.

## References

- [`../TRUST.md`](../TRUST.md) — full envelope contract.
- ADR-010 (throughput tier semantics flow through the envelope).
- ADR-013 (snapshot fallback surfaces stale-data caveats here).
