# ADR-014 — DuckDB only for resource generation; tool calls use Python joins

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in); enforced by AST grep test

## Decision

The cost-cells query layer is plain Python list/dict joins for all
tool-call paths (`fit_check`, `find_cheapest_deployment`,
`compare_deployment_modes`, `budget_to_plan`). DuckDB is reserved
**exclusively** for materializing the `cost-cells://current`
resource as Parquet.

## Context

Two paths share the cost-cell join logic conceptually but have
different requirements:

- **Tool paths** need testable per-cell business logic with full
  Python-debuggable inspection (envelope construction, sufficiency
  caveats, deployment-mode-conditional confidence domains). They
  see at most a few hundred cells per call.
- **Resource path** materializes the full cost-cell table for an
  MCP client to download as Parquet. Declarative SQL is the cleanest
  way to express the materialization.

Mixing the two leads to SQL creeping into tool business logic,
which both hurts testability and re-routes confidence-domain
decisions away from Python (where the type system enforces them).

## Consequences

- A grep-based AST test (`tests/architecture/`) asserts no
  `duckdb` import in the tool-call modules under
  `src/whatcanirun/mcp_tools/` or `src/whatcanirun/plan/`.
- The Python join layer is what most contributors will edit;
  DuckDB stays in one place
  (`render_cost_cells_resource` in `plan/cost_cells.py`).
- Tool-call latency is bounded by Python iteration over in-memory
  caches; no SQL planner runtime cost.

## Alternatives considered

- **DuckDB everywhere.** Cleaner on paper; in practice opens the
  door to SQL leaking into business logic the trust envelope is
  built from.
- **No DuckDB at all; pandas-style assembly for the resource.**
  More fragile, less audited, harder to reason about for a
  declarative output.

## References

- ADR-008 (DuckDB-on-files is the v1 stack's resource-generation
  story)
- [`../../CLAUDE.md`](../../CLAUDE.md) § Invariant 5 (the trust
  contract for this split is operator-facing too)
