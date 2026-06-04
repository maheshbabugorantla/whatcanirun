# ADR-008 — v1 stack: FastMCP + Pydantic + httpx + DuckDB-on-files

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

v1 is implemented on FastMCP 2.x (server framework), Pydantic v2
(schemas), httpx (HTTP client), and DuckDB reading from on-disk
Parquet/JSON files (cost-cell resource materialization). No
Django, no SQL database, no Redis, no Celery.

## Context

The v1 promise is stdio fast-start with zero infrastructure (see
ADR-007). DuckDB queries Parquet and JSON files directly, which
means the "database" is the seeds + cache directories on disk.
Every component is library-grade — there's no runtime to deploy,
no migration to manage, no schema to evolve in lockstep.

FastMCP is the path of least resistance for the stdio MCP server
contract. Pydantic v2 is the load-bearing schema layer that makes
ADR-015's raw + projection storage and ADR-004's TrustEnvelope
contract enforceable in code.

## Consequences

- No SQL in tool business logic. Cost-cells join is plain Python
  list joins; DuckDB is reserved for `cost-cells://current`
  resource materialization. ADR-014 splits these paths and an AST
  grep test enforces the split.
- No persistent process other than the stdio handler itself.
- Caches are on-disk files; `paths.py` centralizes locations.
- v1 development is fast because there's no infra-deploy step
  between "edit file" and "test it."

## Alternatives considered

- **Django + Postgres from day 1.** Mandates hosting (conflicts
  with ADR-007's stdio-only goal). Postponed to v2's ADR-009.
- **SQLite.** Closer to "no infra" but DuckDB's Parquet integration
  is the better fit for the resource materialization path.
- **Drop DuckDB; use pure Python for the resource path too.** The
  materialized cost-cell parquet is what makes the
  `cost-cells://current` resource possible without a fragile
  pandas-style assembly. Kept DuckDB scoped to the resource path.

## References

- ADR-007 (stdio transport drives the "no infra" goal this stack
  implements).
- ADR-014 (Python for tool calls; DuckDB for resource generation).
- ADR-009 (v2 stack — the thing this is deliberately not).
