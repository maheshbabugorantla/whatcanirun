# ADR-011 — Observability in v2 only; v1 logs to stderr

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

v1 has no central observability surface. The stdio server logs to
stderr only. v2 introduces Logfire (free 10M spans/mo) for
structured tracing and Sentry (free 5K errors/mo) for errors.

## Context

A self-hosted v1 has no central place to *send* observability.
Adding a vendor dependency for telemetry would either require
phoning home (privacy issue) or require self-hosting another
service (infra-cost issue). Stderr to the client's MCP log is
sufficient for v1's debugging needs — every supported client
captures the subprocess stderr.

v2's hosted HTTP transport (ADR-009) has the actual observability
surface: a long-lived process with many users, where central
tracing helps diagnose distributed problems.

## Consequences

- v1 has no `logging` library boilerplate beyond `print` to
  stderr.
- Client-side logs (Claude Desktop, Claude Code) are the only v1
  trace surface; troubleshooting tips in
  [`../MCP.md`](../MCP.md) point users to the right file.
- v2 wires Logfire spans around tool entrypoints and upstream
  fetches; Sentry catches escaped exceptions.
- Both v2 services have free tiers sized for the project; no
  paid floor.

## Alternatives considered

- **OpenTelemetry exporter in v1.** Adds dependency surface for a
  feature most self-hosted users will never use.
- **File-based structured logs in v1.** Yet another disk artifact
  to manage; stderr is enough.
- **No observability in v2 either.** Once v2 is a shared service,
  flying blind on errors hurts honesty.

## References

- ADR-009 (v2 stack hosts the observability targets)
- ADR-007 (v1 is stdio; there's no long-lived process)
