# ADR-009 — v2 stack: Django + DRF + Postgres + Redis + Celery on Render

**Status:** Locked (v2 deferred)
**Date:** 2026-05 (v2.1 lock-in)

## Decision

When v2 work begins (after v1 ships and usage signal validates the
direction), the stack pivots to Django 5.x + Django REST Framework +
Postgres + Redis + Celery, deployed on Render.

## Context

v2 introduces three concerns v1 deliberately doesn't solve: a
remote HTTP transport with auth (ADR-007, ADR-012), persistence
that survives across users (a single shared cost-cells dataset),
and scheduled jobs (refreshing upstream snapshots without a user
hitting a tool first).

These three concerns map cleanly onto Django + DRF (HTTP + auth +
ORM), Postgres (persistence), Redis (cache + Celery broker), and
Celery (scheduled refresh + benchmark-cell GuideLLM runs).

## Consequences

- v2 is opt-in: anyone running v1 stdio is undisturbed by the v2
  hosted-server existence.
- v2 stands up on Render at ~$22/month minimum (Render Starter +
  Postgres Basic + cron). Verified projection; see
  [`../../spec/SHARED.md`](../../spec/SHARED.md) § Cost.
- The cost-cells layer's migration shape is open (see open
  decisions in SHARED.md): either uniform Postgres for everything,
  or DuckDB-on-files for cost cells + Postgres for auth only.

## Alternatives considered

- **FastAPI + SQLAlchemy** instead of Django. Tighter wire surface
  but loses Django admin (which v2's curation workflow benefits
  from) and DRF's permission framework.
- **Stay on the v1 stack and graft HTTP transport in.** No persistent
  store means no shared cost-cell artifact, which is the v2
  unlock.

## References

- ADR-007 (stdio for v1, HTTP for v2)
- ADR-012 (auth flow this stack hosts)
- ADR-008 (v1 stack — the thing this deliberately replaces)
