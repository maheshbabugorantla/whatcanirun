# ADR-012 — v2 auth: email-OTP → bearer API key via Resend; no OAuth

**Status:** Locked (v2 deferred)
**Date:** 2026-05 (v2.1 lock-in)

## Decision

v2's authentication flow is email-OTP → bearer API key, delivered
via Resend (free 3k/mo). No OAuth in v1 or v2.

## Context

v2 introduces a remote HTTP transport (ADR-007) that needs to
authenticate callers. The two main MCP clients (Claude Code,
Claude Desktop) both accept bearer-token auth headers cleanly.
Email-OTP is the minimum-friction flow to issue a long-lived
bearer key per user, which is exactly what the clients want.

OAuth would unlock Claude.ai web custom connectors but the
Claude.ai-side OAuth path has open upstream bugs (claude.ai
#2157, #155) that make it unreliable, and OAuth is structurally
overkill for a two-client integration. Re-evaluate in 6 months.

## Consequences

- v2's `users` table stores `(email, hashed_api_key, quota_tier)`.
  No password column, no session column.
- Bearer keys are long-lived but revocable. UI for revocation is
  v2's smallest possible thing — a "regenerate key" button.
- Resend's free tier (3k/mo) covers the expected v2 user volume
  with room.
- No third-party identity provider; no SSO; no MFA in v2.

## Alternatives considered

- **OAuth 2.1 + RFC 9728.** Blocked on upstream Claude.ai bugs.
- **Static API keys distributed manually.** Doesn't scale beyond
  early users.
- **Magic links via email** (instead of OTP). Equivalent UX;
  OTP picked because the failure mode (typo) is recoverable in
  the same flow.

## References

- ADR-007 (stdio for v1; bearer-token HTTP for v2)
- ADR-009 (v2 stack hosts the auth flow)
