# ADR-007 — v1 transport: stdio only

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

v1 ships as a stdio MCP server only. No remote HTTP transport, no
auth, no hosting. v2 will add Streamable HTTP with a bearer token
for Claude Code and Claude Desktop. Claude.ai web custom connectors
remain out of scope for both v1 and v2.

## Context

The product's value proposition is "self-hosted, free public APIs
only, one install command." Anything that requires hosting
infrastructure conflicts with that promise. Stdio gives every
supported client (Claude Desktop, Claude Code, Cursor, Cline) a
working integration with zero accounts, zero ports, zero
TLS-cert management.

Claude.ai web's custom-connector path requires OAuth 2.1 with
RFC 9728 Protected Resource Metadata. As of May 2026, two open
upstream issues (claude.ai #2157, #155) make this unreliable.
Re-evaluate in 6 months.

## Consequences

- No HTTP server in v1. No FastAPI, no uvicorn, no port
  management.
- The stdio entry point (`whatcanirun-mcp = "whatcanirun.server:main"`)
  is the single distribution surface — `uvx whatcanirun-mcp` and
  done.
- No auth code in v1. No tokens to manage, leak, or rotate.
- v2 adds a bearer-token-protected HTTP transport alongside (not
  replacing) stdio. ADR-012 covers the auth flow.
- All four supported clients use the same JSON shape; see
  [`../MCP.md`](../MCP.md).

## Alternatives considered

- **HTTP from day 1.** Mandates hosting; breaks the self-hosted
  promise.
- **stdio + Claude.ai OAuth.** Blocked on upstream bugs as
  documented.
- **gRPC.** No major MCP client supports it.

## References

- [`../MCP.md`](../MCP.md) — per-client config blocks.
- ADR-012 (auth flow planned for v2's HTTP transport).
