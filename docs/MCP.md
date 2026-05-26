# Installing whatcanirun in MCP Clients

> **Placeholder.** M11 populates this with per-client configuration blocks.

The following clients are supported targets for v1 (stdio transport per ADR-007):

- **Claude Desktop** (macOS, Windows)
- **Claude Code** (CLI)
- **Cursor**
- **Cline** (VS Code extension)

Once `uvx whatcanirun-mcp` is installable from PyPI (M12), each client gets a config block that points at the stdio server. Drafts will land here in M11.

## Out of scope for v1

- **Claude.ai web custom connectors.** Requires OAuth 2.1 + RFC 9728 Protected Resource Metadata; currently has Claude.ai-side bugs (issues #2157, #155). Re-evaluate in 6 months.
- **Remote HTTP transport.** v2 work — see ADR-007.
