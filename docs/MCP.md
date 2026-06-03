# Installing whatcanirun in MCP Clients

`whatcanirun` runs as a stdio MCP server (ADR-007). Every supported
client launches the binary as a subprocess and talks JSON-RPC over
stdin/stdout. The configuration block is the same shape in each
client — the file path and naming differ.

## Quickstart

Once the package is published to PyPI (M12), the canonical launch
command is:

```bash
uvx whatcanirun-mcp
```

Until M12 ships, run from a source checkout:

```bash
cd /path/to/whatcanirun
uv run whatcanirun-mcp
```

The blocks below use the post-M12 `uvx` form. Substitute
`uv --directory /path/to/whatcanirun run whatcanirun-mcp` if you're
on a source checkout.

## Environment variables

All three keys are **optional**:

| Variable | Purpose | Failure mode if absent |
|---|---|---|
| `COMPUTEPRICES_API_KEY` | Lifts ComputePrices anonymous rate limits (5k/hr free with email-requested key). | Anonymous reads with lower quota; ADR-013 snapshot fallback covers rate-limit hits. |
| `HF_TOKEN` | Auth for private / gated Hugging Face configs. | Public-only reads (sufficient for every tracked model). |
| `AA_API_KEY` | Enables Artificial Analysis enrichment (ADR-003). AA *is* the provider_anchor (Tier 2) throughput source. | Server runs without AA; throughput falls back to the bandwidth heuristic (Tier 3, batch=1 only) or `requires_measurement` (Tier 4) for cells the heuristic can't anchor. |

Set them in `.env` next to the source checkout, or pass them through
your client's `env:` block (examples below). Empty strings are
treated as "unset" — a deliberate `AA_API_KEY=""` doesn't break the
anonymous path.

---

## Claude Desktop

Edit `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uvx",
      "args": ["whatcanirun-mcp"],
      "env": {
        "COMPUTEPRICES_API_KEY": "your-key-or-empty",
        "AA_API_KEY": "your-key-or-empty",
        "HF_TOKEN": "your-token-or-empty"
      }
    }
  }
}
```

Restart Claude Desktop after editing. The server appears in the MCP
tools menu once the stdio handshake completes.

---

## Claude Code

Two options.

**Option 1 — project-scoped `.mcp.json`** (checked into the repo
that wants the server). Drop this at the repo root:

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uvx",
      "args": ["whatcanirun-mcp"]
    }
  }
}
```

Env vars inherit from the shell that launched Claude Code, so you
can keep keys out of the repo.

**Option 2 — `claude mcp add`** for user- or global-scope
configuration:

```bash
claude mcp add whatcanirun -- uvx whatcanirun-mcp
```

Verify with `claude mcp list`. Logs land in
`~/.claude/logs/mcp-whatcanirun.log` when something refuses to
start.

---

## Cursor

Edit `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per
project):

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uvx",
      "args": ["whatcanirun-mcp"],
      "env": {
        "COMPUTEPRICES_API_KEY": "your-key-or-empty"
      }
    }
  }
}
```

Reload the Cursor window after editing.

---

## Cline (VS Code extension)

Open Cline's MCP settings panel and paste:

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uvx",
      "args": ["whatcanirun-mcp"]
    }
  }
}
```

Cline writes the block to
`~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
on macOS (analogous paths on Linux/Windows). Editing the file
directly works too.

---

## Troubleshooting

### `uvx: command not found`

`uvx` ships with `uv`. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

GUI clients (Claude Desktop, Cursor) don't always see the same
`PATH` your terminal does. If the client can find `bash` but not
`uvx`, give it the absolute path:

```json
{ "command": "/Users/you/.local/bin/uvx", "args": ["whatcanirun-mcp"] }
```

Find the absolute path with `which uvx` in your shell.

### The stdio handshake times out

The server emits its initial JSON-RPC `initialize` response within
a few hundred milliseconds of process start on a warm cache. First
run hits ComputePrices once to populate the snapshot, which takes
1–3 seconds on a healthy network — within the default handshake
window for every supported client, but visible as a one-shot delay.

If the handshake genuinely fails:

1. Run the command yourself: `uvx whatcanirun-mcp`. The process
   should print nothing to stdout (stdio is reserved for protocol
   frames) but stderr surfaces any startup error.
2. Check the client's MCP log (paths above).
3. Make sure no `print(...)` or stray stdout write snuck in — stdio
   transport is unforgiving of non-protocol bytes on stdout.

### `COMPUTEPRICES_API_KEY` env var not reaching the server

GUI clients vary in how they propagate env to subprocesses. The
explicit `env:` block in the config is the only reliable channel:

```json
"env": { "COMPUTEPRICES_API_KEY": "cp_live_..." }
```

A shell-only `export COMPUTEPRICES_API_KEY=...` will reach `uvx`
launched from your terminal but typically NOT one launched by
Claude Desktop. Put the keys in the JSON.

### Stale data after a long-running session

The server caches upstream catalog and pricing reads for the
freshness window declared in
[`docs/TRUST.md`](TRUST.md#freshness-policy).
Restart the client to force a fresh fetch — there's no hot-reload
API in v1.

---

## Out of scope for v1

- **Claude.ai web custom connectors.** Requires OAuth 2.1 + RFC
  9728 Protected Resource Metadata; currently blocked on
  Claude.ai-side bugs (issues #2157, #155). Re-evaluate in 6
  months.
- **Remote HTTP transport.** v2 work — see
  [`ADRs/ADR-007-stdio-transport.md`](ADRs/ADR-007-stdio-transport.md).
- **OAuth, bearer tokens, multi-tenant auth.** Stdio has no auth
  surface and doesn't need one (ADR-007 + ADR-012).
