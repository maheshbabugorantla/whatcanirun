# Installing whatcanirun in MCP Clients

`whatcanirun` runs as a stdio MCP server (ADR-007). Every supported
client launches the binary as a subprocess and talks JSON-RPC over
stdin/stdout. The configuration block is the same shape in each
client — the file path and naming differ.

## v1 distribution: clone-install (not on PyPI yet)

v1 ships as a self-hosted clone target for power users. There is
no `uvx whatcanirun-mcp` published artifact yet — that's deferred
to v2 once the tool surface and trust-envelope shape have churned
through real usage (see
[`../spec/M12-release.md`](../spec/M12-release.md) §
"Deferred to v2"). v1 picks between two install paths:

- **Host-uv** (recommended) — clone the repo, `uv sync`, point
  the MCP client at `uv run --directory /path/to/repo
  whatcanirun-mcp`. Native cache at
  `$XDG_CACHE_HOME/whatcanirun/`. Requires Python 3.12 + `uv`
  installed on the host.
- **Docker** (fallback) — build the image locally, point the
  MCP client at `scripts/run_mcp_docker.sh`. Cache lives on a
  named docker volume. Requires `docker` running on the host.

Both paths produce identical MCP tool/resource/prompt surfaces.
Pick host-uv unless you have a reason to prefer container
isolation.

## Quickstart

### Host-uv

```bash
git clone https://github.com/maheshbabugorantla/whatcanirun
cd whatcanirun
./scripts/install_host_uv.sh
```

The install script runs `uv sync`, warms the upstream caches
via `whatcanirun-mcp prefetch`, runs the release-gate test, and
prints the MCP client config block with the absolute repo path
substituted. Re-run with `--no-prefetch --no-test` to just
re-print the config block.

### Docker

```bash
git clone https://github.com/maheshbabugorantla/whatcanirun
cd whatcanirun
docker build -t whatcanirun:latest .
# Optional: pre-warm the cache (one-time, on the named volume).
docker run --rm -i \
  -v whatcanirun-cache:/var/cache/whatcanirun \
  whatcanirun:latest prefetch
```

Then point your MCP client at `scripts/run_mcp_docker.sh`. The
script handles the `docker run -i --rm`, named-volume mount, and
env-var passthrough — clients only need the script path.

## Environment variables

All three keys are **optional**:

| Variable | Purpose | Failure mode if absent |
|---|---|---|
| `COMPUTEPRICES_API_KEY` | Lifts ComputePrices anonymous rate limits (5k/hr free with email-requested key). | Anonymous reads with lower quota; ADR-013 snapshot fallback covers rate-limit hits. |
| `HF_TOKEN` | Auth for private / gated Hugging Face configs. | Public-only reads (sufficient for every tracked model). |
| `AA_API_KEY` | Enables Artificial Analysis enrichment (ADR-003). AA *is* the provider_anchor (Tier 2) throughput source. | Server runs without AA; throughput falls back to the bandwidth heuristic (Tier 3, batch=1 only) or `requires_measurement` (Tier 4) for cells the heuristic can't anchor. |

The server itself does not source a dotenv-style file — there is no
`python-dotenv` in the install. Set the variables one of these
ways:

- **Client `env:` block** (recommended for Claude Desktop /
  Cursor / Cline — GUI clients don't reliably inherit shell env).
  See the per-client examples below.
- **Shell export** (`export COMPUTEPRICES_API_KEY=...`) if you
  launch the server from a shell whose environment you control
  (e.g. Claude Code running in a terminal, the Docker launcher
  inherits from the parent shell).
- **`direnv`** in the source checkout if you prefer a dotenv-style
  workflow — `direnv` exports into the shell, so `uv run` (or
  the docker launcher's bare `-e VAR` flags) picks the values up.

Empty strings are treated as "unset" — a deliberate
`AA_API_KEY=""` doesn't break the anonymous path.

---

## Claude Desktop

Edit `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

### Host-uv

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"],
      "env": {
        "COMPUTEPRICES_API_KEY": "your-key-or-empty",
        "AA_API_KEY": "your-key-or-empty",
        "HF_TOKEN": "your-token-or-empty"
      }
    }
  }
}
```

Substitute `/abs/path/to/whatcanirun` with the absolute path
`install_host_uv.sh` printed at the end of its run.

### Docker

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "/abs/path/to/whatcanirun/scripts/run_mcp_docker.sh",
      "env": {
        "COMPUTEPRICES_API_KEY": "your-key-or-empty",
        "AA_API_KEY": "your-key-or-empty",
        "HF_TOKEN": "your-token-or-empty"
      }
    }
  }
}
```

The script's `-e COMPUTEPRICES_API_KEY` flags inherit from the
process environment, which Claude Desktop populates from the
`env:` block.

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
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"]
    }
  }
}
```

Or the Docker variant:

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "/abs/path/to/whatcanirun/scripts/run_mcp_docker.sh"
    }
  }
}
```

Env vars inherit from the shell that launched Claude Code, so you
can keep keys out of the repo.

**Option 2 — `claude mcp add`** for user- or global-scope
configuration:

```bash
# Host-uv:
claude mcp add whatcanirun -- uv run --directory /abs/path/to/whatcanirun whatcanirun-mcp

# Docker:
claude mcp add whatcanirun -- /abs/path/to/whatcanirun/scripts/run_mcp_docker.sh
```

Verify with `claude mcp list`. Logs land in
`~/.claude/logs/mcp-whatcanirun.log` when something refuses to
start.

---

## Cursor

Edit `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per
project):

### Host-uv

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"],
      "env": {
        "COMPUTEPRICES_API_KEY": "your-key-or-empty"
      }
    }
  }
}
```

### Docker

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "/abs/path/to/whatcanirun/scripts/run_mcp_docker.sh",
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

### Host-uv

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "uv",
      "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"]
    }
  }
}
```

### Docker

```json
{
  "mcpServers": {
    "whatcanirun": {
      "command": "/abs/path/to/whatcanirun/scripts/run_mcp_docker.sh"
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

### `uv: command not found` (host-uv path)

`uv` is the dependency manager that drives the host-uv install
path. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

GUI clients (Claude Desktop, Cursor) don't always see the same
`PATH` your terminal does. If the client can find `bash` but not
`uv`, give it the absolute path:

```json
{
  "command": "/Users/you/.local/bin/uv",
  "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"]
}
```

Find the absolute path with `which uv` in your shell.

### `docker: command not found` (Docker path)

Docker Desktop ships the daemon + CLI. Confirm `docker info`
prints non-error output before re-trying. Same PATH-from-GUI
issue applies — pin the absolute path to `scripts/run_mcp_docker.sh`
in the client config (which already does — it's a script, not a
PATH lookup), but also ensure `docker` itself is on the
launching client's PATH because the script execs it.

### The stdio handshake times out

The MCP `initialize` handshake itself is fast — the server emits
its response within a few hundred milliseconds of process start
because no upstream is touched during startup. Upstream caches
(ComputePrices, Hugging Face, AA) are lazy-loaded by
`load_runtime_deps()` on the *first tool or resource call*. On a
cold cache that first call adds 1–3 seconds on a healthy network
— visible as a one-shot delay on the first invocation, not on
handshake.

Pre-warm the caches with the prefetch subcommand so the first
client call lands warm:

```bash
# Host-uv:
uv run --directory /abs/path/to/whatcanirun whatcanirun-mcp prefetch

# Docker (writes to the named volume the launcher mounts):
docker run --rm -i \
  -v whatcanirun-cache:/var/cache/whatcanirun \
  whatcanirun:latest prefetch
```

`./scripts/install_host_uv.sh` runs prefetch automatically on a
fresh install.

If the handshake genuinely fails:

1. Run the command yourself in a shell — the same `uv run ...`
   or `scripts/run_mcp_docker.sh` invocation the client uses.
   The process should print nothing to stdout (stdio is reserved
   for protocol frames) but stderr surfaces any startup error.
2. Check the client's MCP log (paths above).
3. Make sure no `print(...)` or stray stdout write snuck in — stdio
   transport is unforgiving of non-protocol bytes on stdout.

### `COMPUTEPRICES_API_KEY` env var not reaching the server

GUI clients vary in how they propagate env to subprocesses. The
explicit `env:` block in the config is the only reliable channel:

```json
"env": { "COMPUTEPRICES_API_KEY": "cp_live_..." }
```

A shell-only `export COMPUTEPRICES_API_KEY=...` will reach a
process launched from your terminal (Claude Code or
`scripts/run_mcp_docker.sh` run interactively) but typically NOT
one launched by Claude Desktop. Put the keys in the JSON.

### Stale data after a long-running session

The server caches upstream catalog and pricing reads as TTL-based
files on disk. **Restarting the client does NOT force a refetch**
— the new server process reads the same on-disk caches and finds
them still within TTL. Per-endpoint TTLs follow the upstream
refresh cadences described in
[`docs/TRUST.md`](TRUST.md#freshness-policy)
(prices refresh hourly; catalogs change rarely).

Two ways to force a fresh fetch:

1. **Wait for TTL expiry** — for pricing, that's roughly an hour
   from the cached timestamp.
2. **Delete the on-disk cache** — caches live under
   `$XDG_CACHE_HOME/whatcanirun` (defaults to
   `~/.cache/whatcanirun` on Linux/macOS) for the host-uv path,
   or on the `whatcanirun-cache` named volume
   (`/var/cache/whatcanirun` inside the container) for the
   Docker path, with per-source subdirectories:
   `computeprices/`, `artificial_analysis/`, `huggingface/`.
   Remove a single cache file (e.g.
   `computeprices/gpus.latest.json`), a source subdirectory, or
   the whole `whatcanirun/` directory; the next tool call
   repopulates from upstream.

For the Docker volume:

```bash
# Inspect:
docker run --rm -v whatcanirun-cache:/var/cache/whatcanirun \
  busybox ls -R /var/cache/whatcanirun

# Nuke:
docker volume rm whatcanirun-cache
```

There's no hot-reload tool surface in v1.

---

## Out of scope for v1

- **PyPI / `uvx whatcanirun-mcp`.** Deferred to v2 — see
  [`../spec/M12-release.md`](../spec/M12-release.md) § "Deferred
  to v2" for the rationale. v2 picks this up once the v1 surface
  has stabilized through real usage.
- **Pre-built Docker image on GHCR.** Same deferral logic — v2
  publishes a tagged image so users don't need to build locally.
- **Claude.ai web custom connectors.** Requires OAuth 2.1 + RFC
  9728 Protected Resource Metadata; currently blocked on
  Claude.ai-side bugs (issues #2157, #155). Re-evaluate in 6
  months.
- **Remote HTTP transport.** v2 work — see
  [`ADRs/ADR-007-stdio-transport.md`](ADRs/ADR-007-stdio-transport.md).
- **OAuth, bearer tokens, multi-tenant auth.** Stdio has no auth
  surface and doesn't need one (ADR-007 + ADR-012).
