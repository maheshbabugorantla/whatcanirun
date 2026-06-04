#!/usr/bin/env bash
# Docker stdio MCP server launcher for whatcanirun (M12 Slice B2).
#
# Wraps the canonical `docker run` invocation so the MCP client
# config block can be a single-line `command` pointing at this
# script. Without the wrapper, users would have to spell out the
# cache-volume mount and the env-var passthrough flags in every
# client's JSON, which is fragile.
#
# Prereqs:
#   1. Built image: `docker build -t whatcanirun:latest .`
#      from a fresh clone of the repo (Dockerfile at the root).
#   2. Optional: pre-warm the cache by running prefetch interactively:
#        docker run --rm -i -v whatcanirun-cache:/var/cache/whatcanirun \
#          whatcanirun:latest prefetch
#      The first MCP `tools/call` will warm it lazily otherwise.
#
# MCP client config block (paste into Claude Desktop /
# Claude Code / Cursor / Cline):
#
#   "mcpServers": {
#     "whatcanirun": {
#       "command": "/abs/path/to/scripts/run_mcp_docker.sh"
#     }
#   }
#
# Env vars (COMPUTEPRICES_API_KEY, HF_TOKEN, AA_API_KEY) are
# inherited from the launching client process (clients vary in
# how they propagate env). To pin them, add an `env:` block in
# the client config or set them before launching the script.

set -euo pipefail

IMAGE="${WHATCANIRUN_IMAGE:-whatcanirun:latest}"
VOLUME="${WHATCANIRUN_CACHE_VOLUME:-whatcanirun-cache}"

# stdio protocol needs stdin attached (-i). --rm cleans up the
# container after the client disconnects. Cache volume persists
# CP + HF + AA caches between runs so the cold-cache delay is a
# one-time event, not a per-launch event.
#
# Env vars: passing -e VAR (without =VALUE) tells docker to
# inherit from the launching shell rather than hard-coding a
# value. Empty / unset env vars stay empty / unset inside the
# container — the server treats empty COMPUTEPRICES_API_KEY etc
# as "anonymous tier" rather than malformed-token, matching M02
# and M03's empty-string-is-unset semantics.
exec docker run --rm -i \
    -v "${VOLUME}:/var/cache/whatcanirun" \
    -e COMPUTEPRICES_API_KEY \
    -e HF_TOKEN \
    -e AA_API_KEY \
    "${IMAGE}" "$@"
