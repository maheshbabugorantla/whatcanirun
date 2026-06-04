# whatcanirun stdio MCP server (M12 Slice B2).
#
# Build:
#   docker build -t whatcanirun:latest .
#
# The image's entry point is `whatcanirun-mcp`, which speaks the
# MCP stdio protocol over stdin/stdout. Launch via
# `scripts/run_mcp_docker.sh` (which wires the cache volume and
# env-var passthrough) or directly with `docker run --rm -i
# whatcanirun:latest` if you don't need cache persistence.
#
# Why python:3.12-slim: pinned Python matches pyproject's
# requires-python = ">=3.12" so the resolver doesn't surprise us
# with a newer minor at build time. -slim trims ~700MB of OS libs
# we don't use.
#
# Why a multi-stage build: the build stage carries uv + a writable
# .venv; the runtime stage carries only Python + the installed
# venv + seeds. Final image ~150MB instead of ~400MB and no
# build toolchain in the published artifact.

ARG PYTHON_TAG=3.12-slim
ARG UV_TAG=0.11.16

# ============ stage 1: build ============
FROM python:${PYTHON_TAG} AS build

# uv pinned via the official Astral image (avoids `curl | sh` at
# build time and freezes the resolver version so the image stays
# reproducible across rebuilds). Bump UV_TAG above when refreshing.
COPY --from=ghcr.io/astral-sh/uv:${UV_TAG} /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /opt/whatcanirun

# Layer 1: deps-only sync. Copying just pyproject + lockfile (NOT
# src/seeds) means this layer is cached until either of those two
# files change. `--no-install-project` defers installing the
# whatcanirun package itself until after src is copied; that way a
# src edit doesn't invalidate the deps layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: source + project install. Now that deps are baked into
# the venv, copying src + seeds and re-running `uv sync` only
# (re-)installs the whatcanirun package itself — fast, and the
# expensive dep layer above stays cached across src edits.
COPY src ./src
COPY seeds ./seeds
RUN uv sync --frozen --no-dev


# ============ stage 2: runtime ============
FROM python:${PYTHON_TAG} AS runtime

# Cache lives at $XDG_CACHE_HOME/whatcanirun/. Set the env var so a
# user-mounted volume at /var/cache/whatcanirun lands at exactly
# the path the code reads, regardless of whether HOME is /root.
ENV XDG_CACHE_HOME=/var/cache \
    PATH="/opt/whatcanirun/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/whatcanirun

COPY --from=build /opt/whatcanirun/.venv /opt/whatcanirun/.venv
COPY --from=build /opt/whatcanirun/src /opt/whatcanirun/src
COPY --from=build /opt/whatcanirun/seeds /opt/whatcanirun/seeds
COPY --from=build /opt/whatcanirun/pyproject.toml /opt/whatcanirun/pyproject.toml

# Pre-create the cache dir so a non-root operator that maps a
# volume here doesn't hit a chown surprise on first run.
RUN mkdir -p /var/cache/whatcanirun

# stdio protocol path: container reads JSON-RPC frames from stdin,
# writes them to stdout. Run with `docker run --rm -i` so stdin
# stays attached; see scripts/run_mcp_docker.sh for the canonical
# invocation with cache + env-var wiring.
ENTRYPOINT ["whatcanirun-mcp"]
