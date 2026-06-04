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

# ============ stage 1: build ============
FROM python:${PYTHON_TAG} AS build

# uv is pinned by the official installer-hosted release.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /opt/whatcanirun

# Copy lockfile + project metadata first so `uv sync` is cached
# until pyproject.toml or uv.lock changes — without this, every
# source edit invalidates the dep layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY seeds ./seeds

# --frozen pins to the committed lockfile so the published image
# matches what the maintainer tested. --no-dev skips pytest/ruff/
# mypy — the runtime image doesn't need them.
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
