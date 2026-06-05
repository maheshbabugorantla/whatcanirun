"""Pytest fixtures for the Claude-in-the-loop e2e harness.

Every test here is `@pytest.mark.e2e` and skipped by default. To
run them locally:

    uv sync --extra dev --extra e2e
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run pytest -m e2e

Or via the wrapper that checks the key + extra are present:

    scripts/run_e2e_scenarios.sh
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

# Skip the entire `tests/e2e/` collection when the anthropic SDK is
# not installed (i.e. the `e2e` optional-deps group hasn't been
# synced). pytest still imports conftest.py during collection even
# when `-m '... and not e2e'` deselects the tests themselves, so an
# unconditional `from anthropic import ...` would crash the default
# `pytest -q` workflow for contributors on `uv sync --extra dev`
# alone. `importorskip` at module top defers the import to a check
# that pytest treats as a skip-the-whole-module signal.
pytest.importorskip(
    "anthropic",
    reason="install with `uv sync --extra e2e` to run the Claude-in-the-loop harness",
)

import pytest_asyncio
from anthropic import AsyncAnthropic
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# Sonnet 4.6 was the chosen driver per the design conversation. It's
# strong enough to pick the right tool path without making the
# scenario suite cost-prohibitive. To experiment with a different
# model, override via `WHATCANIRUN_E2E_MODEL` rather than editing
# this constant — keeps the canonical default reproducible.
_DEFAULT_MODEL = "claude-sonnet-4-6"

# Repo root resolves to two levels up (tests/e2e/<this>/.../workspace).
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Skip the whole e2e suite when the key isn't set. Per the
    `e2e` marker description in pyproject.toml, these tests cost
    real Anthropic API tokens; running them silently is the wrong
    default for any unattended environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set — e2e scenarios skipped")
    return key


@pytest.fixture(scope="session")
def claude_model() -> str:
    """Anthropic model id used by the harness. Defaults to the
    chosen Sonnet driver; honours `WHATCANIRUN_E2E_MODEL` for ad-
    hoc experiments without code edits."""
    return os.environ.get("WHATCANIRUN_E2E_MODEL", _DEFAULT_MODEL)


@pytest_asyncio.fixture
async def anthropic_client(anthropic_api_key: str) -> AsyncIterator[AsyncAnthropic]:
    """One Anthropic async client per test. Async-context closes the
    underlying httpx session on teardown so we don't leak sockets
    across the suite."""
    client = AsyncAnthropic(api_key=anthropic_api_key)
    try:
        yield client
    finally:
        await client.close()


def _build_stdio_transport() -> StdioTransport:
    """Spawn `uv run --directory <repo> whatcanirun-mcp` as a
    subprocess. Same shape as the release-gate test, which proved
    function-scoping avoids the "session closed unexpectedly"
    edge case module-scoping triggered."""
    return StdioTransport(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), "whatcanirun-mcp"],
    )


@pytest_asyncio.fixture
async def mcp_client() -> AsyncIterator[Client[Any]]:
    """Function-scoped FastMCP `Client` over a fresh stdio
    subprocess. Each scenario gets its own subprocess for isolation
    — the cold-cache penalty hits once per upstream per test run,
    since the on-disk cache is shared across subprocesses."""
    transport = _build_stdio_transport()
    async with Client(transport) as client:
        yield client
