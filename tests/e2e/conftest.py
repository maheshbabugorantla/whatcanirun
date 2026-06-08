"""Pytest fixtures for the Claude-in-the-loop e2e harness.

Every test here is `@pytest.mark.e2e` and skipped by default. To
run them locally:

    uv sync --extra dev --extra e2e
    # Either:
    #   - export ANTHROPIC_API_KEY=sk-ant-...  (pay-as-you-go API)
    #   - or be logged into Claude Code locally (Pro/Max plan
    #     Agent SDK credits)
    uv run pytest -m e2e

Or via the wrapper that surfaces both auth paths:

    scripts/run_e2e_scenarios.sh
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# Skip the entire `tests/e2e/` collection when the Agent SDK isn't
# installed (i.e. the `e2e` optional-deps group hasn't been synced).
# pytest still imports conftest.py during collection even when
# `-m '... and not e2e'` deselects the tests themselves, so an
# unconditional `from claude_agent_sdk import ...` would crash the
# default `pytest -q` workflow for contributors on
# `uv sync --extra dev` alone. `importorskip` at module top defers
# the import to a check that pytest treats as a skip-the-whole-
# module signal.
pytest.importorskip(
    "claude_agent_sdk",
    reason="install with `uv sync --extra e2e` to run the Claude-in-the-loop harness",
)


# Sonnet 4.6 is the chosen default — strong enough to pick the
# right tool path without making the scenario suite cost-
# prohibitive. To experiment with a different model, override via
# `WHATCANIRUN_E2E_MODEL` rather than editing this constant.
_DEFAULT_MODEL = "claude-sonnet-4-6"

# Repo root resolves to two levels up (tests/e2e/<this>).
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def claude_runtime_available() -> None:
    """Skip the e2e suite when the `claude` CLI isn't on PATH. The
    Agent SDK spawns the CLI as a subprocess; without it, every
    `query()` call would raise `CLINotFoundError`. Surface the
    setup gap loudly at collection time rather than per-test."""
    if shutil.which("claude") is None:
        pytest.skip(
            "claude CLI not on PATH — install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code/setup) "
            "to run the e2e harness"
        )


@pytest.fixture(scope="session")
def claude_model() -> str:
    """Claude model id used by the harness. Defaults to the chosen
    Sonnet driver; honours `WHATCANIRUN_E2E_MODEL` for ad-hoc
    experiments without code edits."""
    return os.environ.get("WHATCANIRUN_E2E_MODEL", _DEFAULT_MODEL)


@pytest.fixture(scope="session")
def mcp_command() -> str:
    """Subprocess command the Agent SDK uses to launch
    `whatcanirun-mcp`. Mirrors the release-gate fixture so both
    tiers spawn the server the same way."""
    return "uv"


@pytest.fixture(scope="session")
def mcp_args() -> list[str]:
    """Args to the Agent SDK's stdio launcher. The `--directory
    <repo>` form ensures the spawned `uv` uses the right project
    venv even when pytest itself was launched from elsewhere."""
    return ["run", "--directory", str(REPO_ROOT), "whatcanirun-mcp"]


@pytest.fixture(scope="session")
def mcp_env() -> dict[str, str] | None:
    """Env vars to pass through to the spawned whatcanirun-mcp
    subprocess. None when no override is needed; tests can monkey-
    patch this fixture to inject `COMPUTEPRICES_API_KEY`,
    `HF_TOKEN`, or `AA_API_KEY` for variant runs without editing
    the harness module."""
    return None
