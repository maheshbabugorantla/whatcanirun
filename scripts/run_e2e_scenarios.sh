#!/usr/bin/env bash
# Claude-in-the-loop e2e scenario harness runner.
#
# Drives the 8 scenarios in docs/SCENARIOS.md through a real
# Anthropic Claude loop against the local MCP server. Asserts on
# the tool path Claude chose, the trust-envelope shape it saw, and
# a soft keyword match in its final natural-language reply.
#
# Required:
#   ANTHROPIC_API_KEY            Anthropic API key. Each scenario
#                                spends ~$0.05-$0.15 in tokens; the
#                                full 8-scenario suite is under $2
#                                per run on Sonnet.
#
# Optional:
#   WHATCANIRUN_E2E_MODEL        Anthropic model id to drive the
#                                loop. Defaults to `claude-sonnet-4-6`.
#                                Override only for ad-hoc experiments;
#                                keep the canonical default in
#                                tests/e2e/conftest.py so reruns
#                                are reproducible.
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./scripts/run_e2e_scenarios.sh
#
# Or run pytest directly once the extras + key are set up:
#   uv sync --extra dev --extra e2e
#   uv run pytest -m e2e
#
# Why a wrapper exists: the harness is opt-in (key + extra are
# both gates). A user running `pytest -q` blind would get silent
# "9 deselected" and miss the existence of the suite entirely;
# scripts/run_e2e_scenarios.sh is the one-line discoverable
# entry point that fails LOUDLY when the gates aren't met.

set -euo pipefail

# Run from the repo root regardless of where the script was
# invoked from. Same convention as scripts/install_host_uv.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "pyproject.toml" ]] || ! grep -q '^name = "whatcanirun"' pyproject.toml; then
    echo "run_e2e_scenarios.sh: not inside the whatcanirun repo (no whatcanirun pyproject.toml at $REPO_ROOT)" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
run_e2e_scenarios.sh: `uv` is not on PATH.

Install it once with:
    curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run this script.
EOF
    exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
run_e2e_scenarios.sh: ANTHROPIC_API_KEY is not set.

The Claude-in-the-loop scenario harness drives real Anthropic
API calls (~$0.05-$0.15 per scenario; under $2 for the full
8-scenario suite on Sonnet). Set the key and re-run:

    export ANTHROPIC_API_KEY=sk-ant-...
    ./scripts/run_e2e_scenarios.sh

To run the SERVER-side gate without spending tokens, run the
release gate instead:

    uv run pytest -m release
EOF
    exit 1
fi

# Make sure both extras are installed. `uv sync --extra dev
# --extra e2e` is idempotent — if everything's already in place
# this is a no-op; if a clone is fresh or the e2e extra was
# never installed, it adds the anthropic SDK + its transitive
# deps under the new dependency group.
echo "==> uv sync --extra dev --extra e2e (anthropic SDK + dev test deps)"
uv sync --extra dev --extra e2e

echo "==> pytest -m e2e (Claude-in-the-loop, ${WHATCANIRUN_E2E_MODEL:-claude-sonnet-4-6})"
# `-q` matches the rest of the repo's pytest invocations; the
# per-scenario assertion failures stay loud enough on -q to
# triage relay regressions without extra verbosity.
uv run pytest -m e2e -q
