#!/usr/bin/env bash
# Claude-in-the-loop e2e scenario harness runner (Agent SDK driver).
#
# Drives the 8 scenarios in docs/SCENARIOS.md through a real
# Claude Agent SDK loop against the local MCP server. Asserts on
# the tool path Claude chose, the trust-envelope shape it saw, and
# a soft keyword match in its final natural-language reply.
#
# Two valid auth paths — pick whichever your Anthropic account
# is set up for. Both work with the same env var:
#
#   ANTHROPIC_API_KEY            Anthropic API key from
#                                console.anthropic.com. Each
#                                scenario consumes ~$0.05-$0.15
#                                in tokens; the full 8-scenario
#                                suite is under $2 per run on
#                                Sonnet. Billed pay-as-you-go
#                                against your API balance.
#
#   (or no env var) + Claude     If you're on Pro ($20/mo) or
#   Code logged in locally       Max ($200/mo) and Claude Code is
#                                authenticated on this host, the
#                                Agent SDK picks up the
#                                subscription session via the
#                                spawned `claude` CLI. Runs deduct
#                                from the Agent SDK credit pool
#                                (post-2026-06-15 billing split)
#                                instead of pay-as-you-go.
#
# Optional:
#   WHATCANIRUN_E2E_MODEL        Claude model id to drive the
#                                loop. Defaults to `claude-sonnet-4-6`.
#                                Override only for ad-hoc experiments;
#                                keep the canonical default in
#                                tests/e2e/conftest.py so reruns
#                                are reproducible.
#
# Usage:
#   # Either set the API key:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   # ... or just ensure `claude` (Claude Code CLI) is logged in:
#   claude /status
#   ./scripts/run_e2e_scenarios.sh
#
# Or run pytest directly once the extras + auth are set up:
#   uv sync --extra dev --extra e2e
#   uv run pytest -m e2e

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

# The Claude Agent SDK spawns the `claude` CLI as the
# orchestrator subprocess; without it on PATH the harness can't
# run regardless of which auth path the user picked.
if ! command -v claude >/dev/null 2>&1; then
    cat >&2 <<'EOF'
run_e2e_scenarios.sh: `claude` CLI not on PATH.

The Claude Agent SDK spawns the `claude` binary as its
orchestrator subprocess. Install Claude Code:
    https://docs.claude.com/en/docs/claude-code/setup

Then re-run this script.
EOF
    exit 1
fi

# Auth gate: accept EITHER an explicit ANTHROPIC_API_KEY OR a
# logged-in Claude Code session. We can't directly verify
# subscription auth from a shell — `claude /status` would, but
# parsing its output is brittle. Instead, surface both paths in
# the help text and trust the user.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
run_e2e_scenarios.sh: ANTHROPIC_API_KEY is not set.

The Claude Agent SDK supports two auth paths:

  (a) Set an explicit API key (pay-as-you-go API balance):

        export ANTHROPIC_API_KEY=sk-ant-...
        ./scripts/run_e2e_scenarios.sh

  (b) Use your Claude Code subscription session (Pro/Max plan
      Agent SDK credit pool). Verify Claude Code is logged in:

        claude /status

      If it reports a logged-in session, re-run THIS script
      without ANTHROPIC_API_KEY set — the Agent SDK will pick
      up the session automatically.

Either path costs roughly $0.05-$0.15 per scenario on Sonnet;
the full 8-scenario suite is under $2 per run. To run the
server-side gate WITHOUT spending tokens:

    uv run pytest -m release
EOF
    exit 1
fi

# Make sure both extras are installed. `uv sync --extra dev
# --extra e2e` is idempotent — if everything's already in place
# this is a no-op; if a clone is fresh or the e2e extra was
# never installed, it adds the claude-agent-sdk + its transitive
# deps under the new dependency group.
echo "==> uv sync --extra dev --extra e2e (Claude Agent SDK + dev test deps)"
uv sync --extra dev --extra e2e

echo "==> pytest -m e2e (Claude-in-the-loop, ${WHATCANIRUN_E2E_MODEL:-claude-sonnet-4-6})"
# `-q` matches the rest of the repo's pytest invocations; the
# per-scenario assertion failures stay loud enough on -q to
# triage relay regressions without extra verbosity.
uv run pytest -m e2e -q
