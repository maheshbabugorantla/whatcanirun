#!/usr/bin/env bash
# Host-uv install path for whatcanirun (M12 Slice B1).
#
# Run this from a fresh clone of the whatcanirun repo on a host that has
# `git` and `uv` (https://docs.astral.sh/uv/) installed.
#
#   git clone https://github.com/maheshbabugorantla/whatcanirun
#   cd whatcanirun
#   ./scripts/install_host_uv.sh
#
# What it does:
#   1. Verifies uv is on PATH.
#   2. Runs `uv sync` to materialize the project's virtualenv at .venv/.
#   3. Runs `uv run whatcanirun-mcp prefetch` to warm CP + HF caches under
#      $XDG_CACHE_HOME/whatcanirun (defaults to ~/.cache/whatcanirun).
#   4. Optionally runs the release-gate stdio test (pytest -m release) so
#      a clean install confirms the binary handshakes and every tool
#      surfaces a well-formed TrustEnvelope.
#   5. Prints the MCP client config block users paste into Claude Desktop /
#      Claude Code / Cursor / Cline to wire the server.
#
# Flags:
#   --no-prefetch  Skip the cache warmup. Useful when re-running the script
#                  to re-print the client config block.
#   --no-test      Skip the release-gate test. Useful in CI where the test
#                  is invoked separately.
#   --help, -h     Print usage and exit 0.

set -euo pipefail

usage() {
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

PREFETCH=1
RUN_TEST=1
for arg in "$@"; do
    case "$arg" in
        --no-prefetch) PREFETCH=0 ;;
        --no-test)     RUN_TEST=0 ;;
        --help|-h)     usage; exit 0 ;;
        *)
            echo "install_host_uv.sh: unknown flag: $arg" >&2
            echo "  Try --help for the usage." >&2
            exit 2
            ;;
    esac
done

# Run from the repo root regardless of where the script was invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "pyproject.toml" ]] || ! grep -q '^name = "whatcanirun"' pyproject.toml; then
    echo "install_host_uv.sh: not inside the whatcanirun repo (no whatcanirun pyproject.toml at $REPO_ROOT)" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
install_host_uv.sh: `uv` is not on PATH.

Install it once with:
    curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run this script.
EOF
    exit 1
fi

echo "==> uv sync"
uv sync

if [[ "$PREFETCH" -eq 1 ]]; then
    echo "==> warming CP + HF caches (this is the slow first run)"
    uv run whatcanirun-mcp prefetch
else
    echo "==> skipping cache prefetch (--no-prefetch)"
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
    echo "==> release-gate test (pytest -m release)"
    # Exit code 5 means pytest collected no tests matching the marker.
    # That's OK during M12 development before Slice C lands the gate
    # test; after merge there is always at least one release-marked
    # test, so a 5 then means a regression worth surfacing.
    set +e
    uv run pytest -m release -q
    test_rc=$?
    set -e
    if [[ $test_rc -ne 0 && $test_rc -ne 5 ]]; then
        echo "install_host_uv.sh: release-gate test failed (pytest exit $test_rc)" >&2
        exit $test_rc
    fi
    if [[ $test_rc -eq 5 ]]; then
        echo "    (no release-marked tests collected; skipping)"
    fi
else
    echo "==> skipping release-gate test (--no-test)"
fi

cat <<EOF

================================================================
Install complete. Wire the server into your MCP client config:

  "mcpServers": {
    "whatcanirun": {
      "command": "uv",
      "args": ["run", "--directory", "$REPO_ROOT", "whatcanirun-mcp"]
    }
  }

Per-client config files + env-var passthrough are documented at
docs/MCP.md.
================================================================
EOF
