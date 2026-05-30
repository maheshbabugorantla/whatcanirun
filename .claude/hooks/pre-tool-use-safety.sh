#!/usr/bin/env bash
# whatcanirun — PreToolUse safety blocker.
#
# Adapted from disler/claude-code-hooks-mastery's pre_tool_use.py.
# Adds belt-and-suspenders defense against two failure modes that
# matter to this project:
#
#   1. Destructive `rm -rf` against high-blast-radius paths
#      (/, ~, $HOME, *, ., parent-relative). Even with the Bash
#      tool's permission prompt, a typo'd command that slips
#      through is high-cost; this hook returns exit 2 to abort
#      the tool call before it runs.
#
#   2. Read/Edit/Write of .env files. Per CLAUDE.md the project
#      keeps real `COMPUTEPRICES_API_KEY`, `AA_API_KEY`, and
#      `HF_TOKEN` in `.env` (gitignored). Even a benign tool call
#      that loads .env into context bytes risks the model echoing
#      a secret in a later turn. .env.sample is allowed since it
#      carries no secrets.
#
# Exit codes:
#   0 — allow the tool call (default for non-matching inputs)
#   2 — block the tool call with the reason written to stderr
#       (Claude Code surfaces this back to the model so it knows
#       to take a different approach)
#
# Failure modes: missing jq, malformed input, or pattern logic
# errors fall through to exit 0 (allow) rather than block on a
# parser bug. The hook should fail OPEN, not closed — a buggy
# safety check that blocks legitimate work is worse than one
# that occasionally misses an edge case (the user is still in
# the loop via Claude Code's permission prompt).

set -u  # NOT set -e — graceful degradation
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"

input=$(cat 2>/dev/null || echo "{}")
if ! command -v jq >/dev/null 2>&1; then
  exit 0  # without jq we can't parse the input reliably; fail open
fi

tool_name=$(printf '%s' "$input" | jq -r '.tool_name // ""' 2>/dev/null)
tool_input=$(printf '%s' "$input" | jq -c '.tool_input // {}' 2>/dev/null)

# --- Rule 1: dangerous rm patterns in Bash commands.
if [[ "$tool_name" == "Bash" ]]; then
  cmd=$(printf '%s' "$tool_input" | jq -r '.command // ""' 2>/dev/null)
  norm=$(printf '%s' "$cmd" | tr '[:upper:]' '[:lower:]' | tr -s ' ')

  # Detect rm with a recursive + force combination. Allows -r alone
  # (legitimate for empty dirs) and -f alone (no recursive); the
  # combination is the dangerous one.
  rm_recursive_force=0
  if printf '%s' "$norm" | grep -Eq '\brm\s+([^|;&]*\s)?-[a-z]*r[a-z]*f|\brm\s+([^|;&]*\s)?-[a-z]*f[a-z]*r|\brm\s+([^|;&]*\s)?--recursive\s+([^|;&]*)?--force|\brm\s+([^|;&]*\s)?--force\s+([^|;&]*)?--recursive'; then
    rm_recursive_force=1
  fi

  if [[ "$rm_recursive_force" -eq 1 ]]; then
    # Check the target path. Block if it touches a dangerous root.
    # Dangerous: /, ~, $HOME, *, ., .. — anything that could
    # escape the project's current working tree. The `/` case
    # needs to match `/` at end-of-string (`rm -rf /`) and `/`
    # followed by a non-lowercase char (`rm -rf /tmp` is fine,
    # `rm -rf /` or `rm -rf /*` is not).
    if printf '%s' "$norm" | grep -Eq '\brm\s+[^|;&]*(\s|=)(/($|\*|\s)|/[^a-z]|~($|/|\s)|\$home|\.\.|\*($|\s)|\.($|\s))'; then
      printf 'BLOCKED: refusing rm -rf against a high-blast-radius path. Re-issue with a specific in-tree path or remove the recursive+force combination.\nCommand: %s\n' "$cmd" >&2
      exit 2
    fi
  fi
fi

# --- Rule 2: .env access (read, edit, write, or bash command).
# Allow .env.sample explicitly since it carries no secrets.
case "$tool_name" in
  Read|Edit|MultiEdit|Write|NotebookEdit)
    file_path=$(printf '%s' "$tool_input" | jq -r '.file_path // ""' 2>/dev/null)
    base=$(basename -- "$file_path")
    if [[ "$base" == .env || "$base" == .env.* ]] && [[ "$base" != .env.sample ]]; then
      printf 'BLOCKED: refusing %s on %s. The .env file holds real API keys (COMPUTEPRICES_API_KEY, AA_API_KEY, HF_TOKEN per CLAUDE.md); reading it loads secrets into context. Use .env.sample for documentation; ask the user out-of-band for real key values.\n' "$tool_name" "$file_path" >&2
      exit 2
    fi
    ;;
  Bash)
    cmd=$(printf '%s' "$tool_input" | jq -r '.command // ""' 2>/dev/null)
    # Match bare `.env` (and `.env.local`, `.env.production`, etc.)
    # while skipping the documented `.env.sample` exception.
    # Word boundary `\b` doesn't work before `.env` because both
    # space and `.` are non-word — no transition. Explicitly
    # anchor with `(^|[\s/=>])` on the left and `($|[\s/]| )` on
    # the right.
    if printf '%s' "$cmd" | grep -Eq '(^|[[:space:]/=>])\.env(\.[a-z]+)?($|[[:space:]/])' \
       && ! printf '%s' "$cmd" | grep -Eq '(^|[[:space:]/=>])\.env\.sample($|[[:space:]/])'; then
      printf 'BLOCKED: bash command references .env (real-secrets file). Edit .env.sample instead, or ask the user to make the change manually if a real key needs to change.\nCommand: %s\n' "$cmd" >&2
      exit 2
    fi
    ;;
esac

exit 0
