#!/usr/bin/env bash
# whatcanirun — PostToolUse hook with matcher "Bash".
#
# The hook fires on every Bash tool call (the matcher in
# `.claude/settings.local.json` is `"Bash"`, not a tighter
# `Bash(git commit *)` form — Claude Code's matcher syntax
# doesn't support command-pattern filtering, only tool name).
# The script filters internally for `git commit` invocations
# and exits 0 silently on everything else, so the hook is a
# no-op for non-commit Bash calls.
#
# Detects when a git commit flips a milestone row in
# `spec/INDEX.md` from ⬜ to ✓ and:
#
#   1. Writes a `.claude/.compact-pending` sentinel carrying the
#      milestone ID — survives session boundaries so the
#      UserPromptSubmit reminder fires even if the user resumes
#      hours later or in a new conversation.
#
#   2. Returns `additionalContext` for the CURRENT response so
#      the model recommends `/compact` inline (the inline nudge
#      complements the persistent sentinel).
#
# Hooks cannot programmatically invoke `/compact` themselves —
# the slash-command surface is user-side by design. This pair of
# hooks is the closest achievable: detection + reminder, with the
# user (or the model recommending to the user) running the
# command.
#
# Defensive: any failure path (missing jq, malformed input,
# detached HEAD, no INDEX.md) exits 0 silently so the user's
# commit workflow is never blocked.

set -u
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"

# --- Read hook input (JSON on stdin) and extract the bash command.
# Different Claude Code versions have published the bash command
# under different paths; try the documented shapes in order.
input=$(cat 2>/dev/null || echo "{}")
if ! command -v jq >/dev/null 2>&1; then
  exit 0  # without jq we can't parse stdin reliably; degrade quietly
fi
cmd=$(printf '%s' "$input" | jq -r '
  .tool_input.command //
  .hookSpecificOutput.bashToolInput.command //
  .bashToolInput.command //
  ""
' 2>/dev/null)

# --- Only act on `git commit` invocations.
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

cd "$PROJECT_DIR" 2>/dev/null || exit 0

# --- Inspect HEAD's diff against the previous commit. The
# milestone-flip pattern is precisely:
#
#     -| M{NN} | ... | ⬜ |
#     +| M{NN} | ... | ✓ |
#
# Both sides anchored on `| <symbol> |` at end-of-line. The order
# (and pair-up) is enforced by requiring a removed `⬜` row AND
# an added `✓` row in the same diff.
diff=$(git diff HEAD~1 HEAD -- spec/INDEX.md 2>/dev/null || echo "")
removed_unchecked=$(printf '%s' "$diff" | grep -E '^- *\|.*\| ⬜ \|$' | head -1)
added_checked=$(printf '%s' "$diff" | grep -E '^\+ *\|.*\| ✓ \|$' | head -1)

[[ -z "$removed_unchecked" || -z "$added_checked" ]] && exit 0

# --- Extract the milestone ID from the added row (e.g. "M09").
milestone=$(printf '%s' "$added_checked" | grep -oE '\bM[0-9]{2}\b' | head -1)
milestone="${milestone:-unknown}"

# --- Write the persistent sentinel so the next prompt reminder
# fires even across sessions / branch switches.
mkdir -p "${PROJECT_DIR}/.claude"
printf '%s\n' "$milestone" > "${PROJECT_DIR}/.claude/.compact-pending"

# --- Inline nudge for THIS response so the model can recommend
# `/compact` immediately if the next user turn is in the same
# session.
ctx=$(cat <<EOF
Milestone ${milestone} was just marked complete in this commit
(spec/INDEX.md flipped the row to ✓). Recommend running /compact
before starting the next milestone so the new milestone's context
window is clean. A persistent .claude/.compact-pending sentinel
was also written so the reminder fires on the next user prompt
even if this session ends first.
EOF
)
jq -n --arg ctx "$ctx" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

exit 0
