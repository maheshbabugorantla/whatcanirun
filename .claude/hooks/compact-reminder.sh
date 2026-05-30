#!/usr/bin/env bash
# whatcanirun — UserPromptSubmit hook.
#
# If a prior milestone-completion commit dropped
# `.claude/.compact-pending`, surface the `/compact` reminder as
# `additionalContext` on this prompt and delete the sentinel.
#
# Together with `detect-milestone-complete.sh` (PostToolUse), this
# implements the persistence layer: the inline nudge fires when
# the milestone-marking commit runs in the same session, and this
# reminder catches every other case (PR review took a day,
# session was cleared, etc.).
#
# Defensive: missing jq exits 0 silently rather than blocking the
# user's prompt.

set -u
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"
sentinel="${PROJECT_DIR}/.claude/.compact-pending"

[[ -f "$sentinel" ]] || exit 0
command -v jq >/dev/null 2>&1 || exit 0  # leave sentinel for next run

milestone=$(head -1 "$sentinel" 2>/dev/null || echo "unknown")

ctx=$(cat <<EOF
Milestone ${milestone} was marked complete in a recent commit
(spec/INDEX.md flipped to ✓ — see latest git log). Before
continuing substantive work, recommend running /compact so the
next milestone starts with a clean context window. Surface this
in your response if the user hasn't already run it.
EOF
)

# Render the additionalContext JSON FIRST. Only on successful
# render do we clear the sentinel — guarantees "fires exactly
# once on successful delivery" rather than "fires at most once on
# invocation". A jq runtime failure (OOM, corrupted binary)
# leaves the sentinel in place so the next prompt retries.
output=$(jq -n --arg ctx "$ctx" '{
  hookSpecificOutput: {
    hookEventName: "UserPromptSubmit",
    additionalContext: $ctx
  }
}') || exit 0

rm -f "$sentinel"
printf '%s\n' "$output"
exit 0
