#!/usr/bin/env bash
# whatcanirun — post-compaction context reloader.
#
# Fires from `SessionStart` with matcher `compact`. Project-root
# CLAUDE.md auto-reloads, so this hook ONLY injects content that
# DOES NOT survive compaction on its own:
#   - spec/SHARED.md trust-contract excerpts (lines 1-180 ish)
#   - spec/INDEX.md milestone status (so post-compact knows what's
#     in progress vs done)
#   - the active branch + last 5 commits (so post-compact knows
#     where the in-progress work lives)
#   - the pre-compact snapshot file (.claude/.pre-compact-snapshot.md)
#     written by the PreCompact hook, if present. Carries
#     in-flight session state — locked decisions, last assistant
#     messages, recent tool calls, open PRs — that would otherwise
#     be lost in the compaction summary.
#
# Returns JSON on stdout per the Claude Code hooks contract:
#   {"hookSpecificOutput":{"hookEventName":"SessionStart",
#    "additionalContext":"..."}}
#
# Failure modes: any non-zero exit or invalid JSON is treated as
# "no additional context" by the runtime — the session continues
# without the reload. We always exit 0 and fall back to empty
# context rather than blocking session startup on a missing file.

set -u  # NOT set -e — we want graceful degradation, see above
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"

# --- Gather the load-bearing excerpts (degrade silently on miss).
shared=""
if [[ -f "${PROJECT_DIR}/spec/SHARED.md" ]]; then
  shared=$(sed -n '1,180p' "${PROJECT_DIR}/spec/SHARED.md")
fi

index=""
if [[ -f "${PROJECT_DIR}/spec/INDEX.md" ]]; then
  # The milestone status table — first 30 lines covers it.
  index=$(sed -n '1,30p' "${PROJECT_DIR}/spec/INDEX.md")
fi

branch=""
if git -C "${PROJECT_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  branch=$(git -C "${PROJECT_DIR}" branch --show-current 2>/dev/null || echo "")
  recent=$(git -C "${PROJECT_DIR}" log --oneline -5 2>/dev/null || echo "")
fi

# Pre-compact snapshot, written by save-context-before-compact.sh
# (the PreCompact hook). Read once and unlink so a stale snapshot
# from a prior compaction can't masquerade as fresh on a later
# session restart. Missing file is fine — degrade silently.
snapshot=""
snapshot_path="${PROJECT_DIR}/.claude/.pre-compact-snapshot.md"
if [[ -f "$snapshot_path" ]]; then
  snapshot=$(cat "$snapshot_path" 2>/dev/null || echo "")
  rm -f "$snapshot_path"
fi

# --- Compose the additional-context block.
context=$(cat <<EOF
## Post-Compaction Reload (whatcanirun)

Compaction just ran. CLAUDE.md auto-reloaded on its own. The
following spec excerpts + git state DID NOT survive and are
re-injected here so the post-compact session has the same
trust-contract foundation the pre-compact one did.

### Active git state

Branch: ${branch:-<not a git repo>}

Recent commits:
\`\`\`
${recent:-<none>}
\`\`\`

### spec/INDEX.md — milestone status

\`\`\`
${index:-<spec/INDEX.md not found>}
\`\`\`

### spec/SHARED.md — first 180 lines (Trust Contract + Calibration + ADR table)

\`\`\`
${shared:-<spec/SHARED.md not found>}
\`\`\`

Reminder: \`spec/M{NN}-*.md\` files are NOT auto-injected. Read
the active milestone's file before substantive work in its area.

${snapshot:+### Pre-compact snapshot (in-flight state)

${snapshot}
}
EOF
)

# --- Emit the JSON the hooks contract expects. jq is required;
# if absent, fall back to a printf-escaped JSON (best-effort).
if command -v jq >/dev/null 2>&1; then
  jq -n --arg ctx "$context" \
    '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $ctx}}'
else
  # Escape backslashes, quotes, and newlines for JSON without jq.
  esc=$(printf '%s' "$context" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])' 2>/dev/null || echo "")
  printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$esc"
fi

exit 0
