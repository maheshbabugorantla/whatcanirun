#!/usr/bin/env bash
# whatcanirun — PostToolUseFailure logger.
#
# Adapted from disler/claude-code-hooks-mastery's
# post_tool_use_failure.py (theirs was a Python uv-managed
# script; we keep ours in bash for the same low-adoption reason).
#
# Appends a structured JSON line to `logs/post-tool-use-failure.jsonl`
# every time a tool call fails. Each line is one self-contained
# JSON object with:
#   - logged_at  (ISO-8601 UTC timestamp)
#   - tool_name  (the failing tool)
#   - tool_input (the args that produced the failure)
#   - error      (the error payload Claude Code surfaces)
#   - cwd        (working directory at the time)
#   - session_id (Claude Code session ID, if surfaced)
#
# JSONL not JSON — append-friendly, never needs a read-modify-write
# of an entire array. Failures from the same session sort
# chronologically as written; no need for an external `id` field.
#
# Why this exists: M10 D-Phase work will run scripts/m10/sanity_check_cells.py
# against many candidate files and we want a forensic trail when
# something blows up. The existing `git log` + per-PR commit
# messages cover the SUCCESS path; this covers the failure path
# without polluting commit history.
#
# Failure modes: missing jq, unwriteable logs/ dir, or any parsing
# error exits 0 silently. The hook never blocks Claude Code's
# error-handling flow.

set -u  # NOT set -e — graceful degradation
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/post-tool-use-failure.jsonl"

input=$(cat 2>/dev/null || echo "{}")
if ! command -v jq >/dev/null 2>&1; then
  exit 0
fi

mkdir -p "$LOG_DIR" 2>/dev/null || exit 0

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
line=$(printf '%s' "$input" | jq -c --arg ts "$ts" '
  {
    logged_at: $ts,
    hook_event_name: (.hook_event_name // "PostToolUseFailure"),
    session_id: (.session_id // ""),
    tool_name: (.tool_name // "unknown"),
    tool_use_id: (.tool_use_id // ""),
    tool_input: (.tool_input // {}),
    error: (.error // {}),
    cwd: (.cwd // ""),
    permission_mode: (.permission_mode // ""),
    transcript_path: (.transcript_path // "")
  }
' 2>/dev/null) || exit 0

printf '%s\n' "$line" >> "$LOG_FILE" 2>/dev/null

exit 0
