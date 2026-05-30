#!/usr/bin/env bash
# whatcanirun — PreCompact hook.
#
# Fires from `PreCompact` BEFORE Claude Code's /compact summarizer
# runs. Saves a markdown snapshot of the session's in-flight state
# to .claude/.pre-compact-snapshot.md so the SessionStart:compact
# reload (`reload-after-compact.sh`) can splice it back in.
#
# Why this exists: compaction discards the conversation transcript
# wholesale and replaces it with the model-written summary. Useful
# in-flight context that ISN'T in git — locked decisions, the
# "next step" plan, active poll/watcher tasks, the PR-cycle round
# we were on — survives only if (a) the model includes it in the
# summary or (b) a deterministic snapshot captures it on disk.
# This hook handles (b).
#
# What's captured:
#   - Static git state: branch, status, ahead-count, recent commits
#     on the active branch.
#   - Open PRs the user owns (via `gh pr list --author @me`).
#   - WIP branches with commits ahead of origin/main.
#   - Last 5 assistant text messages from the transcript (verbatim,
#     gives the next session "where I was in mid-thought").
#   - Last 30 tool-call invocations summarized as `name(arg-keys)`
#     (gives the next session "what was just being done").
#   - Timestamp + compaction trigger ("manual" vs "auto").
#
# Failure modes: any non-zero exit, missing jq, missing transcript
# file, or git/gh errors all degrade silently (exit 0 with whatever
# partial snapshot was assembled). The hook never blocks /compact.

set -u  # NOT set -e — graceful degradation per above
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/workspace}"
SNAPSHOT="${PROJECT_DIR}/.claude/.pre-compact-snapshot.md"

# --- Read hook input.
input=$(cat 2>/dev/null || echo "{}")
if ! command -v jq >/dev/null 2>&1; then
  exit 0  # without jq we can't parse the JSONL transcript
fi
transcript_path=$(printf '%s' "$input" | jq -r '.transcript_path // ""' 2>/dev/null)
trigger=$(printf '%s' "$input" | jq -r '.trigger // "manual"' 2>/dev/null)

# --- Build snapshot. mkdir -p ensures the .claude/ dir exists even
# on a fresh clone.
mkdir -p "${PROJECT_DIR}/.claude"

{
  echo "# Pre-compact snapshot"
  echo
  echo "- Saved: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Trigger: ${trigger}"
  echo
  echo "## Git state"
  echo
  if git -C "$PROJECT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    branch=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null || echo "?")
    echo "- Branch: \`${branch}\`"
    echo
    echo "### git status"
    echo '```'
    git -C "$PROJECT_DIR" status -sb 2>/dev/null | head -30
    echo '```'
    echo
    echo "### Commits ahead of origin/main on the current branch"
    echo '```'
    git -C "$PROJECT_DIR" log --oneline origin/main..HEAD 2>/dev/null | head -20
    echo '```'
  else
    echo "_not a git repo_"
  fi
  echo

  echo "## Open PRs (yours, --state open)"
  echo '```'
  if command -v gh >/dev/null 2>&1; then
    gh pr list --author "@me" --state open \
      --json number,title,headRefName,mergeStateStatus \
      --jq '.[] | "#\(.number) [\(.headRefName)] \(.mergeStateStatus) — \(.title)"' \
      2>/dev/null | head -20
  else
    echo "_gh not installed_"
  fi
  echo '```'
  echo

  echo "## Other WIP branches (commits ahead of origin/main, excluding current)"
  echo '```'
  if git -C "$PROJECT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    cur=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null)
    for b in $(git -C "$PROJECT_DIR" for-each-ref --format='%(refname:short)' refs/heads/ 2>/dev/null); do
      [[ "$b" == "main" || "$b" == "$cur" ]] && continue
      ahead=$(git -C "$PROJECT_DIR" rev-list --count "origin/main..$b" 2>/dev/null || echo 0)
      [[ "$ahead" -gt 0 ]] && echo "- ${b}: ${ahead} commits ahead"
    done | head -10
  fi
  echo '```'
  echo

  if [[ -n "$transcript_path" && -f "$transcript_path" ]]; then
    echo "## Last 5 assistant text messages (verbatim)"
    echo
    # Claude Code's transcript JSONL has varied across versions. Try
    # several shapes in order of likelihood. Each entry is one JSON
    # object on its own line. Filter to assistant text content only.
    echo '```'
    jq -r '
      select(.type == "assistant" or .message.role == "assistant")
      | (.message.content // .content // [])
      | if type == "array" then .[] else . end
      | select(.type == "text")
      | .text
    ' "$transcript_path" 2>/dev/null | tail -100
    echo '```'
    echo

    echo "## Last 30 tool calls (name + arg-keys only — too noisy at full input)"
    echo
    echo '```'
    jq -r '
      select(.type == "assistant" or .message.role == "assistant")
      | (.message.content // .content // [])
      | if type == "array" then .[] else . end
      | select(.type == "tool_use")
      | "- \(.name)(\(.input | keys | join(", ")))"
    ' "$transcript_path" 2>/dev/null | tail -30
    echo '```'
  else
    echo "## Transcript"
    echo
    echo "_transcript_path missing or unreadable; static state only._"
  fi
} > "$SNAPSHOT" 2>/dev/null

exit 0
