"""Claude Agent SDK driver for the e2e scenario harness.

The release-gate test (`tests/release/test_stdio_install.py`) drives
the MCP server directly via FastMCP `Client`: it asserts the SERVER
returns well-formed trust envelopes. This module sits one layer up
— it drives the SAME server through a real Claude Agent SDK loop,
so tests can assert that *Claude itself* picks the right tool path
and relays the trust contract honestly in natural language.

What this proves vs. the release-gate test:

- The release gate proves the server's contract (every numerical
  response carries an envelope, weakest-link rule holds, etc.).
- This loop proves the CLIENT-SIDE relay — that Claude reading the
  server's `INSTRUCTIONS` string + tool descriptions actually picks
  the documented tool path and surfaces the caveats verbatim.

The `INSTRUCTIONS` string in `src/whatcanirun/server.py` is the
load-bearing piece: it tells Claude what to say. If the e2e
harness catches a relay regression (Claude bluffing instead of
refusing on batch>1, dropping the workload-profile assumption,
etc.), the fix is almost always tightening that string.

Why claude-agent-sdk (not the direct `anthropic` Messages API):

- The Agent SDK spawns the local `claude` CLI binary which
  authenticates against the user's Claude Code session. On Pro
  ($20/mo) / Max ($200/mo) subscriptions this redeems against
  the Agent SDK credit pool (post-2026-06-15 billing split);
  direct `anthropic.messages.create()` would have billed
  pay-as-you-go API balance.
- The Agent SDK has NATIVE MCP server support — passing
  `mcp_servers={"whatcanirun": {"command": "uv", "args": [...]}}`
  in `ClaudeAgentOptions` is enough; no manual FastMCP <->
  Anthropic tool-schema projection, no manual `tool_use` /
  `tool_result` loop. The SDK owns the subprocess lifecycle and
  the tool-dispatch loop; we just stream messages and inspect.
- This drops ~150 LOC of bridge code vs the earlier prototype
  (preserved at git tag `e2e-anthropic-sdk-archive`).

Cost / runtime: each scenario runs a Claude loop until
`stop_reason="end_turn"`. With Sonnet, that's ~1-3 turns and
~$0.05-0.15 per scenario; the full suite is well under $2.
Gated behind `ANTHROPIC_API_KEY` (or a Claude Code session) so
default `pytest -q` never spends money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from claude_agent_sdk.types import McpStdioServerConfig

# Pull the server's `instructions` string from source. The whole
# point of this harness is to verify the LLM client follows that
# string's relay rules (workload-assumption surfacing, refusal
# honesty, etc.); not passing it would test Claude's default
# behaviour given only tool descriptions, which is a weaker
# contract than spec/M09 promises.
# `whatcanirun` lacks a py.typed marker, so mypy treats first-
# party imports as untyped when checked outside the project gate
# (pre-commit only runs mypy on `^src/`). The inline ignore keeps
# `mypy tests/e2e/` clean for local runs.
from whatcanirun.server import INSTRUCTIONS as _SERVER_INSTRUCTIONS  # type: ignore[import-untyped]

_LOG = logging.getLogger(__name__)


# Safety caps. `max_turns` bounds the agentic loop; `max_budget_usd`
# bounds the per-scenario spend so a misbehaving Claude can't burn
# the credit pool. 8 turns covers the longest expected scenario
# (multi-turn elicitation 5+6); $0.25 is ~5x the Sonnet expected
# spend per scenario so a single runaway gets caught.
_MAX_TURNS = 8
_MAX_BUDGET_USD = 0.25

# MCP server name registered with the Agent SDK. Tool names Claude
# sees are prefixed `mcp__<this>__<tool_name>` per the SDK
# convention.
MCP_SERVER_NAME = "whatcanirun"

# Built-in Agent SDK tools we explicitly DENY. The harness only
# exercises whatcanirun's MCP surface; the built-ins (Read, Write,
# Edit, Bash, etc.) would let Claude poke at the test runner's
# filesystem, which neither the scenarios nor the trust contract
# care about. Denying narrows the test surface to exactly the
# server-side contract.
_DENIED_BUILTINS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Task",
    "NotebookEdit",
    "TodoWrite",
]


@dataclass
class ToolCall:
    """One tool call Claude made during a scenario, plus the
    result the Agent SDK paired back. Tests assert on this
    sequence — `name` for the tool-path check, `arguments` to
    confirm Claude inferred the right slugs / budget, `result`
    for the envelope-relay check."""

    name: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool = False


@dataclass
class ScenarioRun:
    """Result of one Claude-in-the-loop scenario. The fields map
    directly onto the three assertion buckets per scenario:

    - `tool_calls` — which tools Claude routed to (assert by `name`)
    - `tool_calls[*].result` — what came back (assert envelope shape)
    - `final_text` — Claude's last natural-language reply
      (assert soft keywords)
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str | None = None
    turns: int = 0
    cost_usd: float | None = None
    is_error: bool = False

    def tool_names(self) -> list[str]:
        """Ordered list of tool names Claude called, with the
        `mcp__<server>__` prefix stripped so assertions can name
        the bare tool (e.g. `fit_check` not
        `mcp__whatcanirun__fit_check`)."""
        return [_strip_mcp_prefix(tc.name) for tc in self.tool_calls]

    def first_call_to(self, name: str) -> ToolCall | None:
        """First call whose stripped name matches `name`. Accepts
        either the bare name (`fit_check`) or the prefixed
        (`mcp__whatcanirun__fit_check`)."""
        target = _strip_mcp_prefix(name)
        for tc in self.tool_calls:
            if _strip_mcp_prefix(tc.name) == target:
                return tc
        return None


def _strip_mcp_prefix(tool_name: str) -> str:
    """Strip the `mcp__<server>__` prefix from a tool name. The
    Agent SDK prefixes MCP tool names with `mcp__<server>__` per
    its convention; scenario assertions name tools by their bare
    server-side identifier (`fit_check`, `budget_to_plan`, etc.).
    Tools that aren't from an MCP server (built-ins, hooks) pass
    through unchanged."""
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    if tool_name.startswith(prefix):
        return tool_name[len(prefix) :]
    return tool_name


def _build_options(
    *,
    mcp_command: str,
    mcp_args: list[str],
    system_prompt: str | None = None,
    model: str | None = None,
    env: dict[str, str] | None = None,
) -> ClaudeAgentOptions:
    """Construct the Agent SDK options for one scenario run.

    Defaults the system prompt to the server's `INSTRUCTIONS`
    string. A real MCP client (Claude Desktop, Claude Code reads
    `instructions` on `initialize`; the Agent SDK doesn't auto-
    pickup, so we must inject it explicitly as a FULL replacement
    of the default Claude Code persona — passing a bare `str` as
    `system_prompt` overrides the preset entirely (per the SDK's
    docs).

    Denies the built-in Claude Code tools (Read, Write, Bash etc.)
    so the harness exercises only whatcanirun's MCP surface.
    """
    server_config: McpStdioServerConfig = {
        "type": "stdio",
        "command": mcp_command,
        "args": mcp_args,
    }
    if env:
        server_config["env"] = env
    return ClaudeAgentOptions(
        system_prompt=system_prompt if system_prompt is not None else _SERVER_INSTRUCTIONS,
        mcp_servers={MCP_SERVER_NAME: server_config},
        # Allow only whatcanirun MCP tools. The wildcard
        # `mcp__whatcanirun__*` matches every tool the server
        # exposes without requiring this code to enumerate them
        # — server-side additions automatically flow through.
        allowed_tools=[f"mcp__{MCP_SERVER_NAME}__*"],
        disallowed_tools=_DENIED_BUILTINS,
        max_turns=_MAX_TURNS,
        max_budget_usd=_MAX_BUDGET_USD,
        model=model,
        # `dontAsk` means: anything not in `allowed_tools` is
        # denied without prompting. The harness has no human in
        # the loop; the default `default` mode would block on a
        # permission prompt forever.
        permission_mode="dontAsk",
        # Keep stderr clean — the SDK's chatter would muddy
        # pytest output. Tests still get tracebacks on real errors
        # via the ResultMessage.errors field.
        stderr=None,
    )


async def run_scenario(
    *,
    question: str,
    mcp_command: str,
    mcp_args: list[str],
    model: str | None = None,
    system_prompt: str | None = None,
    env: dict[str, str] | None = None,
) -> ScenarioRun:
    """Drive a Claude tool-use loop against the MCP server via the
    Agent SDK. Returns a `ScenarioRun` carrying the ordered tool
    calls, paired tool results, Claude's final natural-language
    reply, stop reason, turn count, and aggregate cost.

    `question` is the user's natural-language ask — pasted verbatim
    from `docs/SCENARIOS.md` so the scenarios stay aligned with the
    prose-shape doc.

    The Agent SDK owns the loop: tool dispatch, tool_use_id
    correlation, and turn termination all happen inside `query()`.
    We just stream messages out and assemble the `ScenarioRun`.
    """
    options = _build_options(
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        system_prompt=system_prompt,
        model=model,
        env=env,
    )

    run = ScenarioRun()
    # Pair tool_use blocks with their results by id. The SDK emits
    # the result either as a `ToolResultBlock` inside a `UserMessage`
    # or as `UserMessage.tool_use_result` depending on flow; we
    # accept both shapes and merge into `run.tool_calls[*].result`.
    pending: dict[str, ToolCall] = {}

    async for message in query(prompt=question, options=options):
        if isinstance(message, SystemMessage):
            # Init / status. Useful for debugging MCP attach issues
            # but not load-bearing for scenario assertions.
            _LOG.debug("system message: subtype=%s data=%s", message.subtype, message.data)
            continue

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tc = ToolCall(
                        name=block.name,
                        arguments=dict(block.input) if isinstance(block.input, dict) else {},
                        result=None,
                    )
                    run.tool_calls.append(tc)
                    pending[block.id] = tc
                elif isinstance(block, TextBlock):
                    # AssistantMessage text blocks come BEFORE the
                    # final ResultMessage. Overwrite to keep the
                    # most recent assistant text as the relay.
                    run.final_text = block.text
            continue

        if isinstance(message, UserMessage):
            # The SDK echoes the tool result back as either a
            # `ToolResultBlock` list or a structured
            # `tool_use_result` dict. Walk both.
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id in pending:
                        tc = pending.pop(block.tool_use_id)
                        tc.result = block.content
                        tc.is_error = bool(block.is_error)
            # Some flows route the structured result via the
            # `tool_use_result` dict instead of a `ToolResultBlock`.
            # Apply to the most-recent pending call (the dict
            # isn't required to carry an id back, so we can't
            # disambiguate further when multiple are pending).
            if message.tool_use_result is not None and pending:
                last_id = next(reversed(pending))
                tc = pending.pop(last_id)
                tc.result = message.tool_use_result
            continue

        if isinstance(message, ResultMessage):
            # Terminal. Capture the SDK's authoritative final text
            # (its `result` field is the assembled final reply).
            if message.result:
                run.final_text = message.result
            run.stop_reason = message.stop_reason
            run.turns = message.num_turns
            run.cost_usd = message.total_cost_usd
            run.is_error = message.is_error
            return run

    # Stream ended without a ResultMessage. The SDK shouldn't
    # normally do this; treat as an unfinished run with whatever
    # we collected.
    _LOG.warning("scenario stream exhausted without a terminal ResultMessage")
    return run
