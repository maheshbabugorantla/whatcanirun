"""FastMCP ↔ Anthropic tool-use bridge for the e2e scenario harness.

The release-gate test (`tests/release/test_stdio_install.py`) drives
the MCP server directly via FastMCP `Client`: it asserts the SERVER
returns well-formed trust envelopes. This module sits one layer up
— it drives the SAME server through an Anthropic Claude loop, so
tests can assert that *Claude itself* picks the right tool path
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
etc.), the fix is almost always tightening that string — same
cycle as the manual SCENARIOS walkthrough doc, but mechanically.

Cost / runtime: each scenario runs a Claude loop until
`stop_reason="end_turn"`. With Sonnet, that's ~1-3 turns and
~$0.05-0.15 per scenario. The full 8-scenario suite is well under
$2 against the live API. Gated behind `ANTHROPIC_API_KEY` so
default `pytest -q` never spends money.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from fastmcp import Client

_LOG = logging.getLogger(__name__)


# Hard-cap on turns so a misbehaving Claude can't loop forever and
# burn the user's API budget. 8 turns is generous — every v1
# scenario completes in 1-3 turns; multi-turn elicitation (5/6) in
# 2-3.
_MAX_TURNS = 8

# Per-call token budget. Claude's tool-call responses are short
# (~100-300 tokens of natural language + tool_use blocks), but the
# final relay can be longer when Claude restates a multi-row plan.
_MAX_TOKENS = 2048


@dataclass
class ToolCall:
    """One tool call Claude made during a scenario, plus the
    raw response the harness fed back. Tests assert on this
    sequence — `name` for the tool-path check, `arguments` to
    confirm Claude inferred the right slugs / budget, `result`
    for the envelope-relay check."""

    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class ScenarioRun:
    """Result of one Claude-in-the-loop scenario. The three fields
    map directly onto the three assertion buckets per scenario:

    - `tool_calls` — which tools Claude routed to (assert by `name`)
    - `tool_calls[*].result` — what came back (assert envelope shape)
    - `final_text` — Claude's last natural-language reply
      (assert soft keywords)
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str | None = None
    turns: int = 0

    def tool_names(self) -> list[str]:
        """Ordered list of tool names Claude called. Most scenario
        assertions are subset checks on this list."""
        return [tc.name for tc in self.tool_calls]

    def first_call_to(self, name: str) -> ToolCall | None:
        """First call to `name`, or None if Claude never called it."""
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None


async def list_anthropic_tools(client: Client[Any]) -> list[dict[str, Any]]:
    """Convert the FastMCP server's advertised tools into Anthropic's
    tool-schema shape.

    The bridge is mechanical: MCP's `Tool.inputSchema` (camelCase, JSON
    Schema) → Anthropic's `tool["input_schema"]` (snake_case, same JSON
    Schema body). Name + description carry across verbatim.

    Resources are exposed via a separate `read_mcp_resource` pseudo-
    tool below — Anthropic doesn't have a first-class resource concept,
    so giving Claude an explicit "read this URI" tool is the cleanest
    way to surface `cost-cells://provenance` to the loop without
    pre-prompting Claude with the URI.
    """
    tools = await client.list_tools()
    schemas: list[dict[str, Any]] = []
    for t in tools:
        schemas.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
        )
    # Always advertise the resource-reader pseudo-tool. Scenario 7
    # exercises `cost-cells://provenance`; future scenarios that
    # need `cost-cells://current` (Parquet) would route through here
    # too. URI is constrained to the allowlist so Claude can't be
    # talked into reading something unexpected.
    schemas.append(
        {
            "name": "read_mcp_resource",
            "description": (
                "Read one of this server's MCP resources by URI. "
                "Available URIs: `cost-cells://provenance` (JSON "
                "document with upstream source attributions, "
                "license terms, and audit links — call this when "
                "the user asks about sources, attribution, or where "
                "the numbers come from). Returns the resource's "
                "text content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "enum": ["cost-cells://provenance"],
                        "description": "MCP resource URI to read.",
                    },
                },
                "required": ["uri"],
            },
        }
    )
    return schemas


async def _dispatch_tool_call(
    client: Client[Any],
    tool_name: str,
    tool_input: dict[str, Any],
) -> Any:
    """Route one Claude `tool_use` to either a FastMCP `call_tool`
    or our `read_mcp_resource` pseudo-tool.

    The structured_content unwrap mirrors `tests/release/test_stdio_install.py`'s
    `_unwrap_result`: Pydantic-returning tools ship `{"result": <obj>}`,
    plain-dict tools (e.g. `list_catalog`) ship the dict directly.
    Returning the unwrapped value to Claude means the LLM sees the
    actual response shape documented in the tool's docstring, not
    FastMCP's wire wrapper.
    """
    if tool_name == "read_mcp_resource":
        uri = tool_input["uri"]
        contents = await client.read_resource(uri)
        # FastMCP returns a list of content entries; concatenate the
        # text bodies. Resources we expose are text/JSON, so this
        # gives Claude a single readable blob.
        return "".join(getattr(c, "text", "") or "" for c in contents)

    result = await client.call_tool(tool_name, tool_input)
    sc = result.structured_content
    if isinstance(sc, dict) and "result" in sc and len(sc) == 1:
        return sc["result"]
    return sc


def _stringify_for_claude(payload: Any) -> str:
    """Tool-result blocks must be strings for the Anthropic API. JSON-
    serialize so Claude sees the same structure the server returned.
    `default=str` covers the rare non-JSON-native (datetimes,
    Decimal) without crashing the loop on a stray type."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str)


async def run_scenario(
    *,
    anthropic_client: AsyncAnthropic,
    fastmcp_client: Client[Any],
    question: str,
    model: str,
    system: str | None = None,
    elicitation_replies: list[str] | None = None,
) -> ScenarioRun:
    """Drive a Claude tool-use loop against the MCP server until
    Claude returns `end_turn` (or the safety cap fires).

    `question` is the user's natural-language ask — pasted verbatim
    from `docs/SCENARIOS.md` so the scenarios stay aligned with the
    prose-shape doc.

    `elicitation_replies` simulates the multi-turn flows (Scenarios
    5 + 6). After Claude's first end_turn (asking for repo_id or a
    workload profile), the harness sends the next reply from the
    list and continues looping. Most scenarios pass `None` here.

    The return value is a `ScenarioRun` carrying the tool-call
    sequence and Claude's final reply. Tests assert on it.
    """
    run = ScenarioRun()
    messages: list[MessageParam] = [{"role": "user", "content": question}]
    tools = await list_anthropic_tools(fastmcp_client)
    pending_replies = list(elicitation_replies or [])

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _MAX_TOKENS,
        "tools": tools,
        "messages": messages,
    }
    if system:
        create_kwargs["system"] = system

    for turn in range(_MAX_TURNS):
        run.turns = turn + 1
        create_kwargs["messages"] = messages
        resp = await anthropic_client.messages.create(**create_kwargs)
        run.stop_reason = resp.stop_reason

        # Capture final natural-language text from the assistant. A
        # turn can have both `text` and `tool_use` blocks; we keep
        # the joined text of the last turn so soft keyword checks
        # match whatever Claude said last.
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        if text_parts:
            run.final_text = "".join(text_parts)

        # Persist the assistant turn into the message log so Claude
        # sees its own prior tool_use ids on the next iteration.
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "tool_use":
            tool_use_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            tool_result_blocks: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                tool_input = dict(block.input) if isinstance(block.input, dict) else {}
                try:
                    result_payload = await _dispatch_tool_call(
                        fastmcp_client, block.name, tool_input
                    )
                except Exception as exc:
                    # Surface the tool failure to Claude so it can
                    # recover (e.g. retry with corrected args) rather
                    # than crashing the loop. The test still sees the
                    # call via `run.tool_calls`.
                    result_payload = {"error": f"{type(exc).__name__}: {exc}"}
                    _LOG.warning(
                        "scenario tool call failed: name=%s err=%s",
                        block.name,
                        exc,
                    )
                run.tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=tool_input,
                        result=result_payload,
                    )
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _stringify_for_claude(result_payload),
                    }
                )
            # Cast: Anthropic SDK types `content` as
            # `str | Iterable[ContentBlockParam]`. tool_result_blocks
            # are dicts conforming to ToolResultBlockParam at runtime,
            # but mypy can't narrow the typeddict-item check on a
            # bare list of dicts. The cast preserves runtime
            # behaviour while clearing the typeddict-item error.
            messages.append(
                cast(
                    MessageParam,
                    {"role": "user", "content": tool_result_blocks},
                )
            )
            continue

        # end_turn or max_tokens — Claude is done responding. If a
        # scripted elicitation reply is queued, send it as the next
        # user turn; otherwise the scenario is finished.
        if pending_replies:
            next_reply = pending_replies.pop(0)
            messages.append({"role": "user", "content": next_reply})
            continue
        return run

    # Safety-cap exhaustion — return whatever was captured so the
    # test can assert on the partial state and the cap shows up in
    # the failure message.
    _LOG.warning(
        "scenario hit turn cap (%d) without end_turn — returning partial run",
        _MAX_TURNS,
    )
    return run
