"""Claude-in-the-loop e2e scenario harness.

One test per `docs/SCENARIOS.md` entry. Each test:

1. Spawns `whatcanirun-mcp` over stdio via FastMCP `Client`.
2. Drives a Claude tool-use loop against the SAME question the
   scenario doc says to paste in chat.
3. Asserts three things — the same three the prose doc asks the
   user to check by hand:
   - Which tool(s) Claude called (the doc's "Expected tool path")
   - Envelope shape on the tool result (per spec/SHARED.md § Trust
     Contract — sources non-empty, weakest-link rule, etc.)
   - 1-2 soft keyword(s) in Claude's final natural-language reply
     (catches relay regressions that the server-side gate can't)

Marked `@pytest.mark.e2e` and skipped by default. See
`tests/e2e/conftest.py` for the fixture wiring.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from fastmcp import Client

from tests.e2e._claude_loop import ScenarioRun, run_scenario

pytestmark = pytest.mark.e2e


# Trust-envelope invariants. Identical contract to the release-gate
# `_assert_envelope` helper but reused here to assert against
# whatever envelopes Claude saw in its tool_result blocks. Keeping
# the two copies in sync is mechanical — both follow spec/SHARED.md
# § Trust Contract verbatim.
def _assert_envelope(envelope: dict[str, Any]) -> None:
    assert envelope.get("sources"), "envelope.sources missing or empty"
    breakdown = envelope.get("confidence_breakdown")
    assert isinstance(breakdown, dict) and breakdown, "confidence_breakdown empty"
    confidence = envelope.get("confidence")
    assert confidence is not None, "envelope.confidence missing"
    expected = min(breakdown.values())
    assert math.isclose(confidence, expected), (
        f"weakest-link rule violated: confidence={confidence} min(breakdown)={expected}"
    )
    assert envelope.get("verify_links"), "envelope.verify_links missing or empty"
    assert envelope.get("freshness"), "envelope.freshness missing or empty"


def _find_envelope_in(result: Any) -> dict[str, Any] | None:
    """Locate a trust_envelope in a tool result. CostCell and
    BudgetPlanRow rows nest one per row; FitCheckToolResponse and
    DeploymentComparison nest one at the top level."""
    if isinstance(result, dict):
        if "trust_envelope" in result:
            envelope = result["trust_envelope"]
            return envelope if isinstance(envelope, dict) else None
        # Nested under a Pydantic wrapper field name (e.g. fit_check
        # returns {"fit_result": ..., "trust_envelope": ...} — caught
        # above — but compare_deployment also nests envelopes under
        # its row fields).
        return None
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "trust_envelope" in first:
            envelope = first["trust_envelope"]
            return envelope if isinstance(envelope, dict) else None
    return None


# ============================================================ smoke


async def test_e2e_harness_smoke_loop_completes(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """Anchor test: a trivial question drives the Claude loop end-
    to-end. Proves the bridge wiring (FastMCP tool schemas →
    Anthropic tool schemas → tool_use roundtrip → end_turn) before
    any scenario-specific assertion piles on. If this passes and a
    specific scenario fails, the bug is in the scenario assertion,
    not the harness."""
    question = "Will Llama 3.3 70B fit on an H100 at fp16, batch 1, 4k context?"
    run: ScenarioRun = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question=question,
        model=claude_model,
    )
    assert run.turns >= 1
    assert run.stop_reason in {"end_turn", "max_tokens"}
    # At least one tool call must have happened; fit_check is the
    # obvious route. Don't pin to fit_check specifically here — that
    # specific assertion belongs to Scenario 2. The smoke just
    # asserts that *some* tool route happened.
    assert run.tool_calls, "smoke loop produced no tool calls"
