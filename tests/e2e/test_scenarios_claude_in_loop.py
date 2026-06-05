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

import datetime as dt
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
    freshness = envelope.get("freshness")
    assert isinstance(freshness, dict) and freshness, "envelope.freshness missing or empty"
    # Mirror the release-gate parser: every freshness entry must be
    # a parseable ISO-8601 datetime. A bare truthy check (the
    # original) would let a malformed timestamp slip past, which is
    # the relay-regression class this harness exists to catch.
    for source_name, ts in freshness.items():
        try:
            dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError as exc:
            raise AssertionError(
                f"envelope.freshness[{source_name!r}] = {ts!r} is not ISO-8601"
            ) from exc


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


# ============================================================ scenarios
# One test per docs/SCENARIOS.md entry. Each test pastes the exact
# question from the doc verbatim — if the prose changes there, the
# tests follow. Assertions match the doc's "What to verify"
# checklist projected onto programmatically-checkable predicates.


async def test_scenario_1_budget_to_plan(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 1 — Budget → plan (the headline).

    Tool path: `budget_to_plan` with `workload_profile_slug`. Claude
    SHOULD infer `chat_assistant` from "for chat" in the question;
    `elicitation_replies` carries the slug as a fallback in case
    Claude calls without it and gets a `WorkloadElicitationResponse`.
    The envelope check pins the workload-assumption invariant: the
    confidence_breakdown MUST carry `workload_assumption` whenever
    the response synthesizes `est_total_prompts` (CLAUDE.md
    "Confidence domain" rule).
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question=("I have $50 to spend running Llama 3.3 70B for chat. What can I actually run?"),
        model=claude_model,
        elicitation_replies=["chat_assistant"],
    )
    assert "budget_to_plan" in run.tool_names(), f"expected budget_to_plan; got {run.tool_names()}"
    # Find the budget_to_plan call whose result has rows with
    # workload_assumption envelopes. Tolerates a prior elicitation
    # call returning WorkloadElicitationResponse (no envelope).
    found_workload_envelope = False
    for tc in run.tool_calls:
        if tc.name != "budget_to_plan":
            continue
        if not isinstance(tc.result, list) or not tc.result:
            continue
        envelope = _find_envelope_in(tc.result)
        if envelope is None:
            continue
        _assert_envelope(envelope)
        if "workload_assumption" in envelope.get("confidence_breakdown", {}):
            found_workload_envelope = True
            break
    assert found_workload_envelope, (
        "budget_to_plan never produced rows with a workload_assumption envelope"
    )
    # Soft keyword: Claude should name the workload assumption (the
    # INSTRUCTIONS string rule 6 requires it). "chat" matches both
    # the user's phrasing and the slug.
    assert "chat" in run.final_text.lower(), (
        f"final reply did not surface chat workload context: {run.final_text!r}"
    )


async def test_scenario_2_fit_check_wont_fit(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 2 — Fit check (won't fit).

    Mixtral 8x22B at fp16 is ~280GB weight-only; doesn't fit in an
    80GB H100. The envelope MUST NOT carry workload_assumption
    (fit_check is pure VRAM math). Soft keyword catches Claude
    paraphrasing the negative verdict honestly — "doesn't fit",
    "won't fit", "insufficient", etc.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question=("Will Mixtral 8x22B fit on a single H100 80GB at fp16, 8k context?"),
        model=claude_model,
    )
    assert "fit_check" in run.tool_names(), f"expected fit_check; got {run.tool_names()}"
    fit_call = run.first_call_to("fit_check")
    assert fit_call is not None
    envelope = _find_envelope_in(fit_call.result)
    assert envelope is not None, "fit_check result missing trust_envelope"
    _assert_envelope(envelope)
    assert "workload_assumption" not in envelope.get("confidence_breakdown", {}), (
        "fit_check envelope must not carry workload_assumption (omit-when-not-synthesized rule)"
    )
    # Verify fit_result.fits == False unconditionally. The previous
    # version guarded both isinstance checks, which would silently
    # pass when the response shape regressed (e.g. fit_result
    # missing or returned as something other than a dict). Scenario
    # 2's whole point is catching a fits=True relay where fits=False
    # is correct — make the shape assertion loud.
    assert isinstance(fit_call.result, dict), (
        f"fit_check result not a dict: {type(fit_call.result).__name__} {fit_call.result!r}"
    )
    fit_result = fit_call.result.get("fit_result")
    assert isinstance(fit_result, dict), (
        f"fit_check response missing fit_result dict: {fit_call.result!r}"
    )
    assert fit_result.get("fits") is False, (
        f"expected fits=False (Mixtral 8x22B fp16 doesn't fit in 80GB); got fit_result={fit_result}"
    )
    # Soft keyword: any honest paraphrase of "no".
    text = run.final_text.lower()
    assert any(s in text for s in ("doesn't fit", "won't fit", "not fit", "insufficient")), (
        f"final reply did not surface the doesn't-fit verdict: {run.final_text!r}"
    )


async def test_scenario_3_find_cheapest(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 3 — Find cheapest.

    Ranked list response — per-row envelope contract. Each row
    carries its own envelope; no top-level envelope on the list.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question="What's the cheapest provider hosting Qwen 2.5 72B right now?",
        model=claude_model,
    )
    assert "find_cheapest_deployment" in run.tool_names(), (
        f"expected find_cheapest_deployment; got {run.tool_names()}"
    )
    cheapest_call = run.first_call_to("find_cheapest_deployment")
    assert cheapest_call is not None
    # Tolerate an empty list (cold-cache CP outage) — but if rows
    # exist, every one must carry an envelope.
    if isinstance(cheapest_call.result, list) and cheapest_call.result:
        for row in cheapest_call.result:
            assert isinstance(row, dict), f"row not dict: {row!r}"
            envelope = row.get("trust_envelope")
            assert envelope, "find_cheapest_deployment row missing trust_envelope"
            _assert_envelope(envelope)
    # Soft keyword: model name must surface in the relay.
    assert "qwen" in run.final_text.lower(), (
        f"final reply did not name the requested model: {run.final_text!r}"
    )


async def test_scenario_4_compare_deployment_modes(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 4 — Compare deployment modes.

    Per-prompt cost comparison with `workload_assumption` in the
    envelope. v1 known limitation: `hosted_api_token` is currently
    always None (Issue #25 — `compare_deployment_modes` filtering
    bug). Test still asserts the envelope invariant; the verdict
    being "unknown" is the honest relay for the missing side.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question=("Should I rent an H100 or use a hosted API for Llama 3.3 70B at chat volumes?"),
        model=claude_model,
        elicitation_replies=["chat_assistant"],
    )
    assert "compare_deployment_modes" in run.tool_names(), (
        f"expected compare_deployment_modes; got {run.tool_names()}"
    )
    compare_call = run.first_call_to("compare_deployment_modes")
    assert compare_call is not None
    envelope = _find_envelope_in(compare_call.result)
    assert envelope is not None, "compare_deployment_modes result missing envelope"
    _assert_envelope(envelope)
    assert "workload_assumption" in envelope.get("confidence_breakdown", {}), (
        "compare_deployment_modes must carry workload_assumption "
        "(per-prompt cost is workload-derived)"
    )
    # Soft keyword: must mention one side of the comparison in the
    # natural-language relay.
    text = run.final_text.lower()
    assert any(s in text for s in ("rent", "rental", "hosted", "api")), (
        f"final reply did not surface rental-vs-hosted framing: {run.final_text!r}"
    )


async def test_scenario_5_unknown_model_elicitation(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 5 — Unknown-model elicitation.

    Two-call flow: Claude tries fit_check with a slug it derives →
    server returns `UnknownModelResponse` echoing the slug verbatim
    → Claude routes to `resolve_model` to persist (slug, repo_id)
    → retries fit_check. The HF repo_id is already in the question
    so Claude should not need a separate user prompt; the
    elicitation reply is a safety net.

    Network dependency: `resolve_model` hits Hugging Face Hub. If
    that's unreachable in the test environment, the resolve will
    fail with `status="sync_failed"` and the retry fit_check won't
    happen; the test downgrades to asserting just the unknown-
    model branch fired.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question="Can I run NousResearch/Hermes-2-Pro-Mistral-7B on an A100?",
        model=claude_model,
        elicitation_replies=[
            "The Hugging Face repo_id is NousResearch/Hermes-2-Pro-Mistral-7B",
        ],
    )
    # Either entry point is valid for the unknown-model branch: a
    # smart Claude might recognise the HF-style slug as unknown and
    # route DIRECTLY to resolve_model first; a default Claude tries
    # fit_check and gets UnknownModelResponse → then routes to
    # resolve_model. Both are trust-contract-compliant. Pinning to
    # fit_check exclusively flakes against the smart-routing path.
    names = run.tool_names()
    assert "fit_check" in names or "resolve_model" in names, (
        f"expected fit_check or resolve_model; got {names}"
    )
    # If fit_check WAS called and returned UnknownModelResponse,
    # verify the echo-verbatim contract: the server NEVER
    # canonicalises the slug, it echoes whatever the client sent
    # (per `dispatch.py:68`). This is the most regression-prone
    # surface in the unknown-model flow.
    first_fit = run.first_call_to("fit_check")
    if (
        first_fit is not None
        and isinstance(first_fit.result, dict)
        and first_fit.result.get("status") == "unknown_model"
    ):
        echoed = first_fit.result.get("requested_model_slug")
        sent = first_fit.arguments.get("model_slug")
        assert echoed == sent, (
            f"UnknownModelResponse.requested_model_slug echo broken: "
            f"sent={sent!r} echoed={echoed!r}"
        )
        # When fit_check returned unknown_model, Claude MUST follow
        # up with resolve_model — the elicitation_reply gives it the
        # repo_id verbatim, so there's no excuse to drop the flow.
        assert "resolve_model" in names, (
            "unknown-model branch fired but Claude did not call resolve_model"
        )


async def test_scenario_6_workload_elicitation(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 6 — Workload-profile elicitation.

    Question deliberately omits a workload profile. Server returns
    `WorkloadElicitationResponse` listing the three v1 profiles
    (code_completion, chat_assistant, batch_eval). Claude relays
    the choices, the harness replies with `chat_assistant`, Claude
    retries `budget_to_plan` with the slug — proving the
    elicitation flow round-trips through the LLM client.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question="How many prompts can I run on $20 of Qwen 2.5 7B?",
        model=claude_model,
        elicitation_replies=["chat_assistant"],
    )
    # Two valid trust-contract-compliant paths for "no silent default":
    # (A) Server-side elicitation: Claude calls budget_to_plan
    #     blindly, gets WorkloadElicitationResponse, asks user,
    #     retries with slug. Two budget_to_plan calls total.
    # (B) Client-side elicitation: Claude reads the question, sees
    #     no workload was specified, asks user via natural language
    #     BEFORE any tool call, then calls budget_to_plan once with
    #     the slug. One budget_to_plan call total.
    # Both honor spec/M09's "no silent default" contract. Pinning to
    # path A exclusively flakes against smarter Claude routing.
    budget_calls = [tc for tc in run.tool_calls if tc.name == "budget_to_plan"]
    assert budget_calls, "scenario 6 produced no budget_to_plan calls"

    # The LAST budget_to_plan call must carry a workload slug — that's
    # the execution call regardless of which path was taken.
    final_args = budget_calls[-1].arguments
    assert final_args.get("workload_profile_slug"), (
        f"final budget_to_plan call missing workload_profile_slug: {final_args!r}"
    )

    # If path A was taken (multiple calls + first one blind), validate
    # the server-side elicitation contract: status, profiles, no
    # silent default.
    if len(budget_calls) >= 2:
        first_args = budget_calls[0].arguments
        first_workload = first_args.get("workload_profile_slug")
        assert first_workload in (None, ""), (
            f"first budget_to_plan should omit workload_profile_slug; got {first_workload!r}"
        )
        first_result = budget_calls[0].result
        assert isinstance(first_result, dict), f"unexpected first result: {first_result!r}"
        assert first_result.get("status") == "workload_required", (
            f"first call did not return WorkloadElicitationResponse: {first_result!r}"
        )
        profiles = set(first_result.get("available_profiles", []))
        assert profiles >= {"code_completion", "chat_assistant", "batch_eval"}, (
            f"WorkloadElicitationResponse missing v1 profiles: got {profiles!r}"
        )


async def test_scenario_7_provenance_audit(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 7 — Provenance audit.

    Claude must actually read the `cost-cells://provenance`
    resource rather than relay prior context. The harness exposes
    the resource via the `read_mcp_resource` pseudo-tool so Claude
    can route to it without being pre-prompted on the URI.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question="Show me the sources behind your throughput estimate.",
        model=claude_model,
    )
    assert "read_mcp_resource" in run.tool_names(), (
        f"expected read_mcp_resource; got {run.tool_names()}"
    )
    resource_call = run.first_call_to("read_mcp_resource")
    assert resource_call is not None
    assert resource_call.arguments.get("uri") == "cost-cells://provenance"
    # The resource payload is a JSON string; the three v1 sources
    # must appear by lowercase `name` field.
    payload = resource_call.result
    assert isinstance(payload, str) and payload, "provenance resource returned empty"
    for name in ("computeprices", "huggingface", "artificial_analysis"):
        assert name in payload, f"provenance JSON missing source name {name!r}"
    # Soft keyword on Claude's relay: at least one source surfaced
    # in natural-language form.
    text = run.final_text.lower()
    assert any(
        s in text for s in ("computeprices", "compute prices", "hugging face", "huggingface")
    ), f"final reply did not name any concrete source: {run.final_text!r}"


async def test_scenario_8_honest_refusal_batch_gt_one(
    anthropic_client: AsyncAnthropic,
    mcp_client: Client[Any],
    claude_model: str,
) -> None:
    """SCENARIOS.md § Scenario 8 — Honest "I don't know" (batch > 1).

    The v1 TPS heuristic is single-stream only (ADR-010). batch=32
    must surface a `tps_estimate.source == "requires_measurement"`
    cell somewhere in the response. The relay must NOT invent a
    number — soft keyword catches Claude doing the honest refusal.
    """
    run = await run_scenario(
        anthropic_client=anthropic_client,
        fastmcp_client=mcp_client,
        question=("What's the per-stream decode TPS for Llama 3.3 70B on an H100 at batch=32?"),
        model=claude_model,
    )
    # Any cost-cell-producing tool (find_cheapest_deployment,
    # compare_deployment_modes, budget_to_plan) is acceptable as
    # the entry point — they all route batch>1 through the same
    # tps_estimator.
    assert run.tool_calls, "scenario 8 produced no tool calls"

    def _walk_for_requires_measurement(payload: Any) -> bool:
        """The provenance flag lives at
        `CostCell.tps_estimate.source` (nested), not at the top
        level. Walk the tool result looking for it on any row."""
        if isinstance(payload, dict):
            tps = payload.get("tps_estimate")
            if isinstance(tps, dict) and tps.get("source") == "requires_measurement":
                return True
            for v in payload.values():
                if _walk_for_requires_measurement(v):
                    return True
        elif isinstance(payload, list):
            for item in payload:
                if _walk_for_requires_measurement(item):
                    return True
        return False

    found_requires_measurement = any(
        _walk_for_requires_measurement(tc.result) for tc in run.tool_calls
    )
    assert found_requires_measurement, (
        "no tool result surfaced tps_estimate.source == 'requires_measurement' — "
        "batch>1 should refuse per ADR-010"
    )
    # Soft keyword: Claude must surface the refusal honestly. Any of
    # these phrasings qualifies; an invented TPS number does not.
    text = run.final_text.lower()
    honest_phrases = (
        "requires_measurement",
        "requires measurement",
        "single-stream",
        "single stream",
        "batch=1",
        "batch 1",
        "cannot",
        "can't estimate",
        "don't have",
        "do not have",
    )
    assert any(s in text for s in honest_phrases), (
        f"final reply did not surface honest refusal: {run.final_text!r}"
    )
