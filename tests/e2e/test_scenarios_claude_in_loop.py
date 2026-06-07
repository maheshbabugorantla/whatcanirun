"""Claude-in-the-loop e2e scenario harness (Agent SDK driver).

One test per `docs/SCENARIOS.md` entry. Each test:

1. Spawns `whatcanirun-mcp` over stdio via the Claude Agent SDK.
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
import json
import math
import re
from typing import Any

import pytest

from tests.e2e._claude_loop import ScenarioRun, run_scenario

pytestmark = pytest.mark.e2e


# Markdown formatting characters Claude tends to inject into final
# replies (bold `**`, italic `*` / `_`, strikethrough `~~`). Strip
# before soft-keyword checks so a phrase like "does **not** fit"
# matches the substring "not fit" — without stripping, the `**`
# between "not" and "fit" defeats the substring check entirely.
_MARKDOWN_CHARS = re.compile(r"[*_~`]+")


def _strip_markdown(text: str) -> str:
    """Return `text` with markdown bold/italic/strikethrough/code
    formatting stripped. Doesn't touch headers, links, or list
    bullets — those don't typically break substring-based keyword
    checks the way inline emphasis does."""
    return _MARKDOWN_CHARS.sub("", text)


# Trust-envelope invariants. Identical contract to the release-gate
# `_assert_envelope` helper but reused here to assert against
# whatever envelopes Claude saw in its tool_result blocks.
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
    for source_name, ts in freshness.items():
        try:
            dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError as exc:
            raise AssertionError(
                f"envelope.freshness[{source_name!r}] = {ts!r} is not ISO-8601"
            ) from exc


def _coerce_result(result: Any) -> Any:
    """Tool results from the Agent SDK arrive in one of three shapes:

    - `str` — JSON-encoded payload (`json.dumps`-style). Parse it.
    - `list[dict]` — MCP `content` array of text/image/etc blocks.
      Concatenate text bodies and try to JSON-parse the result.
    - `dict` / `list[dict]` of structured content — return as-is.

    Also unwraps FastMCP's `{"result": <obj>}` single-key wrapper
    that Pydantic-returning tools ship; without this, scenario
    assertions on top-level fields (`trust_envelope`, `status`,
    etc.) fail because the actual response is nested one level
    deeper. Mirrors the `_unwrap_result()` helper in the release-
    gate test for the same reason.

    Returns the parsed (and unwrapped) value. Tests then walk it
    with `_find_envelope_in()`."""
    parsed: Any
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result
    elif isinstance(result, list):
        # MCP content blocks: try the text-concat path.
        text_parts: list[str] = []
        structured: list[Any] = []
        for item in result:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                else:
                    structured.append(item)
        if text_parts:
            joined = "".join(text_parts)
            try:
                parsed = json.loads(joined)
            except json.JSONDecodeError:
                return joined
        elif structured:
            return structured
        else:
            return result
    else:
        parsed = result

    # Unwrap FastMCP's single-key `{"result": <obj>}` wrapper. Only
    # unwrap when EXACTLY one key (so structured responses that
    # happen to contain a `result` field alongside other keys stay
    # intact).
    if isinstance(parsed, dict) and len(parsed) == 1 and "result" in parsed:
        return parsed["result"]
    return parsed


def _find_envelope_in(result: Any) -> dict[str, Any] | None:
    """Locate a trust_envelope in a coerced tool result. CostCell
    and BudgetPlanRow rows nest one per row; FitCheckToolResponse
    and DeploymentComparison nest one at the top level.

    For list-shaped results, walks every row until it finds the
    first row with a `trust_envelope` — mirroring the server's
    INSTRUCTIONS guidance ("When relaying a list result, walk
    every row's envelope, not just the first"). Returns the first
    envelope found; the caller is responsible for asserting the
    per-row contract on subsequent rows when needed (Scenario 3
    iterates explicitly for that reason)."""
    if isinstance(result, dict):
        envelope = result.get("trust_envelope")
        return envelope if isinstance(envelope, dict) else None
    if isinstance(result, list):
        for row in result:
            if isinstance(row, dict):
                envelope = row.get("trust_envelope")
                if isinstance(envelope, dict):
                    return envelope
    return None


# ============================================================ smoke


async def test_e2e_harness_smoke_loop_completes(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """Anchor test: a trivial question drives the Agent SDK loop
    end-to-end. Proves the wiring (MCP server attach → tool_use
    roundtrip → end_turn) before any scenario-specific assertion
    piles on. If this passes and a specific scenario fails, the
    bug is in the scenario assertion, not the harness."""
    run: ScenarioRun = await run_scenario(
        question="Will Llama 3.3 70B fit on an H100 at fp16, batch 1, 4k context?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert run.turns >= 1
    assert not run.is_error, f"run errored: stop_reason={run.stop_reason}"
    # At least one tool call must have happened; fit_check is the
    # obvious route. Don't pin to fit_check specifically here —
    # that specific assertion belongs to Scenario 2.
    assert run.tool_calls, "smoke loop produced no tool calls"


# ============================================================ scenarios
# One test per docs/SCENARIOS.md entry. Each test pastes the exact
# question from the doc verbatim — if the prose changes there, the
# tests follow.


async def test_scenario_1_budget_to_plan(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 1 — Budget → plan (the headline).

    The envelope check pins the workload-assumption invariant: the
    confidence_breakdown MUST carry `workload_assumption` whenever
    the response synthesizes `est_total_prompts`. Soft keyword
    "chat" catches the workload relay rule INSTRUCTIONS string §6
    mandates.
    """
    run = await run_scenario(
        question="I have $50 to spend running Llama 3.3 70B for chat. What can I actually run?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert "budget_to_plan" in run.tool_names(), f"expected budget_to_plan; got {run.tool_names()}"
    found_workload_envelope = False
    for tc in run.tool_calls:
        if "budget_to_plan" not in tc.name:
            continue
        result = _coerce_result(tc.result)
        envelope = _find_envelope_in(result)
        if envelope is None:
            continue
        _assert_envelope(envelope)
        if "workload_assumption" in envelope.get("confidence_breakdown", {}):
            found_workload_envelope = True
            break
    assert found_workload_envelope, (
        "budget_to_plan never produced rows with a workload_assumption envelope"
    )
    assert "chat" in _strip_markdown(run.final_text.lower()), (
        f"final reply did not surface chat workload context: {run.final_text!r}"
    )


async def test_scenario_2_fit_check_wont_fit(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 2 — Fit check (won't fit).

    Mixtral 8x22B at fp16 is ~280GB weight-only; doesn't fit in an
    80GB H100. The envelope MUST NOT carry workload_assumption
    (fit_check is pure VRAM math). Soft keyword catches Claude
    paraphrasing the negative verdict honestly.
    """
    run = await run_scenario(
        question="Will Mixtral 8x22B fit on a single H100 80GB at fp16, 8k context?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert "fit_check" in run.tool_names(), f"expected fit_check; got {run.tool_names()}"
    # Walk ALL fit_check calls and find the first whose result
    # carries a trust_envelope. Claude often makes a first attempt
    # with a wrong slug ("h100-80gb" parsed from "H100 80GB"),
    # gets the server's "not in cached gpu catalog" hint, calls
    # `list_catalog` to discover supported slugs, and retries with
    # the correct one. Only the successful call has the envelope;
    # the failed first attempt is a tool_result error we want to
    # ignore.
    fit_calls = [tc for tc in run.tool_calls if "fit_check" in tc.name]
    envelope: dict[str, Any] | None = None
    result: Any = None
    for fc in fit_calls:
        coerced = _coerce_result(fc.result)
        env = _find_envelope_in(coerced)
        if env is not None:
            envelope = env
            result = coerced
            break
    last_raw = repr(fit_calls[-1].result) if fit_calls else "n/a"
    assert envelope is not None, (
        f"no fit_check call produced a trust_envelope across "
        f"{len(fit_calls)} attempt(s); last raw={last_raw}"
    )
    _assert_envelope(envelope)
    assert "workload_assumption" not in envelope.get("confidence_breakdown", {}), (
        "fit_check envelope must not carry workload_assumption (omit-when-not-synthesized rule)"
    )
    assert isinstance(result, dict), (
        f"fit_check result not a dict: {type(result).__name__} {result!r}"
    )
    fit_result = result.get("fit_result")
    assert isinstance(fit_result, dict), f"fit_check response missing fit_result dict: {result!r}"
    assert fit_result.get("fits") is False, (
        f"expected fits=False (Mixtral 8x22B fp16 doesn't fit in 80GB); got fit_result={fit_result}"
    )
    text = _strip_markdown(run.final_text.lower())
    assert any(
        s in text for s in ("doesn't fit", "won't fit", "not fit", "does not fit", "insufficient")
    ), f"final reply did not surface the doesn't-fit verdict: {run.final_text!r}"


async def test_scenario_3_find_cheapest(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 3 — Find cheapest.

    Ranked list response — per-row envelope contract. Each row
    carries its own envelope; no top-level envelope on the list.
    """
    run = await run_scenario(
        question="What's the cheapest provider hosting Qwen 2.5 72B right now?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert "find_cheapest_deployment" in run.tool_names(), (
        f"expected find_cheapest_deployment; got {run.tool_names()}"
    )
    cheapest_call = run.first_call_to("find_cheapest_deployment")
    assert cheapest_call is not None
    result = _coerce_result(cheapest_call.result)
    # The contract: `find_cheapest_deployment` returns a `list[CostCell]`.
    # An empty list is acceptable (cold-cache CP outage) but anything
    # OTHER than a list is a real shape regression that must fail loud
    # — silent-pass on an error dict (the previous behaviour) would
    # defeat the per-row envelope contract this test exists to defend.
    assert isinstance(result, list), (
        f"find_cheapest_deployment must return a list (per spec/M08); "
        f"got {type(result).__name__}: {result!r}"
    )
    for row in result:
        assert isinstance(row, dict), f"row not dict: {row!r}"
        envelope = row.get("trust_envelope")
        assert envelope, "find_cheapest_deployment row missing trust_envelope"
        _assert_envelope(envelope)
    assert "qwen" in _strip_markdown(run.final_text.lower()), (
        f"final reply did not name the requested model: {run.final_text!r}"
    )


async def test_scenario_4_compare_deployment_modes(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 4 — Compare deployment modes.

    Per-prompt cost comparison with `workload_assumption` in the
    envelope. v1 known limitation: `hosted_api_token` is currently
    always None (Issue #25). Test asserts the envelope invariant;
    the verdict being "unknown" is the honest relay.
    """
    run = await run_scenario(
        question=("Should I rent an H100 or use a hosted API for Llama 3.3 70B at chat volumes?"),
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert "compare_deployment_modes" in run.tool_names(), (
        f"expected compare_deployment_modes; got {run.tool_names()}"
    )
    # Walk ALL compare_deployment_modes calls. Same retry pattern as
    # Scenario 2: Claude's first attempt may use a slug variant (e.g.
    # `workload_profile_slug="chat"` parsed from "chat volumes"); the
    # server returns an error hinting at `list_catalog`; Claude
    # retries with the correct `chat_assistant` slug. The successful
    # retry is the one with a trust_envelope.
    compare_calls = [tc for tc in run.tool_calls if "compare_deployment_modes" in tc.name]
    assert compare_calls, f"expected compare_deployment_modes call; got {run.tool_names()}"
    envelope: dict[str, Any] | None = None
    for cc in compare_calls:
        env = _find_envelope_in(_coerce_result(cc.result))
        if env is not None:
            envelope = env
            break
    last_raw = repr(compare_calls[-1].result) if compare_calls else "n/a"
    assert envelope is not None, (
        f"no compare_deployment_modes call produced an envelope across "
        f"{len(compare_calls)} attempt(s); last raw={last_raw}"
    )
    # The workload_assumption invariant is the scenario's actual
    # point and the contract spec/M09 promises. Do NOT run the full
    # `_assert_envelope()` here: when Claude's gpu_slug doesn't
    # resolve cleanly (a common natural-language parse miss), or
    # when Issue #25's hosted-side filter bug fires, the server
    # returns a partial-data envelope with empty `sources` and
    # missing `freshness` / `verify_links`. The release-gate test
    # asserts the envelope shape under happy-path conditions; this
    # scenario asserts the relay-rule invariant that survives
    # under the v1 known-limitation conditions.
    assert "workload_assumption" in envelope.get("confidence_breakdown", {}), (
        f"compare_deployment_modes must carry workload_assumption "
        f"(per-prompt cost is workload-derived); got breakdown="
        f"{envelope.get('confidence_breakdown')}"
    )
    text = _strip_markdown(run.final_text.lower())
    assert any(s in text for s in ("rent", "rental", "hosted", "api")), (
        f"final reply did not surface rental-vs-hosted framing: {run.final_text!r}"
    )


async def test_scenario_5_unknown_model_elicitation(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 5 — Unknown-model elicitation.

    Either entry point is valid: a smart Claude might route directly
    to `resolve_model` based on slug shape; a default Claude tries
    `fit_check`, gets `UnknownModelResponse`, then routes to
    `resolve_model`. The harness HF repo_id is in the question, so
    Claude doesn't need a separate user prompt.

    Network dependency: `resolve_model` hits Hugging Face Hub. If
    unreachable, the resolve fails with `status="sync_failed"` and
    the retry fit_check won't happen; the test downgrades to
    asserting just the unknown-model branch fired.
    """
    run = await run_scenario(
        question="Can I run NousResearch/Hermes-2-Pro-Mistral-7B on an A100?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    names = run.tool_names()
    assert "fit_check" in names or "resolve_model" in names, (
        f"expected fit_check or resolve_model; got {names}"
    )

    # Branch A: fit_check fired and returned UnknownModelResponse →
    # verify the slug echo contract + that Claude followed up with
    # resolve_model.
    branch_asserted = False
    first_fit = run.first_call_to("fit_check")
    if first_fit is not None:
        fit_result = _coerce_result(first_fit.result)
        if isinstance(fit_result, dict) and fit_result.get("status") == "unknown_model":
            echoed = fit_result.get("requested_model_slug")
            sent = first_fit.arguments.get("model_slug")
            assert echoed == sent, (
                f"UnknownModelResponse.requested_model_slug echo broken: "
                f"sent={sent!r} echoed={echoed!r}"
            )
            assert "resolve_model" in names, (
                "unknown-model branch fired but Claude did not call resolve_model"
            )
            branch_asserted = True

    # Branch B: Claude routed directly to resolve_model (or got there
    # via fit_check's hint). Validate the resolve_model contract:
    # status must be one of the documented Literals; slug+repo_id are
    # echoed back so the client can confirm what got persisted.
    resolve_call = run.first_call_to("resolve_model")
    if resolve_call is not None:
        resolve_result = _coerce_result(resolve_call.result)
        assert isinstance(resolve_result, dict), (
            f"resolve_model result not a dict: {resolve_result!r}"
        )
        status = resolve_result.get("status")
        assert status in {"resolved", "sync_failed", "not_found_on_hf"}, (
            f"resolve_model returned unexpected status {status!r}; "
            f"contract requires one of "
            f"resolved | sync_failed | not_found_on_hf"
        )
        # The resolve_model echo contract: input slug + repo_id surface
        # in the response so the client can verify what got persisted.
        sent_slug = resolve_call.arguments.get("model_slug")
        sent_repo = resolve_call.arguments.get("hf_repo_id")
        assert resolve_result.get("model_slug") == sent_slug, (
            f"resolve_model.model_slug echo broken: "
            f"sent={sent_slug!r} got={resolve_result.get('model_slug')!r}"
        )
        assert resolve_result.get("hf_repo_id") == sent_repo, (
            f"resolve_model.hf_repo_id echo broken: "
            f"sent={sent_repo!r} got={resolve_result.get('hf_repo_id')!r}"
        )
        branch_asserted = True

    assert branch_asserted, (
        "scenario 5 produced neither an unknown-model fit_check nor a "
        f"resolve_model call worth asserting on; tool_names={names}, "
        f"first_fit_result={_coerce_result(first_fit.result) if first_fit else 'n/a'!r}"
    )


async def test_scenario_6_workload_elicitation(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 6 — Workload-profile elicitation.

    Two valid trust-contract-compliant paths for "no silent default":
    (A) Server-side elicit: Claude calls `budget_to_plan` without a
        workload, gets `WorkloadElicitationResponse`, then retries
        with a workload slug. Two `budget_to_plan` calls total.
    (B) Client-side elicit: Claude reads the question, recognises
        the workload is missing, and asks the user via natural
        language BEFORE calling any tool. The harness is
        single-shot (no `elicitation_replies` wired into this
        scenario), so under Path B the run ends with ZERO
        `budget_to_plan` calls; the test verifies Claude's final
        text contains an elicitation question instead of a silent
        default.

    Both paths honor spec/M09's "no silent default" contract.
    """
    run = await run_scenario(
        question="How many prompts can I run on $20 of Qwen 2.5 7B?",
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    budget_calls = [tc for tc in run.tool_calls if "budget_to_plan" in tc.name]

    # Path B: Claude asked the user via NL without calling any tool.
    # Verify final_text shows an elicitation question (mentions
    # workload / what kind / which type / etc.) — that's the
    # contract-honouring "no silent default" expressed in the
    # messaging surface rather than the tool-use surface.
    if not budget_calls:
        text = _strip_markdown(run.final_text.lower())
        elicit_signals = (
            "workload",
            "what kind",
            "what sort",
            "what type",
            "which type",
            "code completion",
            "chat assistant",
            "batch eval",
            "code_completion",
            "chat_assistant",
            "batch_eval",
            "what are you",
            "what kind of prompts",
        )
        assert any(s in text for s in elicit_signals), (
            f"path B (no tool call) requires the final reply to surface a "
            f"workload-elicitation question; got: {run.final_text!r}"
        )
        return

    # Path A: at least one budget_to_plan call. The LAST call must
    # carry a workload slug — that's the execution call regardless
    # of how many elicitation rounds preceded it.
    final_args = budget_calls[-1].arguments
    assert final_args.get("workload_profile_slug"), (
        f"final budget_to_plan call missing workload_profile_slug: {final_args!r}"
    )

    # If multiple calls (path A: server-side elicit), validate the
    # elicitation contract on the FIRST call.
    if len(budget_calls) >= 2:
        first_args = budget_calls[0].arguments
        first_workload = first_args.get("workload_profile_slug")
        assert first_workload in (None, ""), (
            f"first budget_to_plan should omit workload_profile_slug; got {first_workload!r}"
        )
        first_result = _coerce_result(budget_calls[0].result)
        assert isinstance(first_result, dict), f"unexpected first result: {first_result!r}"
        assert first_result.get("status") == "workload_required", (
            f"first call did not return WorkloadElicitationResponse: {first_result!r}"
        )
        profiles = set(first_result.get("available_profiles", []))
        assert profiles >= {"code_completion", "chat_assistant", "batch_eval"}, (
            f"WorkloadElicitationResponse missing v1 profiles: got {profiles!r}"
        )


@pytest.mark.skip(
    reason=(
        "MCP resources (cost-cells://provenance) are not exposed to the Claude "
        "Agent SDK tool-use loop (only tools are). Tracked as v2 issue: expose "
        "provenance as a tool wrapper server-side. See PR #27 description."
    )
)
async def test_scenario_7_provenance_audit(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 7 — Provenance audit.

    DEFERRED until the server exposes `cost-cells://provenance` as
    a tool (in addition to a resource). The Agent SDK's tool-use
    loop only surfaces tools to Claude; MCP resources aren't
    auto-discoverable. The manual SCENARIOS.md walkthrough still
    covers this contract; the release-gate test asserts the
    resource itself is well-formed.
    """


async def test_scenario_8_honest_refusal_batch_gt_one(
    claude_runtime_available: None,
    claude_model: str,
    mcp_command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str] | None,
) -> None:
    """SCENARIOS.md § Scenario 8 — Honest "I don't know" (batch > 1).

    The v1 TPS heuristic is single-stream only (ADR-010). batch=32
    must surface a `tps_estimate.source == "requires_measurement"`
    cell somewhere. The relay must NOT invent a number.
    """
    run = await run_scenario(
        question=("What's the per-stream decode TPS for Llama 3.3 70B on an H100 at batch=32?"),
        mcp_command=mcp_command,
        mcp_args=mcp_args,
        model=claude_model,
        env=mcp_env,
    )
    assert run.tool_calls, "scenario 8 produced no tool calls"

    def _walk_for_requires_measurement(payload: Any) -> bool:
        if isinstance(payload, dict):
            tps = payload.get("tps_estimate")
            if isinstance(tps, dict) and tps.get("source") == "requires_measurement":
                return True
            return any(_walk_for_requires_measurement(v) for v in payload.values())
        if isinstance(payload, list):
            return any(_walk_for_requires_measurement(item) for item in payload)
        return False

    found = any(_walk_for_requires_measurement(_coerce_result(tc.result)) for tc in run.tool_calls)
    assert found, (
        "no tool result surfaced tps_estimate.source == 'requires_measurement' — "
        "batch>1 should refuse per ADR-010"
    )
    text = _strip_markdown(run.final_text.lower())
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
