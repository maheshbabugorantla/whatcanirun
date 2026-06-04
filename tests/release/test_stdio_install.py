"""M12 Slice C — release-gate stdio install test.

Drives the installed `whatcanirun-mcp` binary over real stdio
with FastMCP's `Client` + `StdioTransport`, walks the tool /
resource / prompt battery, and asserts the trust-envelope
invariants on every numerical response.

Marked `@pytest.mark.release` so it does NOT run in the default
`pytest -q` CI suite — it spawns a real subprocess, hits live
ComputePrices + Hugging Face upstreams, and takes ~30-60s on a
cold cache. The host-uv install script (`scripts/install_host_uv.sh`)
invokes it via `pytest -m release` as the final gate after
`uv sync` + prefetch.

What the gate proves vs. existing in-process tests:

- `tests/integration/test_mcp_protocol_flows.py` exercises the
  same tools via `Client(mcp)` — an in-process MCP client
  talking to the imported FastMCP instance. That catches API
  shape and routing bugs but does NOT prove the installed
  `whatcanirun-mcp` script entry point works, the stdio
  framing roundtrips, or the lazy-load path through
  `load_runtime_deps()` succeeds on a real on-disk cache.
- This gate spawns `uv run whatcanirun-mcp` as a subprocess
  and talks JSON-RPC over its stdin/stdout. A regression in
  pyproject's `[project.scripts]` registration, the
  module-level FastMCP wiring, or the stdio transport itself
  fails this test where the in-process battery wouldn't.

Trust-envelope invariants asserted on every numerical response:

- `trust_envelope` field present and well-formed;
- `confidence == min(confidence_breakdown.values())`
  (the weakest-link rule per spec/SHARED.md § Trust Contract);
- `workload_assumption` present iff the response synthesized a
  workload-derived count;
- `verify_links` non-empty so users can audit upstream;
- `freshness` per source is a real ISO-8601 datetime.

Catalog-only responses (`list_catalog`, `resolve_model`) are
envelope-exempt by design (spec/M09 § "Envelope-exempt tools").
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

pytestmark = pytest.mark.release


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def stdio_client() -> StdioTransport:
    """`uv run --directory <repo> whatcanirun-mcp` over stdio.

    Function-scoped: each test spawns its own subprocess. Module-
    scoping with keep_alive=True triggered "session closed
    unexpectedly" on the second test against this server (likely
    a stdio reconnect-after-context-exit edge case). The on-disk
    cache is shared across subprocesses so the cold-cache penalty
    only hits the first test that touches any given upstream,
    not every test."""
    return StdioTransport(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), "whatcanirun-mcp"],
    )


def _unwrap_result(result: Any) -> Any:
    """Pull the actual response payload out of a FastMCP
    `CallToolResult.structured_content`. FastMCP wraps Pydantic-
    returning tool outputs in a top-level `{"result": ...}` key
    so the structured-content stays a JSON object even when the
    tool returns a list or scalar. Tools that already return a
    plain dict (e.g. `list_catalog`) ship the dict directly with
    no `result` wrapper. Handling both keeps callers from caring
    which convention any given tool follows."""
    sc = result.structured_content
    assert sc is not None, "tool response missing structured_content"
    if isinstance(sc, dict) and "result" in sc and len(sc) == 1:
        return sc["result"]
    return sc


def _assert_envelope(envelope: dict[str, Any]) -> None:
    """Per-spec trust-envelope invariants. Failing any one of
    these means the response is shipping a number the LLM client
    can't safely surface to a user — the entire point of the
    server."""
    assert "sources" in envelope and len(envelope["sources"]) > 0, "envelope.sources empty"
    assert "confidence_breakdown" in envelope, "envelope.confidence_breakdown missing"
    breakdown = envelope["confidence_breakdown"]
    assert isinstance(breakdown, dict) and len(breakdown) > 0, "breakdown empty"
    assert "confidence" in envelope, "envelope.confidence missing"
    expected = min(breakdown.values())
    assert math.isclose(envelope["confidence"], expected), (
        f"confidence {envelope['confidence']} != min(breakdown)={expected} — "
        "weakest-link rule violated"
    )
    assert envelope.get("verify_links"), "envelope.verify_links missing or empty"
    assert "freshness" in envelope, "envelope.freshness missing"
    # Each freshness entry must be a parseable ISO-8601 datetime.
    for source_name, ts in envelope["freshness"].items():
        try:
            dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError as exc:
            raise AssertionError(
                f"envelope.freshness[{source_name!r}] = {ts!r} is not ISO-8601"
            ) from exc


# ============================================================ tests


async def test_release_initialize_handshake_succeeds(
    stdio_client: StdioTransport,
) -> None:
    """The MCP `initialize` handshake completes within the
    transport's default timeout. Any failure here means the
    installed script entry point or the FastMCP wiring is broken
    — every other gate test would also fail with a less
    actionable error, so this lands first for fast-fail
    diagnostics."""
    async with Client(stdio_client) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        # Every M09 tool must be advertised — a missing tool means a
        # registration regression in server.py's decorator block.
        assert names >= {
            "fit_check",
            "find_cheapest_deployment",
            "compare_deployment_modes",
            "budget_to_plan",
            "list_catalog",
            "resolve_model",
        }


async def test_release_fit_check_returns_well_formed_envelope(
    stdio_client: StdioTransport,
) -> None:
    """`fit_check` is the simplest numerical tool — no upstream
    pricing, just pure-math VRAM arithmetic. If its envelope is
    malformed, no other numerical tool's will be right either."""
    async with Client(stdio_client) as client:
        result = await client.call_tool(
            "fit_check",
            {
                "model_slug": "qwen-2-5-7b",
                "gpu_slug": "h100",
                "quant_slug": "fp16",
                "tp_size": 1,
                "batch_size": 1,
                "context_length": 4096,
            },
        )
        data = _unwrap_result(result)
        assert isinstance(data, dict)
        # FitCheckToolResponse: fit_result + trust_envelope.
        fit_result = data["fit_result"]
        # `fits` is the verdict; `sufficiency_caveat` is always
        # populated even on a True verdict (spec/M06 § FitResult).
        assert "fits" in fit_result
        assert fit_result.get("sufficiency_caveat"), "sufficiency_caveat empty on FitResult"
        envelope = data["trust_envelope"]
        _assert_envelope(envelope)
        # fit_check has no workload component → workload_assumption
        # must be absent per the omit-when-not-synthesized rule.
        assert "workload_assumption" not in envelope["confidence_breakdown"]


async def test_release_find_cheapest_returns_per_row_envelopes(
    stdio_client: StdioTransport,
) -> None:
    """`find_cheapest_deployment` returns `list[CostCell]`. Per
    the per-row-envelope contract (server INSTRUCTIONS string),
    every row carries its own envelope and there is no top-level
    envelope for the list. A regression that wraps the list in a
    top-level envelope breaks LLM clients that walk row envelopes."""
    async with Client(stdio_client) as client:
        result = await client.call_tool(
            "find_cheapest_deployment",
            {"model_slug": "qwen-2-5-7b", "quant_slug": "fp16", "top_n": 5},
        )
        rows = _unwrap_result(result)
        # The list may be empty on a cold-cache CP outage; the test
        # still asserts the response is a list-shape, not an error.
        assert isinstance(rows, list), f"expected list, got {type(rows).__name__}"
        for row in rows:
            envelope = row.get("trust_envelope")
            assert envelope, "find_cheapest row missing trust_envelope"
            _assert_envelope(envelope)


async def test_release_budget_to_plan_workload_assumption_present(
    stdio_client: StdioTransport,
) -> None:
    """`budget_to_plan` with a workload slug derives
    `est_total_prompts` from the workload profile, so the
    response envelope MUST carry the `workload_assumption`
    domain (CLAUDE.md invariant + spec/M08 § BudgetPlanRow).
    The corollary — that calls without workload synthesis OMIT
    the key — is covered by the fit_check test above."""
    async with Client(stdio_client) as client:
        result = await client.call_tool(
            "budget_to_plan",
            {
                "budget_usd": 100.0,
                "model_slug": "qwen-2-5-7b",
                "workload_profile_slug": "chat_assistant",
                "top_n": 3,
            },
        )
        rows = _unwrap_result(result)
        assert isinstance(rows, list)
        for row in rows:
            envelope = row.get("trust_envelope")
            assert envelope, "budget_to_plan row missing trust_envelope"
            _assert_envelope(envelope)
            assert "workload_assumption" in envelope["confidence_breakdown"], (
                "budget_to_plan with workload_profile_slug must populate "
                "workload_assumption in confidence_breakdown"
            )
            # And `est_total_prompts` must be present per spec/M08.
            assert "est_total_prompts" in row
            assert "est_wallclock_minutes" in row


async def test_release_compare_deployment_modes_both_modes_present(
    stdio_client: StdioTransport,
) -> None:
    """`compare_deployment_modes` returns one envelope-bearing
    payload covering both `cloud_gpu_rental` and
    `hosted_api_token` for the same op-point. A regression that
    drops one mode would silently strip half the trust contract."""
    async with Client(stdio_client) as client:
        result = await client.call_tool(
            "compare_deployment_modes",
            {
                "model_slug": "qwen-2-5-7b",
                "gpu_slug": "h100",
                "quant_slug": "fp16",
                "batch_size": 1,
                "context_length": 4096,
                "workload_profile_slug": "chat_assistant",
            },
        )
        data = _unwrap_result(result)
        assert isinstance(data, dict)
        envelope = data.get("trust_envelope")
        assert envelope, "compare_deployment_modes missing trust_envelope"
        _assert_envelope(envelope)
        # workload_assumption populated because per-prompt cost is
        # workload-derived (build_deployment_comparison_envelope).
        assert "workload_assumption" in envelope["confidence_breakdown"]


async def test_release_list_catalog_returns_facts_no_envelope(
    stdio_client: StdioTransport,
) -> None:
    """`list_catalog` is envelope-exempt (spec/M09). It returns
    catalog facts — GPU SKUs, providers, tracked models, quants,
    workload profiles — and a regression that started attaching
    a trust_envelope would be a sign somebody mis-classified a
    tool as numerical."""
    async with Client(stdio_client) as client:
        result = await client.call_tool("list_catalog", {})
        data = _unwrap_result(result)
        assert isinstance(data, dict)
        # Facts surface: at least one of the documented catalog
        # keys must be present.
        assert {"gpus", "providers", "models", "quantizations", "workload_profiles"} & set(
            data.keys()
        ), f"list_catalog returned unexpected keys: {list(data.keys())}"
        assert "trust_envelope" not in data, (
            "list_catalog must be envelope-exempt per spec/M09 § envelope-exempt tools"
        )


async def test_release_provenance_resource_lists_all_upstreams(
    stdio_client: StdioTransport,
) -> None:
    """`cost-cells://provenance` is the trust contract's
    operator-facing surface — names every upstream that
    contributed numbers to a cost cell, with license terms +
    audit links. ADR-006 + spec/M11 require attribution coverage."""
    async with Client(stdio_client) as client:
        contents = await client.read_resource("cost-cells://provenance")
        # FastMCP returns a list of resource content entries.
        assert contents, "provenance resource returned no content"
        # Find the JSON payload across one or more content entries.
        joined = "".join(c.text for c in contents if hasattr(c, "text") and c.text)
        assert joined, "provenance resource has no text content"
        for upstream in ("ComputePrices", "Hugging Face", "Artificial Analysis"):
            assert upstream in joined, f"provenance JSON missing attribution for {upstream}"


async def test_release_benchmark_prompt_renders(
    stdio_client: StdioTransport,
) -> None:
    """`/benchmark-on-budget` is the one v1 prompt. A regression
    that breaks rendering (e.g. a kwargs-name change without an
    upstream prompt update) lands here."""
    async with Client(stdio_client) as client:
        result = await client.get_prompt(
            "benchmark-on-budget",
            arguments={
                "budget_usd": "100",
                "model_slug": "qwen-2-5-7b",
            },
        )
        assert result.messages, "benchmark-on-budget returned no messages"
        # At least the user-role message body must mention the model.
        joined = " ".join(
            getattr(m.content, "text", "") or "" for m in result.messages if hasattr(m, "content")
        )
        assert "qwen-2-5-7b" in joined
