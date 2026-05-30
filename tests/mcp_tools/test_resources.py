"""M09 Slice G + H: cost-cells MCP resources.

Two resources land in this file:

- `cost-cells://current` — Parquet materialization of all current
  cost cells. Re-rendered when any contributing cache invalidates.
  Carries `generated_at` + per-source freshness via the parquet
  metadata block.

- `cost-cells://provenance` — JSON document. Every data source
  named with attribution string, every ADR linked, the "what we
  do NOT model" list, license declarations. The single document
  anyone can audit to decide whether to trust this server.

Both are FastMCP resources (not tools). Resources are addressable
by URI and cacheable on the client side; tools are imperative
calls. A regression that re-implements either as a tool fails
the "resources are not tools" pitfall check from spec/M09 §
Common pitfalls.
"""

from __future__ import annotations

import asyncio
import io
import json

import pyarrow.parquet as pq

from whatcanirun.server import mcp


def _list_resource_uris() -> set[str]:
    """FastMCP exposes resources via async `get_resources()`.
    Drive it from the test thread."""
    resources = asyncio.run(mcp.get_resources())
    return set(resources.keys())


def _read_provenance() -> str:
    """Helper: drive the provenance resource read from a sync test."""
    resource = asyncio.run(mcp.get_resource("cost-cells://provenance"))
    raw = asyncio.run(resource.read())
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    assert isinstance(raw, str)
    return raw


def test_cost_cells_current_resource_registered() -> None:
    """Spec/M09 § Resources §1: `cost-cells://current` must be
    advertised as a resource at the standard URI. A regression
    that drops the registration (or re-registers as a tool) fails
    here."""
    uris = _list_resource_uris()
    assert "cost-cells://current" in uris, (
        f"`cost-cells://current` not registered as resource; registered resources: {sorted(uris)}"
    )


def test_cost_cells_current_returns_parquet_bytes() -> None:
    """Reading the resource must produce valid Parquet that pyarrow
    can round-trip. An empty cache state is acceptable (returns an
    empty table with the documented schema) — the resource
    materializer degrades gracefully rather than failing the read."""
    resource = asyncio.run(mcp.get_resource("cost-cells://current"))
    raw = asyncio.run(resource.read())
    assert isinstance(raw, bytes)
    # Validate via pyarrow round-trip — a malformed parquet would
    # raise here, surfacing the resource-format regression at the
    # MCP boundary rather than at client-side render time.
    table = pq.read_table(io.BytesIO(raw))
    # The documented column set per M08's `_resource_schema`.
    expected_cols = {
        "gpu_slug",
        "provider_slug",
        "model_slug",
        "quant_slug",
        "tp_size",
        "batch_size",
        "context_length",
        "deployment_mode",
        "hourly_usd",
        "pricing_type",
        "price_per_m_input_usd",
        "price_per_m_output_usd",
        "decode_tps",
        "cost_per_m_output_usd_self_hosted",
        "availability_modeled",
        "trust_confidence",
    }
    assert expected_cols.issubset(set(table.column_names)), (
        f"missing expected columns; got: {sorted(table.column_names)}"
    )


def test_current_resource_handler_is_async_coroutine() -> None:
    """Spec/M09 § Resources §1: the `cost-cells://current` handler
    must be a coroutine function so it can `await load_runtime_deps`.
    The earlier placeholder was a sync function returning empty
    parquet; the wired version (commit 11bf4cd) is `async def`. A
    regression that converts the handler back to sync would lose
    the deps loading and silently re-render empty.

    This test asserts ONLY the coroutine-function property — the
    end-to-end "warm caches produce non-empty parquet rows" path
    is covered by the integration suite, which exercises the
    handler through the real FastMCP Client with stubbed CP/HF
    state."""
    import inspect

    from whatcanirun.mcp_tools.resources import render_current_cost_cells

    assert inspect.iscoroutinefunction(render_current_cost_cells), (
        "render_current_cost_cells must be async; the sync placeholder "
        "couldn't await load_runtime_deps"
    )


def test_cost_cells_provenance_resource_registered() -> None:
    """Spec/M09 § Resources §2: `cost-cells://provenance` must be
    advertised as a resource. Same pitfall guard as above."""
    uris = _list_resource_uris()
    assert "cost-cells://provenance" in uris, (
        f"`cost-cells://provenance` not registered as resource; "
        f"registered resources: {sorted(uris)}"
    )


def test_provenance_resource_is_valid_json() -> None:
    """The provenance resource is JSON. Reading it should produce
    bytes that `json.loads` accepts without error — a stray
    f-string formatting bug or a non-serializable value would
    surface as a JSONDecodeError, not a silent corruption of the
    audit trail."""
    raw = _read_provenance()
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_provenance_resource_cites_each_upstream() -> None:
    """Spec/M09 acceptance: 'cost-cells://provenance contains AA
    attribution and ComputePrices disclaimer verbatim'. The
    provenance doc must name every upstream the server depends
    on so a user auditing trust has a single source of truth."""
    data = json.loads(_read_provenance())
    source_names = {s["name"] for s in data["sources"]}
    assert "computeprices" in source_names
    assert "huggingface" in source_names
    assert "artificial_analysis" in source_names


def test_provenance_resource_lists_what_we_do_not_model() -> None:
    """Spec/M09 acceptance: the 'what we DO NOT model' list is a
    hard requirement — it's the section that prevents users from
    misinterpreting present figures as covering things they don't
    (rentability, latency, kernel support, etc.)."""
    data = json.loads(_read_provenance())
    assert "what_we_do_not_model" in data
    assert isinstance(data["what_we_do_not_model"], list)
    assert data["what_we_do_not_model"], "empty 'what we do NOT model' list"


def test_provenance_resource_lists_locked_adrs() -> None:
    """ADRs are the architectural decisions the server is built
    on. Surfacing them in the provenance doc lets a user trace
    why a particular caveat exists ('this server doesn't measure
    latency because ADR-010 says...')."""
    data = json.loads(_read_provenance())
    assert "adrs" in data
    # Sanity bound: a handful of ADRs land in the locked set per
    # spec/SHARED.md (we have 15 today; expect at least the
    # foundational 5).
    assert len(data["adrs"]) >= 5
