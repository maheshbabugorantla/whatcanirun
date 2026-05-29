"""ADR-014 enforcement — `query_cost_cells` and its helpers must
not contain SQL or `import duckdb`.

The architectural split:

  - `query_cost_cells` (tool-call hot path) — pure Python list
    comprehensions over in-memory caches. Easier to debug, faster
    at v1 scale (~hundreds of rows), no DB round-trip overhead.

  - `render_cost_cells_resource` (resource materialization for
    `cost-cells://current`) — DuckDB ONLY here. The function
    body uses `import duckdb` and `con.sql(...)` to assemble the
    parquet output.

This test scans the source of `cost_cells.py`, locates the body
of every function that is part of the tool-call path, and asserts
no DuckDB or SQL patterns appear. A future refactor that 'just
adds a quick SQL join' for a filter case goes red.

The test also asserts that `render_cost_cells_resource` IS still
present (we want the resource path; we just want it isolated).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import whatcanirun.plan.cost_cells as cost_cells_module

# Tool-call path functions — must NOT contain SQL or DuckDB.
# Must cover every helper reachable from `query_cost_cells`'s call
# graph; a future SQL leak into one of these is what the test is
# trying to catch.
_TOOL_PATH_FUNCTIONS = {
    "query_cost_cells",
    "_self_hosted_cost",
    "_partial_envelope_for_gpu_rental",
    "_partial_envelope_for_hosted_api",
    "_freshness_from_sources",
    "_find_matched_bench_cell",
}

# Resource-path functions — ALLOWED to import duckdb / use SQL.
_RESOURCE_PATH_FUNCTIONS = {
    "render_cost_cells_resource",
    "_empty_table",
    "_resource_schema",
}


def _source_of(name: str) -> str:
    obj = getattr(cost_cells_module, name)
    return inspect.getsource(obj)


def test_query_cost_cells_does_not_import_duckdb() -> None:
    """Spec slice H + acceptance criterion 2: NO `import duckdb`
    inside the tool-call path. Module-level `from duckdb import
    ...` would leak too, but the layout puts the DuckDB import
    INSIDE `render_cost_cells_resource` precisely so this test
    can pass."""
    src = _source_of("query_cost_cells")
    assert "duckdb" not in src, (
        "query_cost_cells references duckdb — DuckDB belongs in "
        "render_cost_cells_resource only (ADR-014)"
    )


def test_query_cost_cells_does_not_execute_sql() -> None:
    """No con.sql, con.execute, con.execute_many patterns in
    the tool-call path."""
    src = _source_of("query_cost_cells")
    for forbidden in ("con.sql", "con.execute", "duckdb.sql", "duckdb.execute"):
        assert forbidden not in src, (
            f"query_cost_cells contains {forbidden!r} — SQL belongs in "
            "render_cost_cells_resource only (ADR-014)"
        )


def test_tool_path_helpers_dont_touch_duckdb() -> None:
    """The private helpers `query_cost_cells` calls (cost math,
    envelope builders) likewise must not reach for DuckDB. A
    refactor that 'optimizes' the cost loop with a SQL join
    goes red here."""
    for fn_name in _TOOL_PATH_FUNCTIONS - {"query_cost_cells"}:
        src = _source_of(fn_name)
        for forbidden in ("duckdb", "con.sql", "con.execute"):
            assert forbidden not in src, (
                f"{fn_name} contains {forbidden!r} — tool-path helpers "
                "must stay pure Python (ADR-014)"
            )


def test_resource_path_is_still_present() -> None:
    """The OTHER side of the split — make sure
    render_cost_cells_resource still exists and uses DuckDB.
    A refactor that removes it would silently break
    cost-cells://current rendering."""
    assert hasattr(cost_cells_module, "render_cost_cells_resource")
    src = _source_of("render_cost_cells_resource")
    assert "duckdb" in src, (
        "render_cost_cells_resource lost its DuckDB usage — that's "
        "the SOLE function in this module allowed to use DuckDB; "
        "removing it means the resource render has no implementation"
    )


def test_render_cost_cells_resource_closes_duckdb_connection() -> None:
    """Copilot review (round 4): `duckdb.connect(":memory:")`
    without explicit cleanup leaks native resources across
    repeated resource renders in a long-running MCP server. The
    fix is the `with` context manager so the handle closes even
    if the pyarrow write raises."""
    src = _source_of("render_cost_cells_resource")
    assert "with duckdb.connect" in src, (
        "render_cost_cells_resource opens a DuckDB connection without "
        "deterministic cleanup — use `with duckdb.connect(...) as con:` "
        "so the native handle is closed even if pyarrow raises"
    )


def test_module_level_imports_dont_leak_duckdb() -> None:
    """`import duckdb` at module level would make DuckDB available
    to every function, including the tool-call path. The split
    requires the import live INSIDE `render_cost_cells_resource`
    so static analysis (and this test) can confirm the surface
    above stays pure."""
    src_path = Path(cost_cells_module.__file__)
    tree = ast.parse(src_path.read_text())
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "duckdb", (
                    "module-level `import duckdb` — must be inside "
                    "render_cost_cells_resource per ADR-014"
                )
        if isinstance(node, ast.ImportFrom) and node.module == "duckdb":
            raise AssertionError(
                "module-level `from duckdb import ...` — must be inside "
                "render_cost_cells_resource per ADR-014"
            )
