"""Load `seeds/benchmark_cells.parquet` into typed `BenchmarkCell`
rows for M07's `estimate_tps` Tier 1a/1b lookup.

The parquet is the seed for v1; M10 (when it ships its 20-30 hand-
curated rows) replaces the bootstrap M07 ships with a more
comprehensive set, but the loader contract stays identical.

Every row in the seed MUST have `source="public_benchmark_anchor"`
in v1 — the BenchmarkCell validator already enforces this at
construction; this loader's value-add is just `parquet → list[
BenchmarkCell]` plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from whatcanirun.catalog.benchmark_cells import BenchmarkCell


def load_benchmark_cells(path: Path) -> list[BenchmarkCell]:
    """Read the parquet at `path` and return validated
    `BenchmarkCell` rows. Raises whatever Pydantic raises if a row
    fails validation — including the v1 `own_measured` rejection,
    which is the load-bearing guard that makes acceptance
    criterion 5 mechanically enforceable."""
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    rows: list[BenchmarkCell] = []
    raw_rows: list[dict[str, Any]] = table.to_pylist()
    for raw in raw_rows:
        # pyarrow leaves `prefill_tps` / `ttft_ms` as None when
        # the column is null-typed — pass through to Pydantic,
        # which has those fields as Optional[float].
        cleaned: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
        # Required fields with no value would already fail
        # validation; this is just stripping the explicit-Null
        # entries pyarrow produces for null-typed columns.
        rows.append(BenchmarkCell(**cleaned))
    return rows
