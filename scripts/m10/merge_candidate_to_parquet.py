"""V2 merge tool — appends sanity-checked candidate cells to the
canonical `seeds/benchmark_cells.parquet`.

Coordination with V1: this tool refuses to run unless the candidate
file has a corresponding `.sanity-passed` sidecar (V1 emits one on
exit 0). That keeps every parquet append gated behind a passing
sanity check by construction, even if the curator forgets the
manual step.

Atomicity: writes to `<parquet>.tmp` then renames over the canonical
path. A crash mid-write leaves the original parquet intact and the
.tmp file as forensic material (next run will overwrite it).

Op-point dedup: even though V1 already enforces uniqueness against
the canonical parquet, V2 re-checks defensively — a candidate file
that was sanity-checked against an OLDER parquet state could conflict
with newer rows added in the meantime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from whatcanirun.catalog.benchmark_cells import BenchmarkCell

_BENCH_SCHEMA = pa.schema(
    [
        ("gpu_slug", pa.string()),
        ("model_slug", pa.string()),
        ("quant_slug", pa.string()),
        ("tp_size", pa.int64()),
        ("batch_size", pa.int64()),
        ("context_length", pa.int64()),
        ("decode_tps", pa.float64()),
        ("prefill_tps", pa.float64()),
        ("ttft_ms", pa.float64()),
        ("engine", pa.string()),
        ("engine_version", pa.string()),
        ("measured_at", pa.date32()),
        ("source", pa.string()),
        ("source_url", pa.string()),
        ("notes", pa.string()),
    ]
)


def _op_point_key(cell_or_row: BenchmarkCell | dict[str, object]) -> tuple:
    """Six-tuple primary key. Accepts either a Pydantic BenchmarkCell
    or a dict from a parquet read — both share the same field names."""
    if isinstance(cell_or_row, BenchmarkCell):
        return (
            cell_or_row.gpu_slug,
            cell_or_row.model_slug,
            cell_or_row.quant_slug,
            cell_or_row.tp_size,
            cell_or_row.batch_size,
            cell_or_row.context_length,
        )
    return (
        cell_or_row["gpu_slug"],
        cell_or_row["model_slug"],
        cell_or_row["quant_slug"],
        cell_or_row["tp_size"],
        cell_or_row["batch_size"],
        cell_or_row["context_length"],
    )


def _load_candidates(path: Path) -> list[BenchmarkCell]:
    raw = yaml.safe_load(path.read_text())
    return [BenchmarkCell.model_validate(row) for row in raw]


def _load_existing(parquet: Path) -> list[dict[str, object]]:
    if not parquet.exists():
        return []
    table = pq.read_table(parquet)
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


def _cell_to_row(cell: BenchmarkCell) -> dict[str, object]:
    """Pydantic → dict with measured_at as dt.date for date32."""
    row = cell.model_dump()
    if isinstance(row.get("measured_at"), str):
        row["measured_at"] = dt.date.fromisoformat(row["measured_at"])
    return row


def _atomic_write(parquet: Path, rows: list[dict[str, object]]) -> None:
    """Write to `<parquet>.tmp` then os.replace over the canonical
    path. os.replace is atomic within a filesystem on POSIX."""
    tmp = parquet.with_suffix(parquet.suffix + ".tmp")
    table = pa.Table.from_pylist(rows, schema=_BENCH_SCHEMA)
    pq.write_table(table, tmp)
    os.replace(tmp, parquet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merge_candidate_to_parquet")
    parser.add_argument("candidate", type=Path, help="path to sanity-checked candidate YAML")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("seeds/benchmark_cells.parquet"),
        help="path to the canonical benchmark_cells parquet",
    )
    args = parser.parse_args(argv)

    sidecar = args.candidate.with_suffix(args.candidate.suffix + ".sanity-passed")
    if not sidecar.exists():
        print(
            f"ERROR: candidate {args.candidate} has no .sanity-passed sidecar at {sidecar}. "
            f"Run scripts/m10/sanity_check_cells.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    candidates = _load_candidates(args.candidate)
    existing_rows = _load_existing(args.parquet)
    existing_keys = {_op_point_key(r) for r in existing_rows}

    conflicts = [c for c in candidates if _op_point_key(c) in existing_keys]
    if conflicts:
        for c in conflicts:
            print(
                f"ERROR: op-point {_op_point_key(c)!r} already in parquet; "
                f"V2 refuses to overwrite. If the candidate is a re-measurement, "
                f"delete the existing row from the parquet first.",
                file=sys.stderr,
            )
        sys.exit(3)

    new_rows = [_cell_to_row(c) for c in candidates]
    merged = existing_rows + new_rows
    _atomic_write(args.parquet, merged)

    print(f"merged {len(new_rows)} new rows into {args.parquet}")
    print(f"  total rows: {len(merged)} (was {len(existing_rows)})")
    for cell in candidates:
        print(f"  added op-point {_op_point_key(cell)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
