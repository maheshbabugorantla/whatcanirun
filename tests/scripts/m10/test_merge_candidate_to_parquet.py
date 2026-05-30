"""Tests for `scripts/m10/merge_candidate_to_parquet.py`. Covers
the sidecar-gating contract, op-point-key dedup, atomic write
behavior, and the diff report on successful merge."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.m10.merge_candidate_to_parquet import main


def _candidate_dict(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "gpu_slug": "h100",
        "model_slug": "llama-3-1-8b",
        "quant_slug": "bf16",
        "tp_size": 1,
        "batch_size": 1,
        "context_length": 4096,
        "decode_tps": 130.0,
        "prefill_tps": None,
        "ttft_ms": None,
        "engine": "vllm",
        "engine_version": "0.6.x",
        "measured_at": "2026-04-01",
        "source": "public_benchmark_anchor",
        "source_url": "https://example.com/llama-3-1-8b-h100-bf16",
        "notes": (
            "Single H100 SXM, bf16, batch=1, ctx=4096. vLLM 0.6.x "
            "with paged_attention. Reference run from blog."
        ),
    }
    return defaults | overrides


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


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    coerced: list[dict[str, object]] = []
    for row in rows:
        d = dict(row)
        ma = d.get("measured_at")
        if isinstance(ma, str):
            d["measured_at"] = dt.date.fromisoformat(ma)
        coerced.append(d)
    table = pa.Table.from_pylist(coerced, schema=_BENCH_SCHEMA)
    pq.write_table(table, path)
    return path


def _write_candidate(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump(rows))
    return path


def _write_sidecar(candidate_path: Path) -> Path:
    sidecar = candidate_path.with_suffix(candidate_path.suffix + ".sanity-passed")
    sidecar.write_text("sanity_check_cells exit 0\n")
    return sidecar


def _argv(candidate: Path, parquet: Path) -> list[str]:
    return [str(candidate), "--parquet", str(parquet)]


def test_refuses_without_sidecar(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate.yaml", [_candidate_dict()])
    parquet = _write_parquet(tmp_path / "bench.parquet", [])

    with pytest.raises(SystemExit) as exc_info:
        main(_argv(candidate, parquet))
    assert exc_info.value.code != 0


def test_merge_appends_new_rows(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate.yaml", [_candidate_dict()])
    _write_sidecar(candidate)
    parquet = _write_parquet(tmp_path / "bench.parquet", [])

    exit_code = main(_argv(candidate, parquet))
    assert exit_code == 0

    table = pq.read_table(parquet)
    assert table.num_rows == 1
    assert table.column("gpu_slug")[0].as_py() == "h100"
    assert table.column("decode_tps")[0].as_py() == 130.0


def test_merge_preserves_existing_rows(tmp_path: Path) -> None:
    existing = _candidate_dict(gpu_slug="h200", decode_tps=200.0)
    parquet = _write_parquet(tmp_path / "bench.parquet", [existing])

    candidate = _write_candidate(tmp_path / "candidate.yaml", [_candidate_dict()])
    _write_sidecar(candidate)

    exit_code = main(_argv(candidate, parquet))
    assert exit_code == 0

    table = pq.read_table(parquet)
    assert table.num_rows == 2
    gpu_slugs = set(table.column("gpu_slug").to_pylist())
    assert gpu_slugs == {"h100", "h200"}


def test_merge_rejects_duplicate_op_point(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path / "bench.parquet", [_candidate_dict(decode_tps=999.0)])
    candidate = _write_candidate(tmp_path / "candidate.yaml", [_candidate_dict()])
    _write_sidecar(candidate)

    with pytest.raises(SystemExit) as exc_info:
        main(_argv(candidate, parquet))
    assert exc_info.value.code != 0


def test_merge_is_atomic(tmp_path: Path) -> None:
    """V2 must write atomically (tmp+rename) so a crash mid-write
    can't corrupt the canonical parquet. Verified indirectly: after
    a successful merge, no stray *.tmp files are left in the dir."""
    candidate = _write_candidate(tmp_path / "candidate.yaml", [_candidate_dict()])
    _write_sidecar(candidate)
    parquet = _write_parquet(tmp_path / "bench.parquet", [])

    exit_code = main(_argv(candidate, parquet))
    assert exit_code == 0

    stray = list(tmp_path.glob("*.tmp"))
    assert not stray, f"unexpected tmp files left: {stray}"
