"""Tests for the `sanity_check_cells.main` CLI wrapper. Covers the
exit-code contract (0 clean, 1 warn, 2 error), .sanity-passed
sidecar emission, and the load-from-files behavior."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scripts.m10.sanity_check_cells import main

from whatcanirun.catalog.benchmark_cells import BenchmarkCell


def _candidate_dict(**overrides: object) -> dict[str, object]:
    """A serializable dict that round-trips through YAML and parses
    as a BenchmarkCell. Tests override fields to exercise the
    different exit-code paths."""
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


def _write_candidate_yaml(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump(rows))
    return path


_BENCH_PARQUET_SCHEMA = pa.schema(
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


def _write_empty_parquet(path: Path) -> Path:
    table = pa.Table.from_pylist([], schema=_BENCH_PARQUET_SCHEMA)
    pq.write_table(table, path)
    return path


def _write_catalog_yamls(tmp_path: Path) -> dict[str, Path]:
    """Minimal gpu_catalog / quantizations / tracked_models YAMLs."""
    gpu_path = tmp_path / "gpu_catalog.yaml"
    gpu_path.write_text(
        yaml.safe_dump(
            [
                {
                    "slug": "h100",
                    "name": "H100",
                    "manufacturer": "NVIDIA",
                    "architecture": "hopper",
                    "vram_gb": 80,
                    "release_date": None,
                    "specs": {"memory_bandwidth_gbps": 3350.0},
                }
            ]
        )
    )
    quant_path = tmp_path / "quantizations.yaml"
    quant_path.write_text(
        yaml.safe_dump(
            [
                {
                    "slug": "bf16",
                    "bits_per_weight": 16,
                    "kv_cache_bits_default": 16,
                    "introduced_architecture": "ampere",
                    "notes": "test",
                },
                {
                    "slug": "fp8",
                    "bits_per_weight": 8,
                    "kv_cache_bits_default": 16,
                    "introduced_architecture": "hopper",
                    "notes": "test",
                },
            ]
        )
    )
    tracked_path = tmp_path / "tracked_models.yaml"
    tracked_path.write_text(
        yaml.safe_dump(
            [
                {
                    "slug": "llama-3-1-8b",
                    "hf_repo_id": "meta-llama/Meta-Llama-3.1-8B",
                    "total_params_b": 8.0,
                }
            ]
        )
    )
    return {"gpu": gpu_path, "quant": quant_path, "tracked": tracked_path}


def _argv(candidate: Path, parquet: Path, catalogs: dict[str, Path]) -> list[str]:
    return [
        str(candidate),
        "--parquet",
        str(parquet),
        "--gpu-catalog",
        str(catalogs["gpu"]),
        "--quantizations",
        str(catalogs["quant"]),
        "--tracked-models",
        str(catalogs["tracked"]),
    ]


def test_clean_candidate_exits_zero_and_writes_sidecar(tmp_path: Path) -> None:
    candidate = _write_candidate_yaml(tmp_path / "candidate.yaml", [_candidate_dict()])
    parquet = _write_empty_parquet(tmp_path / "bench.parquet")
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))

    assert exit_code == 0
    sidecar = candidate.with_suffix(candidate.suffix + ".sanity-passed")
    assert sidecar.exists(), "sidecar should be emitted on exit 0"


def test_warn_candidate_exits_one_no_sidecar(tmp_path: Path) -> None:
    # 2024-01-01 is >18 months stale from 2026-05-30 → warn.
    candidate = _write_candidate_yaml(
        tmp_path / "candidate.yaml", [_candidate_dict(measured_at="2024-01-01")]
    )
    parquet = _write_empty_parquet(tmp_path / "bench.parquet")
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))

    assert exit_code == 1
    sidecar = candidate.with_suffix(candidate.suffix + ".sanity-passed")
    assert not sidecar.exists(), "sidecar must NOT be emitted on warn"


def test_error_candidate_exits_two_no_sidecar(tmp_path: Path) -> None:
    # engine_version="latest" → check_engine_version_format errors.
    candidate = _write_candidate_yaml(
        tmp_path / "candidate.yaml", [_candidate_dict(engine_version="latest")]
    )
    parquet = _write_empty_parquet(tmp_path / "bench.parquet")
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))

    assert exit_code == 2
    sidecar = candidate.with_suffix(candidate.suffix + ".sanity-passed")
    assert not sidecar.exists(), "sidecar must NOT be emitted on error"


def test_unknown_gpu_slug_errors_two(tmp_path: Path) -> None:
    candidate = _write_candidate_yaml(
        tmp_path / "candidate.yaml", [_candidate_dict(gpu_slug="rtx-9090")]
    )
    parquet = _write_empty_parquet(tmp_path / "bench.parquet")
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))
    assert exit_code == 2


def test_multiple_candidates_aggregate(tmp_path: Path) -> None:
    # First row clean, second errors. Aggregate exit code is max severity.
    candidate = _write_candidate_yaml(
        tmp_path / "candidate.yaml",
        [_candidate_dict(), _candidate_dict(engine_version="main")],
    )
    parquet = _write_empty_parquet(tmp_path / "bench.parquet")
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))
    assert exit_code == 2


def test_existing_parquet_blocks_duplicate_op_point(tmp_path: Path) -> None:
    # Pre-populate parquet with a row whose op-point matches the candidate's.
    existing_cell = BenchmarkCell(**_candidate_dict(decode_tps=999.0))
    row = existing_cell.model_dump()
    row["measured_at"] = dt.date.fromisoformat(str(_candidate_dict()["measured_at"]))
    table = pa.Table.from_pylist([row], schema=_BENCH_PARQUET_SCHEMA)
    parquet = tmp_path / "bench.parquet"
    pq.write_table(table, parquet)

    candidate = _write_candidate_yaml(tmp_path / "candidate.yaml", [_candidate_dict()])
    catalogs = _write_catalog_yamls(tmp_path)

    exit_code = main(_argv(candidate, parquet, catalogs))
    assert exit_code == 2
