"""V1 sanity-check tool for M10 candidate benchmark cells.

Validates a YAML/JSON file of candidate `BenchmarkCell` rows against
methodology + join-key + sanity rules before the rows are merged
into `seeds/benchmark_cells.parquet`. Exit code: 0 clean, 1 warning,
2 blocking error.

Each check is a pure function `(cell, ctx) -> CheckResult` that the
CLI iterates over. The check set is documented in `scripts/m10/README.md`
§ "Tools / V1"; this file owns the implementations.

The CLI surface (`main`) lands once enough checks exist to make the
tool useful. Until then, the module is just a check library that
tests import directly.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from whatcanirun.catalog.benchmark_cells import BenchmarkCell

# Stale-numbers threshold per spec's "Stale numbers" pitfall: a cell
# from 2024 with vLLM 0.4 is less applicable to 2026 stacks. 18 months
# is the rough lifetime over which an engine version stays current.
_RECENCY_THRESHOLD = dt.timedelta(days=30 * 18)

# Semver-ish: MAJOR.MINOR[.(PATCH|x)]. Rejects "latest", "main", "dev".
_ENGINE_VERSION_RE = re.compile(r"^\d+\.\d+(\.(\d+|x))?$")

Severity = Literal["pass", "warn", "error"]


@dataclass(frozen=True)
class CheckResult:
    """The protocol every check returns. Frozen so the CLI can
    safely thread results through aggregation without defensive
    copies; severity is a closed Literal so exit-code mapping is
    exhaustive at the type level."""

    severity: Severity
    message: str


@dataclass(frozen=True)
class CheckContext:
    """Side-data the catalog-aware checks need. Grows as later
    cycles land catalog-join checks (gpu_catalog, quantizations,
    tracked_models go here next). Frozen so a single context can
    be threaded safely across every cell in a candidate file
    without checks mutating shared state."""

    existing_cells: list[BenchmarkCell]


# ---------------------------------------------------------------- checks


def check_source_url_well_formed(cell: BenchmarkCell) -> CheckResult:
    """`source_url` must parse as an absolute http(s) URL — scheme
    must be `http` or `https`, and netloc must be non-empty. The
    Pydantic `min_length=1` on the field accepts strings like
    `javascript:alert(1)`, `example.com/article`, and `https://`
    which can't lead a reader to a methodology disclosure."""
    parsed = urlparse(cell.source_url)
    if parsed.scheme not in {"http", "https"}:
        return CheckResult(
            severity="error",
            message=(
                f"source_url scheme must be http or https, got "
                f"{parsed.scheme!r} for {cell.source_url!r}"
            ),
        )
    if not parsed.netloc:
        return CheckResult(
            severity="error",
            message=(f"source_url has no host (netloc); got {cell.source_url!r}"),
        )
    return CheckResult(severity="pass", message="source_url is well-formed")


def check_engine_version_format(cell: BenchmarkCell) -> CheckResult:
    """`engine_version` must be `MAJOR.MINOR[.(PATCH|x)]`. Rejects
    `latest`, `main`, `dev`, and empty strings, all of which break
    auditability — a cell tagged `latest` today means a different
    engine next quarter."""
    if not _ENGINE_VERSION_RE.match(cell.engine_version):
        return CheckResult(
            severity="error",
            message=(
                f"engine_version must be semver-shaped "
                f"(MAJOR.MINOR[.(PATCH|x)]); got {cell.engine_version!r}. "
                f"Floating refs like 'latest' / 'main' break the audit trail."
            ),
        )
    return CheckResult(
        severity="pass", message=f"engine_version {cell.engine_version!r} is well-formed"
    )


def check_measured_at_recency(cell: BenchmarkCell, *, _today: dt.date | None = None) -> CheckResult:
    """`measured_at` should be within ~18 months of today. Older
    numbers correspond to older engine/driver/PyTorch versions and
    warrant a warning rather than auto-rejection (the curator can
    still keep the cell if the methodology is solid). Future dates
    are hard errors — they can't be a real measurement.

    `_today` is injected for deterministic testing across the
    calendar; production callers leave it as None to use the real
    clock."""
    today = _today if _today is not None else dt.date.today()
    if cell.measured_at > today:
        return CheckResult(
            severity="error",
            message=(
                f"measured_at {cell.measured_at.isoformat()} is in the "
                f"future relative to today {today.isoformat()}; almost "
                f"certainly a typo."
            ),
        )
    age = today - cell.measured_at
    if age > _RECENCY_THRESHOLD:
        months = age.days // 30
        return CheckResult(
            severity="warn",
            message=(
                f"measured_at {cell.measured_at.isoformat()} is "
                f"~{months} months stale (>18 month threshold); the "
                f"engine version in this cell may no longer be representative."
            ),
        )
    return CheckResult(
        severity="pass",
        message=f"measured_at {cell.measured_at.isoformat()} is recent",
    )


def check_methodology_complete(cell: BenchmarkCell) -> CheckResult:
    """`notes` must be ≥30 chars AND mention both the cell's
    `engine_version` and its `batch_size`. The 30-char floor catches
    'see source' / 'reference' placeholder rows; the engine+batch
    mention catches the M10 pitfall of blog posts that don't
    disclose methodology."""
    if len(cell.notes) < 30:
        return CheckResult(
            severity="error",
            message=(
                f"notes is too short ({len(cell.notes)} chars < 30); "
                f"must include a 1-2 sentence methodology summary"
            ),
        )
    if cell.engine_version not in cell.notes:
        return CheckResult(
            severity="error",
            message=(
                f"notes does not mention engine_version "
                f"{cell.engine_version!r}; the curator must explain "
                f"which version was measured"
            ),
        )
    batch_token_a = f"batch={cell.batch_size}"
    batch_token_b = f"batch_size={cell.batch_size}"
    if batch_token_a not in cell.notes and batch_token_b not in cell.notes:
        return CheckResult(
            severity="error",
            message=(
                f"notes does not mention batch_size {cell.batch_size!r}; "
                f"expected literal 'batch={cell.batch_size}' or "
                f"'batch_size={cell.batch_size}' in notes"
            ),
        )
    return CheckResult(severity="pass", message="notes contains required methodology fields")


def _op_point_key(cell: BenchmarkCell) -> tuple[str, str, str, int, int, int]:
    """The six-tuple that BenchmarkCell uses as its primary key
    for tps_estimator Tier 1b matching. Two cells with the same
    key but different decode_tps are an ambiguity the tool path
    can't resolve without a tiebreaker."""
    return (
        cell.gpu_slug,
        cell.model_slug,
        cell.quant_slug,
        cell.tp_size,
        cell.batch_size,
        cell.context_length,
    )


def check_op_point_unique(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """Reject candidate cells whose op-point key already exists in
    the canonical parquet. The Tier 1b matcher takes the first
    match it finds, so silently shadowing an existing row is
    behavior the curator must opt into explicitly (by deleting the
    old row in the same PR). Errors here are blocking."""
    key = _op_point_key(cell)
    for existing in ctx.existing_cells:
        if _op_point_key(existing) == key:
            return CheckResult(
                severity="error",
                message=(
                    f"duplicate op-point {key!r} already in the canonical "
                    f"parquet (existing decode_tps={existing.decode_tps}, "
                    f"candidate decode_tps={cell.decode_tps}). If this is "
                    f"a re-measurement, delete the existing row in the same PR."
                ),
            )
    return CheckResult(severity="pass", message=f"op-point {key!r} is new to the parquet")
