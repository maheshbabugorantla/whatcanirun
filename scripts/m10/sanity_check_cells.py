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

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from whatcanirun.catalog.benchmark_cells import BenchmarkCell

Severity = Literal["pass", "warn", "error"]


@dataclass(frozen=True)
class CheckResult:
    """The protocol every check returns. Frozen so the CLI can
    safely thread results through aggregation without defensive
    copies; severity is a closed Literal so exit-code mapping is
    exhaustive at the type level."""

    severity: Severity
    message: str


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
