"""Unit tests for scripts/m10/sanity_check_cells.py — one test file
that covers every check function. Each check has at least one pass
case, one error case, and (where applicable) one warn case.

Per the /tdd skill, each check function is TDD'd in its own cycle —
the test for cycle N lands BEFORE the impl in commit N, then both
together. This file grows incrementally as cycles land.

Tests import from `scripts.m10.sanity_check_cells` (the package
layout's `scripts/m10/__init__.py` makes that path importable from
the test process)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from scripts.m10.sanity_check_cells import (
    CheckResult,
    check_source_url_well_formed,
)

from whatcanirun.catalog.benchmark_cells import BenchmarkCell


def _valid_cell(**overrides: Any) -> BenchmarkCell:
    """Returns a BenchmarkCell that passes ALL checks. Tests
    override one field at a time to exercise a single check's
    error path without tripping unrelated ones."""
    defaults: dict[str, Any] = {
        "gpu_slug": "h100",
        "model_slug": "llama-3-1-8b",
        "quant_slug": "bf16",
        "tp_size": 1,
        "batch_size": 1,
        "context_length": 4096,
        "decode_tps": 100.0,
        "prefill_tps": None,
        "ttft_ms": None,
        "engine": "vllm",
        "engine_version": "0.6.x",
        "measured_at": dt.date(2026, 4, 1),
        "source": "public_benchmark_anchor",
        "source_url": "https://example.com/llama-3-1-8b-h100",
        "notes": (
            "Single H100 SXM, bf16, batch=1, ctx=4096. vLLM 0.6.x "
            "with paged_attention. Reference run from blog."
        ),
    }
    return BenchmarkCell(**(defaults | overrides))


# ---------------------------------------------------------------- check_source_url_well_formed


class TestSourceUrlWellFormed:
    """The cell's `source_url` must be a real http(s) URL — the
    Pydantic `min_length=1` is too weak: empty-prefixed strings,
    relative paths, and `javascript:` URIs all pass that check but
    can't be audited by a human reader.

    Per the M10 trust-contract spec, the URL has to lead a reader
    to the methodology disclosure. If it doesn't have a scheme +
    netloc, it can't."""

    def test_https_url_passes(self) -> None:
        cell = _valid_cell(source_url="https://example.com/article")
        result = check_source_url_well_formed(cell)
        assert result.severity == "pass"

    def test_http_url_passes(self) -> None:
        # Plain http is acceptable (some valid academic / blog
        # sources don't have https), but the check should still
        # let it through.
        cell = _valid_cell(source_url="http://example.com/article")
        result = check_source_url_well_formed(cell)
        assert result.severity == "pass"

    def test_missing_scheme_errors(self) -> None:
        cell = _valid_cell(source_url="example.com/article")
        result = check_source_url_well_formed(cell)
        assert result.severity == "error"
        assert "scheme" in result.message.lower()

    def test_javascript_uri_errors(self) -> None:
        # Defensive — `javascript:alert(1)` parses fine through
        # urlparse but is not an auditable methodology source.
        cell = _valid_cell(source_url="javascript:alert(1)")
        result = check_source_url_well_formed(cell)
        assert result.severity == "error"

    def test_missing_netloc_errors(self) -> None:
        # `https:` alone, or `https://` with no host.
        cell = _valid_cell(source_url="https://")
        result = check_source_url_well_formed(cell)
        assert result.severity == "error"
        assert "host" in result.message.lower() or "netloc" in result.message.lower()


# ---------------------------------------------------------------- shape tests


def test_check_result_is_immutable_dataclass() -> None:
    """CheckResult is the protocol every check returns. Frozen +
    explicit severity literal so the CLI can pattern-match cleanly
    across exit codes (0 pass, 1 warn, 2 error)."""
    r = CheckResult(severity="pass", message="ok")
    assert r.severity == "pass"
    assert r.message == "ok"
    with pytest.raises(AttributeError):
        r.severity = "warn"  # type: ignore[misc]
