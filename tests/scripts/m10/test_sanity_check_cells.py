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
    check_engine_version_format,
    check_measured_at_recency,
    check_methodology_complete,
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


# ---------------------------------------------------------------- check_engine_version_format


class TestEngineVersionFormat:
    """`engine_version` must be semver-shaped: `MAJOR.MINOR` with
    an optional patch suffix that's either a digit or the literal
    `.x` (vLLM commonly publishes blog-post numbers as `0.6.x`
    without committing to a patch — this is fine; `latest`, `main`,
    `dev`, and empty strings are not). Catches the "stale numbers"
    pitfall up front: a cell tagged `latest` today means something
    different next quarter, breaking auditability."""

    def test_simple_major_minor_passes(self) -> None:
        cell = _valid_cell(engine_version="0.6")
        result = check_engine_version_format(cell)
        assert result.severity == "pass"

    def test_major_minor_patch_passes(self) -> None:
        cell = _valid_cell(engine_version="0.6.3")
        result = check_engine_version_format(cell)
        assert result.severity == "pass"

    def test_major_minor_x_passes(self) -> None:
        # vLLM 0.6.x convention.
        cell = _valid_cell(engine_version="0.6.x")
        result = check_engine_version_format(cell)
        assert result.severity == "pass"

    def test_latest_errors(self) -> None:
        cell = _valid_cell(engine_version="latest")
        result = check_engine_version_format(cell)
        assert result.severity == "error"
        assert "engine_version" in result.message

    def test_main_errors(self) -> None:
        cell = _valid_cell(engine_version="main")
        result = check_engine_version_format(cell)
        assert result.severity == "error"

    def test_empty_errors(self) -> None:
        cell = _valid_cell(engine_version="")
        result = check_engine_version_format(cell)
        assert result.severity == "error"


# ---------------------------------------------------------------- check_measured_at_recency


class TestMeasuredAtRecency:
    """`measured_at` should be within the last 18 months of TODAY.
    Older numbers correspond to older engine versions, older drivers,
    older PyTorch — they're not wrong, but they're less applicable
    to today's stacks and warrant a warning so the curator can
    decide whether to keep the cell.

    `_today` is injected so tests are deterministic across calendars."""

    def test_recent_date_passes(self) -> None:
        cell = _valid_cell(measured_at=dt.date(2026, 4, 1))
        result = check_measured_at_recency(cell, _today=dt.date(2026, 5, 30))
        assert result.severity == "pass"

    def test_just_under_18_months_passes(self) -> None:
        # 17 months back from 2026-05-30 → 2024-12-30.
        cell = _valid_cell(measured_at=dt.date(2024, 12, 30))
        result = check_measured_at_recency(cell, _today=dt.date(2026, 5, 30))
        assert result.severity == "pass"

    def test_just_over_18_months_warns(self) -> None:
        # 19 months back from 2026-05-30 → 2024-10-30.
        cell = _valid_cell(measured_at=dt.date(2024, 10, 30))
        result = check_measured_at_recency(cell, _today=dt.date(2026, 5, 30))
        assert result.severity == "warn"
        assert "stale" in result.message.lower() or "months" in result.message.lower()

    def test_future_date_errors(self) -> None:
        # A future measured_at can't be a real measurement — almost
        # certainly a typo. Hard error.
        cell = _valid_cell(measured_at=dt.date(2027, 1, 1))
        result = check_measured_at_recency(cell, _today=dt.date(2026, 5, 30))
        assert result.severity == "error"
        assert "future" in result.message.lower()


# ---------------------------------------------------------------- check_methodology_complete


class TestMethodologyComplete:
    """`notes` must (a) be at least 30 chars (the spec requires a
    1-2 sentence methodology summary, which is at least that long
    when honest) AND (b) reference both the engine version AND the
    batch size somewhere in the text. Catches the M10 pitfall:
    "some 'benchmark' blog posts don't specify batch size or engine
    version — skip those rows."

    The check is strict about *mentioning* the values, not about
    parsing them: any notes string that contains the literal
    `engine_version` value and `batch=N` (or `batch_size=N`) passes.
    """

    def test_full_notes_passes(self) -> None:
        cell = _valid_cell(
            engine_version="0.6.x",
            batch_size=1,
            notes="Single H100 SXM, bf16, batch=1, ctx=4096. vLLM 0.6.x with paged_attention.",
        )
        result = check_methodology_complete(cell)
        assert result.severity == "pass"

    def test_too_short_errors(self) -> None:
        cell = _valid_cell(notes="see source")
        result = check_methodology_complete(cell)
        assert result.severity == "error"
        assert "notes" in result.message.lower() and (
            "30" in result.message or "short" in result.message.lower()
        )

    def test_missing_engine_version_errors(self) -> None:
        cell = _valid_cell(
            engine_version="0.6.x",
            notes=(
                "Single H100 SXM, bf16, batch=1, ctx=4096. Reference run "
                "from blog without engine version detail."
            ),
        )
        result = check_methodology_complete(cell)
        assert result.severity == "error"
        assert "engine_version" in result.message

    def test_missing_batch_errors(self) -> None:
        cell = _valid_cell(
            engine_version="0.6.x",
            batch_size=1,
            notes=(
                "Single H100 SXM, bf16, ctx=4096. vLLM 0.6.x with "
                "paged_attention. Reference run from blog."
            ),
        )
        result = check_methodology_complete(cell)
        assert result.severity == "error"
        assert "batch" in result.message.lower()

    def test_batch_size_keyword_form_passes(self) -> None:
        # Some authors write `batch_size=4`, others `batch=4`. Both
        # satisfy the "mentions batch" requirement.
        cell = _valid_cell(
            engine_version="0.6.x",
            batch_size=4,
            notes="H100 SXM, bf16, batch_size=4, ctx=4096. vLLM 0.6.x reference.",
        )
        result = check_methodology_complete(cell)
        assert result.severity == "pass"


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
