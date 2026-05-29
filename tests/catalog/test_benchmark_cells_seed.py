"""M07 acceptance criterion 5: 'v1 NEVER returns
source="own_measured" — seeds/benchmark_cells.parquet validated
to contain only public_benchmark_anchor rows. This is a fast
test that scans the seed file.'

The BenchmarkCell validator already rejects own_measured at row
construction, so a parquet with an own_measured row WOULD fail
to load via load_benchmark_cells — but the spec explicitly wants
a SEPARATE acceptance test that scans the seed file. This is it.
The test fails loud if anyone slips a non-anchor row past the
validator (e.g. by hand-editing the parquet with pyarrow).
"""

from __future__ import annotations

from pathlib import Path

from whatcanirun.catalog.benchmark_cells_loader import load_benchmark_cells

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_PARQUET = _REPO_ROOT / "seeds" / "benchmark_cells.parquet"


def test_seed_parquet_loads_without_error() -> None:
    """Spec acceptance criterion 5 prerequisite: the seed file
    exists, is parseable as parquet, and every row passes
    BenchmarkCell validation. Failing this fails M07 merge."""
    rows = load_benchmark_cells(_SEED_PARQUET)
    assert len(rows) >= 1, "seed parquet is empty — M07 needs at least 1 anchor"


def test_every_seed_row_is_public_benchmark_anchor() -> None:
    """Spec acceptance criterion 5: seed validated to contain
    ONLY public_benchmark_anchor rows in v1. M17 (v2) introduces
    own_measured cells via GuideLLM runs; until then, this test
    fails loudly if a row's source drifts.

    Belt-and-suspenders: BenchmarkCell.source field_validator
    already rejects own_measured at row construction, so a
    rogue row would fail to load — but the spec calls for an
    explicit scan-the-seed-file test, and the dual enforcement
    makes the v2 unlock a deliberate two-step (flip the
    validator off AND update this test) rather than a one-step
    that risks shipping unverified own_measured data."""
    rows = load_benchmark_cells(_SEED_PARQUET)
    bad = [row for row in rows if row.source != "public_benchmark_anchor"]
    assert not bad, (
        f"v1 seed must contain ONLY public_benchmark_anchor rows; "
        f"found {len(bad)} other: {[r.source for r in bad]!r}"
    )


def test_every_seed_row_has_a_source_url() -> None:
    """Trust contract: every anchor row must cite a public URL
    so the LLM caller can disclose the methodology. The
    BenchmarkCell `source_url: Field(min_length=1)` already
    enforces this — this test is the seed-level smoke check."""
    rows = load_benchmark_cells(_SEED_PARQUET)
    missing_url = [r.model_slug for r in rows if not r.source_url]
    assert not missing_url, f"seed rows without source_url: {missing_url!r}"
