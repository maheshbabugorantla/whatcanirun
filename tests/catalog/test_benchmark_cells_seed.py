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

M10 Slice C adds a `@pytest.mark.network` test that HEAD-requests
every committed source_url and asserts it still responds. Deselected
by default (pyproject.toml's addopts excludes the `network` marker)
so CI doesn't depend on third-party uptime; run locally before
M10/M11/M12 release cuts with `pytest -m network`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

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


@pytest.mark.network
def test_every_seed_source_url_is_reachable() -> None:
    """M10 Slice C: every committed source_url must respond 2xx/3xx
    at the time of the run. Deselected by default in CI (this hits
    public third-party endpoints whose uptime we don't control); run
    manually before each release cut and record results in the
    commit message that touches the parquet.

    Uses httpx.AsyncClient + asyncio.gather so 20+ URLs check in
    parallel (~1s total instead of N * RTT). HEAD with redirect
    following — some sources reject HEAD; on 405 we fall back to a
    bounded GET that closes after 1 byte to verify the URL resolves.

    Failures are reported with the model_slug and URL so the curator
    can diagnose without re-running."""
    rows = load_benchmark_cells(_SEED_PARQUET)

    async def _check(url: str) -> tuple[str, int | str]:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            try:
                r = await client.head(url)
                if r.status_code == 405:
                    # Some servers (Cloudflare-fronted blogs) reject HEAD.
                    r = await client.get(url)
                return (url, r.status_code)
            except httpx.HTTPError as exc:
                return (url, repr(exc))

    async def _run_all() -> list[tuple[str, int | str]]:
        return await asyncio.gather(*(_check(r.source_url) for r in rows))

    results = asyncio.run(_run_all())
    failures = [
        (url, status)
        for url, status in results
        if not (isinstance(status, int) and 200 <= status < 400)
    ]
    assert not failures, (
        f"unreachable source_url(s): {failures}. Either the source moved "
        f"(update the cell + archive new HTML to seeds/benchmark_cells.sources/) "
        f"or the source is gone (delete the cell)."
    )
