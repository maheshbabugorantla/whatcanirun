"""Join integrity test for seeds/gpus_supplement.yaml.

Every supplement slug must match a real ComputePrices `/api/v1/gpus`
slug (captured offline in tests/fixtures/). If this test fails after a
fresh CP fixture capture, either:

  (a) ComputePrices renamed a slug — update seeds/gpus_supplement.yaml
      to track the new name and run scripts/capture_cp_gpus_fixture.py
      to refresh the fixture, OR
  (b) We added a supplement row for a GPU CP doesn't list yet — push
      ComputePrices to add it; do NOT silently ship a row that can't
      be joined.

Live network calls are forbidden per ADR-013; this test reads only the
committed fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whatcanirun.catalog.loaders import load_gpu_supplements

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEEDS_PATH = _REPO_ROOT / "seeds" / "gpus_supplement.yaml"
_CP_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "cp_gpus_2026-05-26.json"


@pytest.fixture(scope="module")
def cp_slugs() -> set[str]:
    payload = json.loads(_CP_FIXTURE.read_text())
    return {row["slug"] for row in payload["data"]}


@pytest.fixture(scope="module")
def supplement_slugs() -> list[str]:
    return [r.slug for r in load_gpu_supplements(_SEEDS_PATH)]


def test_supplement_has_exactly_12_rows(supplement_slugs: list[str]) -> None:
    assert len(supplement_slugs) == 12


def test_supplement_slugs_are_unique(supplement_slugs: list[str]) -> None:
    duplicates = [s for s in supplement_slugs if supplement_slugs.count(s) > 1]
    assert duplicates == [], f"duplicate slugs in supplement: {sorted(set(duplicates))}"


def test_every_supplement_slug_joins_cp(supplement_slugs: list[str], cp_slugs: set[str]) -> None:
    missing = [s for s in supplement_slugs if s not in cp_slugs]
    assert missing == [], (
        f"supplement slugs not found in ComputePrices fixture: {missing}. "
        f"Either CP renamed the slug or the supplement row is for a GPU CP "
        f"doesn't list yet — see test docstring."
    )


def test_every_supplement_row_has_datasheet_url() -> None:
    rows = load_gpu_supplements(_SEEDS_PATH)
    for row in rows:
        assert row.datasheet_url.startswith(
            "https://"
        ), f"row {row.slug!r} has non-https datasheet_url: {row.datasheet_url!r}"
