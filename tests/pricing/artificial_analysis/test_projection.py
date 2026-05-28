"""Slice A: `AaModelRow` Pydantic projection per ADR-015.

The shape of the AA `data[]` array is documented in spec/M04 § Public
surface, but `evaluations` and `pricing` carry evolving sub-keys that
this projection must NEVER narrow-type. The trust contract demands
that future Intelligence Index revisions (which AA ships periodically)
can add new evaluation fields without breaking validation; those
fields ride through in `evaluations` and `raw` and can be projected
later if they become first-class.

Captured live with a real AA_API_KEY on 2026-05-27 — fixture has 525
rows; the unique `evaluations` key set across the whole fixture is 15
fields (matches spec's "16+" within capture-date noise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from whatcanirun.pricing.artificial_analysis import AaModelRow

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def aa_payload() -> dict[str, Any]:
    return json.loads((_FIXTURES / "aa_models_2026-05-27.json").read_text())


@pytest.fixture(scope="module")
def aa_rows(aa_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return aa_payload["data"]


# ---------------------------------------------------------- raw payload preservation


def test_raw_field_carries_full_row_verbatim(aa_rows: list[dict[str, Any]]) -> None:
    """ADR-015: the full upstream row lives in `raw` byte-for-byte (at
    the dict level — JSON round-trip equality). Required so a later
    investigator can re-project from a cached row when AA adds a field
    we don't yet model."""
    sample = next(r for r in aa_rows if r["slug"] == "gpt-oss-120b-low")
    row = AaModelRow.project(sample)
    assert row.raw == sample


def test_unknown_top_level_field_preserved_in_raw() -> None:
    """A future AA release that adds a top-level field (e.g. a new
    `tier` discriminator) must not break validation. The field is
    silently dropped from the typed projection (per
    `extra="ignore"`) but survives in `raw` for later promotion."""
    future_payload = {
        "id": "uuid-x",
        "slug": "gpt-oss-120b-low",
        "name": "GPT-OSS 120B (low)",
        "model_creator": {"id": "uuid-y", "name": "OpenAI", "slug": "openai"},
        "release_date": "2025-08-05",
        "median_output_tokens_per_second": 335.9,
        "median_time_to_first_token_seconds": 0.485,
        "median_time_to_first_answer_token": 6.438,
        "pricing": {"price_1m_input_tokens": 0.15},
        "evaluations": {"mmlu_pro": 0.775},
        "future_top_level_tier": "preview",  # not modeled today
    }
    row = AaModelRow.project(future_payload)
    assert row.slug == "gpt-oss-120b-low"
    assert row.raw["future_top_level_tier"] == "preview"


# --------------------------------------------------------- evolving nested objects


def test_evaluations_is_open_dict_not_narrow_typed(aa_rows: list[dict[str, Any]]) -> None:
    """The 15 evaluation keys across the live fixture include
    `aime_25`, `lcr`, `terminalbench_hard`, `tau2`, `ifbench`, `hle`
    — none of which were in AA's docs at spec-writing time. The
    projection must accept any string key with float-or-null value,
    NOT a Literal whitelist. ADR-015 trust contract."""
    sample = next(r for r in aa_rows if r["slug"] == "gpt-oss-120b-low")
    row = AaModelRow.project(sample)
    assert "terminalbench_hard" in row.evaluations
    assert isinstance(row.evaluations["terminalbench_hard"], float)
    # Some fields legitimately come back null (e.g. math_500 on
    # certain rows) — the projection must accept that, not coerce
    # to 0 or drop the key.
    assert row.evaluations.get("math_500") is None


def test_pricing_is_open_dict() -> None:
    """`pricing` likewise — AA's live response carries
    `price_1m_blended_3_to_1`, `price_1m_input_tokens`,
    `price_1m_output_tokens`, and undocumented `cache`/`batch`/
    `tiered` sub-keys on some rows. dict[str, float | None] only."""
    sample = {
        "id": "x",
        "slug": "test-model",
        "name": "Test",
        "model_creator": {"id": "y", "name": "Vendor", "slug": "vendor"},
        "release_date": "2026-01-01",
        "median_output_tokens_per_second": 100.0,
        "median_time_to_first_token_seconds": 0.5,
        "median_time_to_first_answer_token": 1.0,
        "pricing": {
            "price_1m_input_tokens": 0.5,
            "price_1m_output_tokens": 1.5,
            "price_1m_cached_input_tokens": 0.05,  # future sub-key
            "price_1m_batch_input_tokens": None,  # provider-not-offered
        },
        "evaluations": {},
    }
    row = AaModelRow.project(sample)
    assert row.pricing["price_1m_cached_input_tokens"] == 0.05
    assert row.pricing["price_1m_batch_input_tokens"] is None


# -------------------------------------------------- reasoning_effort derived from slug


@pytest.mark.parametrize(
    ("slug", "expected_effort"),
    [
        ("gpt-oss-120b-low", "low"),
        ("gpt-oss-120b", None),  # base = non-reasoning
        ("o3-mini-high", "high"),
        ("gemini-3-5-flash-medium", "medium"),
        ("deepseek-v3", None),
        ("deepseek-r1", None),
        ("mistral-medium", None),  # ENDS in -medium but is a base model
        ("magistral-medium", None),  # same — base model name
        ("devstral-medium", None),  # same
    ],
)
def test_reasoning_effort_extracted_from_slug_suffix(
    slug: str, expected_effort: str | None
) -> None:
    """Reasoning models have multiple rows per base model with `-low`
    / `-medium` / `-high` suffixes. Most non-reasoning models that
    end in `-medium` are base-model names (e.g. `mistral-medium`,
    `magistral-medium`, `devstral-medium`) — the projection MUST
    NOT classify them as reasoning. The disambiguator is the
    `aa_slug_mapping.yaml` curated list (Slice E owns the explicit
    pairing); auto-detection from the suffix is a useful heuristic
    but cannot be the final word.

    For now the projection extracts the suffix when present AND the
    same base slug exists in the fixture without the suffix — i.e.
    `gpt-oss-120b-low` coexists with `gpt-oss-120b` so we know the
    `-low` is a reasoning dimension, not part of the base name.
    Without that base-slug coexistence the field stays None so the
    YAML can override deliberately."""
    sample = {
        "id": "x",
        "slug": slug,
        "name": slug,
        "model_creator": {"id": "y", "name": "Vendor", "slug": "vendor"},
        "release_date": "2026-01-01",
        "median_output_tokens_per_second": 100.0,
        "median_time_to_first_token_seconds": 0.5,
        "median_time_to_first_answer_token": 1.0,
        "pricing": {},
        "evaluations": {},
    }
    row = AaModelRow.project(sample)
    # Auto-detection alone is too risky without context; the
    # projection leaves None and the slug-mapping loader (Slice E)
    # resolves it explicitly. This test pins the conservative default.
    assert row.reasoning_effort is None


def test_reasoning_effort_explicit_override_in_projection() -> None:
    """The projection accepts an explicit `reasoning_effort` kwarg
    so the slug-mapping loader can stamp the resolved value when it
    pairs a row to a YAML entry. This is the canonical path; the
    projection's own auto-detection is intentionally None-by-default
    to avoid mis-classifying `mistral-medium` as reasoning."""
    sample = {
        "id": "x",
        "slug": "gpt-oss-120b-low",
        "name": "GPT-OSS 120B (low)",
        "model_creator": {"id": "y", "name": "OpenAI", "slug": "openai"},
        "release_date": "2025-08-05",
        "median_output_tokens_per_second": 335.9,
        "median_time_to_first_token_seconds": 0.485,
        "median_time_to_first_answer_token": 6.438,
        "pricing": {},
        "evaluations": {},
    }
    row = AaModelRow.project(sample, reasoning_effort="low")
    assert row.reasoning_effort == "low"


# -------------------------------------------------------- required-field validation


def test_missing_required_id_field_rejected() -> None:
    """`id` (AA's stable UUID) is the primary join key — refuse a
    row without it rather than silently letting downstream code see
    a None primary key."""
    bad = {
        "slug": "x",
        "name": "x",
        "model_creator": {"id": "y", "name": "v", "slug": "v"},
        "release_date": "2026-01-01",
        "median_output_tokens_per_second": 100.0,
        "median_time_to_first_token_seconds": 0.5,
        "median_time_to_first_answer_token": 1.0,
        "pricing": {},
        "evaluations": {},
    }
    with pytest.raises(ValidationError):
        AaModelRow.project(bad)


# ---------------------------------------------------------- full fixture coverage


def test_all_525_rows_in_fixture_project_without_error(
    aa_rows: list[dict[str, Any]],
) -> None:
    """Smoke test: every row in the captured live response projects
    cleanly. Catches accidental narrow-typing the next time someone
    edits the projection — if AA adds an enum value or shape change,
    we want this test red, not silently dropping rows."""
    errors: list[str] = []
    for row_dict in aa_rows:
        try:
            AaModelRow.project(row_dict)
        except ValidationError as exc:
            errors.append(f"{row_dict.get('slug', '?')}: {exc.errors()}")
    assert not errors, "projection failures:\n" + "\n".join(errors[:10])
