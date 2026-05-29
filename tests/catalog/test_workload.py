"""Slice A: `WorkloadProfile` Pydantic schema with strict validation.

Workload profiles are OUR OWN curated data (3 hand-picked rows in
`seeds/workload_profiles.yaml`), so the schema uses `extra="forbid"`
— a typo in the YAML must fail loudly, not silently drop into an
unknown-field bucket. Mirrors the convention for the other
controlled-data schemas in `seed_schemas.py` (`GpuSupplement`,
`Quantization`, `TrackedModelRow`).

Token counts must be positive: zero or negative would let
`budget_to_plan` divide by zero or produce nonsense prompt counts.
Caught at the row-validation boundary so the failure mode is
"YAML rejected at load" rather than "trust envelope ships a
non-finite est_total_prompts".
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whatcanirun.catalog.workload import WorkloadProfile

# ----------------------------------------------------------- shape


def test_minimal_valid_row_parses() -> None:
    """The 6 required fields all populated → construction succeeds."""
    row = WorkloadProfile(
        slug="chat_assistant",
        display_name="Chat assistant",
        avg_input_tokens=400,
        avg_output_tokens=250,
        is_default=True,
        description="Conversational use.",
    )
    assert row.slug == "chat_assistant"
    assert row.is_default is True


# ----------------------------------------------------------- extra=forbid


def test_unknown_field_rejected() -> None:
    """`extra="forbid"` — a typo in the YAML (e.g. `is_defualt`
    instead of `is_default`) must fail validation rather than
    silently drop into an unknown-field bucket. Catching at the
    row-validation boundary is much earlier than catching at
    "every profile is non-default; budget_to_plan crashes when no
    profile is selected"."""
    with pytest.raises(ValidationError):
        WorkloadProfile(
            slug="x",
            display_name="x",
            avg_input_tokens=1,
            avg_output_tokens=1,
            is_default=False,
            description="x",
            unexpected_field="oops",  # type: ignore[call-arg]
        )


# ----------------------------------------------------------- positive tokens


@pytest.mark.parametrize("bad_value", [0, -1, -1000])
def test_avg_input_tokens_must_be_positive(bad_value: int) -> None:
    """Zero or negative input tokens would let downstream
    `budget_to_plan` divide by zero or produce a negative prompt
    count. Catch at the row boundary so the trust envelope never
    has to surface a non-finite `est_total_prompts`."""
    with pytest.raises(ValidationError, match="positive"):
        WorkloadProfile(
            slug="x",
            display_name="x",
            avg_input_tokens=bad_value,
            avg_output_tokens=1,
            is_default=False,
            description="x",
        )


@pytest.mark.parametrize("bad_value", [0, -1, -1000])
def test_avg_output_tokens_must_be_positive(bad_value: int) -> None:
    """Same rationale for output tokens — zero output is a
    legitimate-looking but nonsensical profile (you'd ask the
    model for 0 tokens), and negative is impossible. Reject both."""
    with pytest.raises(ValidationError, match="positive"):
        WorkloadProfile(
            slug="x",
            display_name="x",
            avg_input_tokens=1,
            avg_output_tokens=bad_value,
            is_default=False,
            description="x",
        )


# ----------------------------------------------------------- required fields


@pytest.mark.parametrize(
    "missing_field",
    [
        "slug",
        "display_name",
        "avg_input_tokens",
        "avg_output_tokens",
        "is_default",
        "description",
    ],
)
def test_missing_required_field_rejected(missing_field: str) -> None:
    """All 6 fields are required — no defaults, since each carries
    semantic meaning the caller must supply explicitly."""
    base = {
        "slug": "x",
        "display_name": "x",
        "avg_input_tokens": 1,
        "avg_output_tokens": 1,
        "is_default": False,
        "description": "x",
    }
    del base[missing_field]
    with pytest.raises(ValidationError):
        WorkloadProfile(**base)  # type: ignore[arg-type]


# ============================================================ loader (Slice B+C)


from pathlib import Path  # noqa: E402

from whatcanirun.catalog.loaders import SeedLoadError, load_workload_profiles  # noqa: E402

_VALID_YAML = """\
- slug: code_completion
  display_name: "Code completion"
  avg_input_tokens: 800
  avg_output_tokens: 120
  is_default: false
  description: "Editor-integrated completion."
- slug: chat_assistant
  display_name: "Chat assistant"
  avg_input_tokens: 400
  avg_output_tokens: 250
  is_default: true
  description: "Conversational use."
- slug: batch_eval
  display_name: "Batch eval"
  avg_input_tokens: 1200
  avg_output_tokens: 200
  is_default: false
  description: "Long context, short output."
"""


def test_loads_three_valid_rows(tmp_path: Path) -> None:
    """Happy path: 3 rows in → 3 `WorkloadProfile` objects out, in
    YAML order."""
    f = tmp_path / "w.yaml"
    f.write_text(_VALID_YAML)
    rows = load_workload_profiles(f)
    assert [r.slug for r in rows] == ["code_completion", "chat_assistant", "batch_eval"]
    assert sum(r.is_default for r in rows) == 1


def test_unknown_field_in_yaml_rejected(tmp_path: Path) -> None:
    """The Pydantic `extra="forbid"` propagates through the loader's
    `SeedLoadError` wrapper — typo in YAML gets caught at load."""
    f = tmp_path / "w.yaml"
    f.write_text(
        _VALID_YAML + "  bogus_field: oops\n"  # appended to the last row
    )
    with pytest.raises(SeedLoadError):
        load_workload_profiles(f)


def test_zero_default_rows_rejected(tmp_path: Path) -> None:
    """Cross-row invariant: exactly one `is_default=True`. With
    none, `budget_to_plan` has no sensible fallback when the caller
    omits `workload_profile` — better to fail at load with a clear
    message than silently degrade later."""
    yaml = """\
- slug: a
  display_name: "A"
  avg_input_tokens: 1
  avg_output_tokens: 1
  is_default: false
  description: "x"
- slug: b
  display_name: "B"
  avg_input_tokens: 1
  avg_output_tokens: 1
  is_default: false
  description: "x"
"""
    f = tmp_path / "w.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError, match="exactly one"):
        load_workload_profiles(f)


def test_multiple_default_rows_rejected(tmp_path: Path) -> None:
    """Symmetric: two rows both marked default is a typo /
    merge-conflict footgun — the second one would silently shadow
    the first in any "find the default" lookup. Catch at load with
    a message that names BOTH offending slugs so the operator can
    diff them."""
    yaml = """\
- slug: a
  display_name: "A"
  avg_input_tokens: 1
  avg_output_tokens: 1
  is_default: true
  description: "x"
- slug: b
  display_name: "B"
  avg_input_tokens: 1
  avg_output_tokens: 1
  is_default: true
  description: "x"
"""
    f = tmp_path / "w.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError, match="exactly one") as exc_info:
        load_workload_profiles(f)
    # Both offending slugs should appear in the error message so
    # the operator can find them in their editor immediately.
    assert "a" in str(exc_info.value)
    assert "b" in str(exc_info.value)


def test_yaml_parse_error_surfaced_with_line_number(tmp_path: Path) -> None:
    """Acceptance criterion: loader raises with line number on
    malformed YAML. Inherits the M01-era _load_rows formatting."""
    f = tmp_path / "w.yaml"
    f.write_text("- slug: a\n  display_name: [unterminated\n")
    with pytest.raises(SeedLoadError, match="line"):
        load_workload_profiles(f)


# ============================================================ seed file smoke


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_FILE = _REPO_ROOT / "seeds" / "workload_profiles.yaml"


def test_seeds_workload_profiles_yaml_loads() -> None:
    """The shipped `seeds/workload_profiles.yaml` parses with no
    schema drift between loader and committed YAML. Acceptance
    criterion: file has exactly 3 rows + exactly one default."""
    rows = load_workload_profiles(_SEED_FILE)
    assert len(rows) == 3
    defaults = [r.slug for r in rows if r.is_default]
    assert defaults == ["chat_assistant"], (
        f"expected chat_assistant as the sole default, got {defaults!r}"
    )


def test_seed_slugs_are_stable_set() -> None:
    """Pin the slug set so a drive-by edit can't silently swap one
    profile out from under M09's `budget_to_plan` (which keys on
    these strings)."""
    rows = load_workload_profiles(_SEED_FILE)
    assert {r.slug for r in rows} == {
        "code_completion",
        "chat_assistant",
        "batch_eval",
    }


def test_negative_token_count_in_yaml_rejected(tmp_path: Path) -> None:
    """Pydantic field_validator from Slice A propagates through the
    loader — bad row blocks load even if the rest of the file is
    syntactically fine."""
    yaml = """\
- slug: a
  display_name: "A"
  avg_input_tokens: -50
  avg_output_tokens: 1
  is_default: true
  description: "x"
"""
    f = tmp_path / "w.yaml"
    f.write_text(yaml)
    with pytest.raises(SeedLoadError, match="positive"):
        load_workload_profiles(f)
