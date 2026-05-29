"""YAML loaders for catalog supplement files.

Wraps `yaml.safe_load` + Pydantic validation and re-raises any error as
`SeedLoadError` carrying the file path (and the YAML parser's line/column
when available). The supplement YAMLs are our own controlled data, so a
malformed file should fail loudly enough to find the offending row.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from whatcanirun.catalog.seed_schemas import (
    GpuSupplement,
    Quantization,
    TrackedModelRow,
)
from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.pricing.artificial_analysis import AaSlugMappingRow


class SeedLoadError(Exception):
    """Raised when a supplement YAML fails to parse or validate."""


def _load_rows[Row: BaseModel](path: Path, row_model: type[Row]) -> list[Row]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SeedLoadError(f"{path}: cannot read file: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        loc = (
            f"line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else "unknown line"
        )
        raise SeedLoadError(f"{path}: YAML parse error at {loc}: {exc}") from exc

    if not isinstance(data, list):
        raise SeedLoadError(f"{path}: expected a YAML list at the root, got {type(data).__name__}")

    rows: list[Row] = []
    for index, raw_row in enumerate(data):
        if not isinstance(raw_row, dict):
            raise SeedLoadError(
                f"{path}: row #{index + 1} is not a mapping (got {type(raw_row).__name__})"
            )
        try:
            rows.append(row_model.model_validate(raw_row))
        except ValidationError as exc:
            raise SeedLoadError(
                f"{path}: row #{index + 1} (slug={raw_row.get('slug', '?')!r}) failed validation:\n{exc}"
            ) from exc
    return rows


def load_gpu_supplements(path: Path) -> list[GpuSupplement]:
    """Load `seeds/gpus_supplement.yaml` into validated `GpuSupplement` rows."""
    return _load_rows(path, GpuSupplement)


def load_quantizations(path: Path) -> list[Quantization]:
    """Load `seeds/quantizations.yaml` into validated `Quantization` rows."""
    return _load_rows(path, Quantization)


def load_workload_profiles(path: Path) -> list[WorkloadProfile]:
    """Load `seeds/workload_profiles.yaml` into validated
    `WorkloadProfile` rows.

    Enforces the cross-row "exactly one `is_default=True`"
    invariant — Pydantic can't see it because the rule spans the
    list and each `WorkloadProfile` only knows its own bool. With
    zero defaults, M09's `budget_to_plan` has no sensible fallback
    when the caller omits `workload_profile`; with two, the second
    silently shadows the first in any "find the default" lookup.
    Both modes fail loudly with the offending slugs named so the
    operator can diff them in their editor.
    """
    rows = _load_rows(path, WorkloadProfile)
    defaults = [row.slug for row in rows if row.is_default]
    if len(defaults) != 1:
        raise SeedLoadError(
            f"{path}: exactly one row must have `is_default: true`; "
            f"got {len(defaults)} default(s) (slugs: {defaults!r})"
        )
    return rows


def load_aa_slug_mapping(path: Path) -> list[AaSlugMappingRow]:
    """Load `seeds/aa_slug_mapping.yaml` into validated
    `AaSlugMappingRow` rows.

    Duplicate `cp_slug` values are rejected — two rows sharing a
    `cp_slug` would silently shadow each other in any dict-keyed
    lookup, defeating the curated-mapping guarantee. Same
    duplicate-detection pattern as `load_tracked_models`.
    """
    rows = _load_rows(path, AaSlugMappingRow)
    seen: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        if row.cp_slug in seen:
            raise SeedLoadError(
                f"{path}: duplicate cp_slug {row.cp_slug!r} appears at rows "
                f"#{seen[row.cp_slug]} and #{idx}; each cp_slug must appear at "
                f"most once per file"
            )
        seen[row.cp_slug] = idx
    return rows


def load_tracked_models(path: Path) -> list[TrackedModelRow]:
    """Load `seeds/tracked_models.yaml` (or a user_models.yaml extension
    file) into validated `TrackedModelRow` rows.

    Slugs must be unique within a single file. Two rows sharing the
    same `slug` is almost always a typo / merge-conflict footgun —
    without detection, both rows would load, both syncs would run,
    and the second would silently overwrite the first in the cache.
    Cross-file slug conflicts (seed vs user) are a different concern
    handled by `HfModelSync._load_merged_tracked_rows` with a
    seed-wins policy.
    """
    rows = _load_rows(path, TrackedModelRow)
    seen: dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        if row.slug in seen:
            raise SeedLoadError(
                f"{path}: duplicate slug {row.slug!r} appears at rows "
                f"#{seen[row.slug]} and #{idx}; each slug must appear at most "
                f"once per file"
            )
        seen[row.slug] = idx
    return rows
