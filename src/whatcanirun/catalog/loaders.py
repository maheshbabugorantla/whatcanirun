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


def load_tracked_models(path: Path) -> list[TrackedModelRow]:
    """Load `seeds/tracked_models.yaml` (or a user_models.yaml extension
    file) into validated `TrackedModelRow` rows."""
    return _load_rows(path, TrackedModelRow)
