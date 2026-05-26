"""Round-trip integrity for the supplement YAMLs.

Catches the silent-default failure mode: a YAML row that's missing an
optional field gets a Pydantic-provided default at load time. If we
later `model_dump` and persist, the source-of-truth YAML grows the
default value automatically — and we lose the ability to tell at a
glance which rows were author-specified and which were defaults.

For both seed YAMLs, every key the source row carried must round-trip
through Pydantic identically. Defaults that weren't in the source
row must NOT appear in the dumped output (exclude_unset=True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from whatcanirun.catalog.loaders import load_gpu_supplements, load_quantizations

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GPUS = _REPO_ROOT / "seeds" / "gpus_supplement.yaml"
_QUANTS = _REPO_ROOT / "seeds" / "quantizations.yaml"


def _read_raw_rows(path: Path) -> list[dict[str, Any]]:
    return list(yaml.safe_load(path.read_text()))


def test_gpu_supplement_round_trips_authored_fields() -> None:
    raw_rows = _read_raw_rows(_GPUS)
    loaded = load_gpu_supplements(_GPUS)
    assert len(raw_rows) == len(loaded)
    for raw, model in zip(raw_rows, loaded, strict=True):
        dumped = model.model_dump(exclude_unset=True)
        assert dumped == raw, f"row {model.slug!r} mismatch:\n  source: {raw}\n  dumped: {dumped}"


def test_quantization_round_trips_authored_fields() -> None:
    raw_rows = _read_raw_rows(_QUANTS)
    loaded = load_quantizations(_QUANTS)
    assert len(raw_rows) == len(loaded)
    for raw, model in zip(raw_rows, loaded, strict=True):
        dumped = model.model_dump(exclude_unset=True)
        assert dumped == raw, f"row {model.slug!r} mismatch:\n  source: {raw}\n  dumped: {dumped}"
