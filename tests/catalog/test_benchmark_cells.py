"""M10 `BenchmarkCell` Pydantic schema — minimum viable shape M07
needs to run Tier 1a/1b lookups.

M10 owns the seed `benchmark_cells.parquet` (20-30 hand-curated
rows from public sources); M07 needs the row TYPE to write
`estimate_tps`. This test suite lives in `tests/catalog/` because
the schema itself lives under `whatcanirun.catalog`, not because
M07 is consuming the test fixtures.

Critical v1 invariant per spec/M10: `source="own_measured"` is
forbidden in v1 — enforced by validator, NOT just convention.
v2's M17 introduces own_measured cells via GuideLLM runs.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from whatcanirun.catalog.benchmark_cells import BenchmarkCell


def _valid_kwargs() -> dict:
    return dict(
        gpu_slug="h100",
        model_slug="llama-3-3-70b",
        quant_slug="fp8",
        tp_size=1,
        batch_size=1,
        context_length=4096,
        decode_tps=35.2,
        prefill_tps=None,
        ttft_ms=None,
        engine="vllm",
        engine_version="0.6.x",
        measured_at=date(2026, 3, 15),
        source="public_benchmark_anchor",
        source_url="https://www.spheron.network/blog/llama-3-3-70b-fp8",
        notes="Single H100 SXM, FP8, batch=1, ctx=4096. vLLM 0.6.x paged_attention.",
    )


def test_valid_v1_anchor_row_parses() -> None:
    row = BenchmarkCell(**_valid_kwargs())
    assert row.source == "public_benchmark_anchor"
    assert row.decode_tps == 35.2


def test_extra_field_rejected() -> None:
    """`extra="forbid"` — our own data; a typo in the seed file
    must fail loudly, not silently drop into an unknown bucket."""
    kwargs = _valid_kwargs() | {"surprise_field": "oops"}
    with pytest.raises(ValidationError):
        BenchmarkCell(**kwargs)


def test_v1_rejects_source_own_measured() -> None:
    """The v1 invariant: `source="own_measured"` is forbidden by
    validator. M17 (v2) will introduce GuideLLM-measured cells;
    until then, every committed row MUST be
    `source="public_benchmark_anchor"`. Spec/M10 line 28
    explicitly: 'The Pydantic schema rejects source="own_measured"
    in v1. This is enforced by validator, not just convention.'"""
    kwargs = _valid_kwargs() | {"source": "own_measured"}
    with pytest.raises(ValidationError, match="own_measured"):
        BenchmarkCell(**kwargs)


def test_engine_enum_is_closed_literal() -> None:
    """vllm | sglang | tensorrt_llm | tgi | other. Anything else
    fails validation — we don't want `vLLM` (capitalization typo)
    or `sgl` (truncation typo) silently coerced to a string."""
    kwargs = _valid_kwargs() | {"engine": "tritom"}
    with pytest.raises(ValidationError):
        BenchmarkCell(**kwargs)


def test_decode_tps_required() -> None:
    """No anchor cell without a decode_tps — `prefill_tps` and
    `ttft_ms` are optional, but the load-bearing decode rate
    must always be present (otherwise the row isn't actually an
    anchor M07 can use)."""
    kwargs = _valid_kwargs()
    del kwargs["decode_tps"]
    with pytest.raises(ValidationError):
        BenchmarkCell(**kwargs)


def test_source_url_required_for_anchor() -> None:
    """Trust contract: every `public_benchmark_anchor` row MUST
    have a `source_url` so the LLM can disclose the methodology
    citation. Empty string allowed? No — it'd defeat the
    transparency purpose. Make it strictly non-empty."""
    kwargs = _valid_kwargs() | {"source_url": ""}
    with pytest.raises(ValidationError):
        BenchmarkCell(**kwargs)


def test_notes_field_present_even_if_short() -> None:
    """Notes carry the methodology summary that the trust envelope
    surfaces. Empty notes is acceptable schema-wise (some seeds
    may have terse rows), but the field itself is required so a
    typo can't drop it."""
    kwargs = _valid_kwargs() | {"notes": ""}
    row = BenchmarkCell(**kwargs)
    assert row.notes == ""
