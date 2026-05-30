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
    CheckContext,
    CheckResult,
    check_batch_scaling_not_linear,
    check_decode_tps_vs_bandwidth_heuristic,
    check_engine_version_format,
    check_gpu_form_factor_disambiguated,
    check_gpu_slug_exists,
    check_measured_at_recency,
    check_methodology_complete,
    check_model_slug_exists,
    check_op_point_unique,
    check_quant_slug_exists,
    check_source_url_well_formed,
)

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.seed_schemas import Quantization, TrackedModelRow
from whatcanirun.pricing.projections import GpuCatalogRow


def _gpu(slug: str = "h100") -> GpuCatalogRow:
    """Minimal GpuCatalogRow for join tests."""
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="hopper",
        vram_gb=80,
        release_date=None,
        specs={"memory_bandwidth_gbps": 3350.0},
    )


def _quant(slug: str = "bf16", bits: int = 16) -> Quantization:
    """Minimal Quantization for join tests."""
    return Quantization(
        slug=slug,
        bits_per_weight=bits,
        kv_cache_bits_default=16,
        introduced_architecture="ampere",
        notes="seed test quant",
    )


def _tracked(slug: str = "llama-3-1-8b", hf_repo_id: str | None = None) -> TrackedModelRow:
    """Minimal TrackedModelRow for join tests."""
    return TrackedModelRow(
        slug=slug,
        hf_repo_id=hf_repo_id or f"meta-llama/{slug}",
    )


def _ctx(
    existing_cells: list[BenchmarkCell] | None = None,
    gpu_catalog: list[GpuCatalogRow] | None = None,
    quantizations: list[Quantization] | None = None,
    tracked_models: list[TrackedModelRow] | None = None,
) -> CheckContext:
    """Builds a CheckContext with sensible defaults. Tests override
    only the field they're exercising."""
    return CheckContext(
        existing_cells=existing_cells if existing_cells is not None else [],
        gpu_catalog=gpu_catalog if gpu_catalog is not None else [_gpu()],
        quantizations=quantizations if quantizations is not None else [_quant()],
        tracked_models=tracked_models if tracked_models is not None else [_tracked()],
    )


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


# ---------------------------------------------------------------- check_op_point_unique


class TestOpPointUnique:
    """Reject candidate rows whose op-point key
    `(gpu_slug, model_slug, quant_slug, tp_size, batch_size, context_length)`
    already exists in the canonical parquet. Forces the curator
    to be explicit about overriding an existing cell rather than
    silently shadowing one. If a re-measurement legitimately
    supersedes an old cell, the curator must delete the old row
    in the same PR."""

    def test_unique_op_point_passes(self) -> None:
        cell = _valid_cell(gpu_slug="h200", model_slug="llama-3-3-70b", quant_slug="fp8")
        existing = [_valid_cell(gpu_slug="h100", model_slug="llama-3-3-70b", quant_slug="fp8")]
        ctx = _ctx(existing_cells=existing)
        result = check_op_point_unique(cell, ctx)
        assert result.severity == "pass"

    def test_exact_op_point_collision_errors(self) -> None:
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="llama-3-3-70b",
            quant_slug="fp8",
            tp_size=1,
            batch_size=1,
            context_length=4096,
        )
        existing = [
            _valid_cell(
                gpu_slug="h100",
                model_slug="llama-3-3-70b",
                quant_slug="fp8",
                tp_size=1,
                batch_size=1,
                context_length=4096,
                decode_tps=999.0,
            )
        ]
        ctx = _ctx(existing_cells=existing)
        result = check_op_point_unique(cell, ctx)
        assert result.severity == "error"
        assert "op-point" in result.message.lower() or "duplicate" in result.message.lower()

    def test_same_model_different_batch_passes(self) -> None:
        # batch_size is part of the op-point key; same model with
        # different batch is a legitimate new op-point.
        cell = _valid_cell(batch_size=8)
        existing = [_valid_cell(batch_size=1)]
        ctx = _ctx(existing_cells=existing)
        result = check_op_point_unique(cell, ctx)
        assert result.severity == "pass"

    def test_empty_existing_passes(self) -> None:
        # First-ever cell with that op-point.
        cell = _valid_cell()
        ctx = _ctx(existing_cells=[])
        result = check_op_point_unique(cell, ctx)
        assert result.severity == "pass"


# ---------------------------------------------------------------- check_gpu_slug_exists


class TestGpuSlugExists:
    """Cell's gpu_slug must resolve to a row in CP's `gpu_catalog`.
    Without an exact match, tps_estimator Tier 1b can't join the
    cell to a real GPU's bandwidth + form factor — the row is dead
    data."""

    def test_known_gpu_passes(self) -> None:
        cell = _valid_cell(gpu_slug="h100")
        ctx = _ctx(gpu_catalog=[_gpu("h100"), _gpu("a100")])
        result = check_gpu_slug_exists(cell, ctx)
        assert result.severity == "pass"

    def test_unknown_gpu_errors(self) -> None:
        cell = _valid_cell(gpu_slug="rtx-9090")
        ctx = _ctx(gpu_catalog=[_gpu("h100")])
        result = check_gpu_slug_exists(cell, ctx)
        assert result.severity == "error"
        assert "gpu_slug" in result.message
        assert "rtx-9090" in result.message

    def test_empty_catalog_errors(self) -> None:
        # Edge case: catalog hasn't been loaded; better to fail loud
        # than silently pass every gpu_slug.
        cell = _valid_cell(gpu_slug="h100")
        ctx = _ctx(gpu_catalog=[])
        result = check_gpu_slug_exists(cell, ctx)
        assert result.severity == "error"


# ---------------------------------------------------------------- check_quant_slug_exists


class TestQuantSlugExists:
    """Cell's quant_slug must resolve to a row in
    `seeds/quantizations.yaml`. Without that, fit_check can't get
    `bits_per_weight` and tps_estimator can't compute memory traffic."""

    def test_known_quant_passes(self) -> None:
        cell = _valid_cell(quant_slug="fp8")
        ctx = _ctx(quantizations=[_quant("fp8", 8), _quant("bf16", 16)])
        result = check_quant_slug_exists(cell, ctx)
        assert result.severity == "pass"

    def test_unknown_quant_errors(self) -> None:
        cell = _valid_cell(quant_slug="int3")  # not a real quant
        ctx = _ctx(quantizations=[_quant("fp8", 8)])
        result = check_quant_slug_exists(cell, ctx)
        assert result.severity == "error"
        assert "quant_slug" in result.message
        assert "int3" in result.message


# ---------------------------------------------------------------- check_model_slug_exists


class TestModelSlugExists:
    """Cell's model_slug must resolve to a row in merged
    tracked_models (project seed + user_models.yaml). Without that,
    HfModelSync can't find the HF repo and the cell's join keys
    are stranded."""

    def test_known_model_passes(self) -> None:
        cell = _valid_cell(model_slug="llama-3-1-8b")
        ctx = _ctx(
            tracked_models=[
                _tracked("llama-3-1-8b"),
                _tracked("mistral-7b"),
            ]
        )
        result = check_model_slug_exists(cell, ctx)
        assert result.severity == "pass"

    def test_unknown_model_errors(self) -> None:
        cell = _valid_cell(model_slug="mistral-large")  # not yet in tracked_models
        ctx = _ctx(tracked_models=[_tracked("llama-3-1-8b")])
        result = check_model_slug_exists(cell, ctx)
        assert result.severity == "error"
        assert "model_slug" in result.message
        assert "mistral-large" in result.message
        # The error should hint at the resolution path: add to tracked_models.
        assert "tracked_models" in result.message


# ---------------------------------------------------------------- check_gpu_form_factor_disambiguated


class TestGpuFormFactorDisambiguated:
    """Per the M10 pitfall: "H100 in a benchmark might mean H100 SXM
    or PCIe. They have different bandwidth." For cells whose notes
    mention a data-center GPU known to ship in multiple form factors
    (H100, H200, A100), the notes must ALSO mention which form
    factor (SXM, PCIe, NVL, OAM) was tested. Otherwise the cell's
    bandwidth assumptions are ambiguous."""

    def test_h100_with_sxm_passes(self) -> None:
        cell = _valid_cell(
            notes=(
                "Single H100 SXM5, fp8, batch=1, ctx=4096. vLLM 0.6.x "
                "with paged_attention. Reference run from blog."
            )
        )
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "pass"

    def test_h100_with_pcie_passes(self) -> None:
        cell = _valid_cell(notes=("H100 PCIe, fp8, batch=1, ctx=4096. vLLM 0.6.x reference run."))
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "pass"

    def test_h100_without_form_factor_errors(self) -> None:
        cell = _valid_cell(
            notes=(
                "H100, fp8, batch=1, ctx=4096. vLLM 0.6.x reference "
                "run from blog (no form factor disclosed)."
            )
        )
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "error"
        assert "form factor" in result.message.lower() or "sxm" in result.message.lower()

    def test_h200_without_form_factor_errors(self) -> None:
        cell = _valid_cell(notes=("H200, fp8, batch=1, ctx=4096. vLLM 0.6.x reference."))
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "error"

    def test_a100_with_form_factor_passes(self) -> None:
        cell = _valid_cell(
            notes=("A100 80GB SXM, bf16, batch=1, ctx=4096. vLLM 0.6.x reference run from blog.")
        )
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "pass"

    def test_l40s_no_disambiguation_required_passes(self) -> None:
        # L40S only ships in one form factor (PCIe), so the notes
        # need not call it out. Avoid false-positive errors on
        # single-form-factor GPUs.
        cell = _valid_cell(notes=("L40S, bf16, batch=1, ctx=4096. vLLM 0.6.x reference."))
        result = check_gpu_form_factor_disambiguated(cell)
        assert result.severity == "pass"


# ---------------------------------------------------------------- check_batch_scaling_not_linear


class TestBatchScalingNotLinear:
    """ADR-010: TPS does NOT scale linearly with batch size. Verified
    6x wrong at batch=128. A candidate cell with batch>1 should
    have decode_tps that's noticeably sub-linear vs its single-stream
    peer. If the cell's ratio (batched_tps / (batch * single_tps))
    is >= 0.85, that's suspicious — either the source reported per-
    batch throughput as if it were per-stream, or methodology is
    ambiguous. Error."""

    def test_batch_one_skipped(self) -> None:
        # batch=1 cells don't have a scaling claim; the check
        # passes trivially.
        cell = _valid_cell(batch_size=1, decode_tps=100.0)
        ctx = _ctx(existing_cells=[])
        result = check_batch_scaling_not_linear(cell, ctx)
        assert result.severity == "pass"

    def test_sublinear_batch_passes(self) -> None:
        # batch=8 at 350 tok/s vs single at 100 tok/s → ratio 0.44,
        # well below 0.85 — believable batched throughput.
        cell = _valid_cell(batch_size=8, decode_tps=350.0)
        existing = [_valid_cell(batch_size=1, decode_tps=100.0)]
        ctx = _ctx(existing_cells=existing)
        result = check_batch_scaling_not_linear(cell, ctx)
        assert result.severity == "pass"

    def test_near_linear_batch_errors(self) -> None:
        # batch=8 at 800 tok/s vs single at 100 tok/s → ratio 1.0,
        # impossibly linear; either the source mislabeled or it's
        # per-stream throughput presented as per-batch.
        cell = _valid_cell(batch_size=8, decode_tps=800.0)
        existing = [_valid_cell(batch_size=1, decode_tps=100.0)]
        ctx = _ctx(existing_cells=existing)
        result = check_batch_scaling_not_linear(cell, ctx)
        assert result.severity == "error"
        assert "linear" in result.message.lower() or "batch" in result.message.lower()

    def test_no_single_stream_peer_warns(self) -> None:
        # Batched cell with no single-stream peer in existing rows;
        # we can't verify so warn rather than pass or error.
        cell = _valid_cell(batch_size=8, decode_tps=350.0)
        ctx = _ctx(existing_cells=[])
        result = check_batch_scaling_not_linear(cell, ctx)
        assert result.severity == "warn"
        assert "peer" in result.message.lower() or "single-stream" in result.message.lower()


# ---------------------------------------------------------------- check_decode_tps_vs_bandwidth_heuristic


class TestDecodeTpsVsBandwidthHeuristic:
    """Cross-check the cell's decode_tps against the same bandwidth
    heuristic tps_estimator Tier 3 uses. For batch=1 only — at
    batch>1 the heuristic doesn't apply per ADR-010.

    MoE-aware: when the tracked_models row carries active_params_b
    (sparse model), use it; otherwise fall back to total_params_b.
    Naively using total_params_b for MoE produces wildly-off
    predictions (the prototype caught DeepSeek-V3 at a 404% over-
    prediction this way).

    Returns warn if predicted/actual is outside [0.5, 1.5]; that's
    a curator hint, not a blocking error — the actual may legitimately
    differ from the heuristic (kernel efficiency varies, sparse
    activation, speculative decoding, etc.).

    Skips the check entirely (returns pass with 'skipped' message)
    when any of bandwidth / params / bits_per_weight is missing —
    the curator can't be blamed for the data gap."""

    def _h100_ctx(self, **tracked_overrides: Any) -> CheckContext:
        return _ctx(
            gpu_catalog=[_gpu("h100")],  # bandwidth 3350 GB/s
            quantizations=[_quant("bf16", 16), _quant("fp8", 8)],
            tracked_models=[
                TrackedModelRow(
                    slug="llama-3-1-8b",
                    hf_repo_id="meta-llama/Meta-Llama-3.1-8B",
                    total_params_b=8.0,
                    **tracked_overrides,
                )
            ],
        )

    def test_in_band_actual_passes(self) -> None:
        # 8B bf16 on H100: predicted ~157 tok/s. Actual 130 → ratio
        # 0.83, in band.
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="llama-3-1-8b",
            quant_slug="bf16",
            batch_size=1,
            decode_tps=130.0,
        )
        ctx = self._h100_ctx()
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        assert result.severity == "pass"

    def test_actual_much_lower_warns(self) -> None:
        # 8B bf16 on H100: predicted ~157 tok/s. Actual 40 → ratio
        # 0.25, out of [0.5, 1.5].
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="llama-3-1-8b",
            quant_slug="bf16",
            batch_size=1,
            decode_tps=40.0,
        )
        ctx = self._h100_ctx()
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        assert result.severity == "warn"
        assert "predicted" in result.message.lower() or "heuristic" in result.message.lower()

    def test_actual_much_higher_warns(self) -> None:
        # 8B bf16 on H100: predicted ~157 tok/s. Actual 400 → ratio
        # 2.5, out of [0.5, 1.5].
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="llama-3-1-8b",
            quant_slug="bf16",
            batch_size=1,
            decode_tps=400.0,
        )
        ctx = self._h100_ctx()
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        assert result.severity == "warn"

    def test_batch_greater_than_one_skipped(self) -> None:
        # At batch>1 the heuristic doesn't apply; skip cleanly.
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="llama-3-1-8b",
            quant_slug="bf16",
            batch_size=8,
            decode_tps=350.0,
        )
        ctx = self._h100_ctx()
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        assert result.severity == "pass"
        assert "skip" in result.message.lower() or "batch" in result.message.lower()

    def test_missing_params_skipped(self) -> None:
        # No total_params_b → can't compute; pass with 'skipped'.
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="unknown-7b",
            quant_slug="bf16",
            batch_size=1,
            decode_tps=100.0,
        )
        ctx = _ctx(
            gpu_catalog=[_gpu("h100")],
            quantizations=[_quant("bf16", 16)],
            tracked_models=[
                TrackedModelRow(
                    slug="unknown-7b",
                    hf_repo_id="example/unknown-7b",
                    # total_params_b deliberately omitted
                )
            ],
        )
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        assert result.severity == "pass"
        assert "skip" in result.message.lower() or "missing" in result.message.lower()

    def test_moe_uses_active_params(self) -> None:
        # MoE model — total=685B, active=37B (DeepSeek-V3-ish).
        # Using total naively gives prediction ~3.7 tok/s (way
        # below 18.5 actual). Using active gives ~68 tok/s, so
        # ratio = 18.5/68 = 0.27 → warn (still out of band but a
        # MUCH closer / more useful diagnostic).
        cell = _valid_cell(
            gpu_slug="h100",
            model_slug="deepseek-v3",
            quant_slug="fp8",
            batch_size=1,
            decode_tps=18.5,
        )
        ctx = _ctx(
            gpu_catalog=[_gpu("h100")],
            quantizations=[_quant("fp8", 8)],
            tracked_models=[
                TrackedModelRow(
                    slug="deepseek-v3",
                    hf_repo_id="deepseek-ai/DeepSeek-V3",
                    total_params_b=685.0,
                    active_params_b=37.0,
                )
            ],
        )
        result = check_decode_tps_vs_bandwidth_heuristic(cell, ctx)
        # The check should NOT just say "predicted 3.7" (naive total)
        # — it should reference the active-params figure in the message.
        assert "active" in result.message.lower() or "37" in result.message


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
