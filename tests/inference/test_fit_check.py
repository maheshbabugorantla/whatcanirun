"""M06 `compute_fit` — pure-math VRAM verdict.

The spec defines per-cycle test inputs in § Vertical slices; each
test below corresponds to one slice. Tests assert against values
computed FROM the documented formulas in spec/M06 § Formulas, not
against the prose-level "fits" / "doesn't fit" labels in the spec
narrative — when a narrative example doesn't match its own
formula (e.g. Llama-3.3-70B FP8 single H100 is described as
"fits" but 70.6GB + 15% overhead + KV exceeds 80GB), the formula
is authoritative and the narrative gets filed as a spec note.

Test fixtures construct synthetic `Model` / `GpuCatalogRow` /
`Quantization` instances rather than relying on shipped catalog
rows — `compute_fit` is a pure function with no I/O, so it makes
no sense to gate the tests on having a populated cache.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.inference.fit_check import FitResult, compute_fit
from whatcanirun.pricing.projections import GpuCatalogRow

# ---------------------------------------------------------------- factories


def _model(
    *,
    slug: str = "test-model",
    total_params_b: float = 70.6,
    active_params_b: float | None = None,
    n_layers: int = 80,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    kv_cache_strategy: str = "standard_gqa",
    raw_config: dict[str, Any] | None = None,
) -> Model:
    return Model(
        slug=slug,
        hf_repo_id=f"vendor/{slug}",
        display_name=slug,
        total_params_b=total_params_b,
        active_params_b=active_params_b,
        n_layers=n_layers,
        n_attention_heads=64,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=8192,
        max_position_embeddings=131072,
        native_dtype="bfloat16",
        architecture_family="llama",
        kv_cache_strategy=kv_cache_strategy,  # type: ignore[arg-type]
        raw_config=raw_config or {},
        raw_safetensors_meta={},
        hf_revision_sha="x",
        last_synced_at=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


def _gpu(slug: str, vram_gb: int) -> GpuCatalogRow:
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="Hopper",
        vram_gb=vram_gb,
        release_date=None,
        specs={},
        raw={},
    )


def _quant(slug: str, bits_per_weight: int, kv_cache_bits_default: int) -> Quantization:
    return Quantization(
        slug=slug,
        bits_per_weight=bits_per_weight,
        kv_cache_bits_default=kv_cache_bits_default,
        introduced_architecture="Hopper",
        notes="",
        experimental=False,
    )


# Reusable named factories matching the spec's example models.
def _llama_3_3_70b() -> Model:
    return _model(slug="llama-3-3-70b", total_params_b=70.6)


def _qwen3_coder_30b() -> Model:
    """Spec example. 30B dense, standard GQA."""
    return _model(
        slug="qwen-3-coder-30b",
        total_params_b=30.0,
        n_kv_heads=8,
    )


def _deepseek_v3() -> Model:
    """MLA architecture — raw_config carries kv_lora_rank,
    qk_rope_head_dim. M03 fixture captured these live."""
    return _model(
        slug="deepseek-v3",
        total_params_b=671.0,
        active_params_b=37.0,
        n_layers=61,
        n_kv_heads=128,
        kv_cache_strategy="mla",
        raw_config={
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
        },
    )


def _mixtral_8x22b() -> Model:
    """MoE. total_params_b for memory, active_params_b for compute
    (the latter is M07's, not ours — pin that fit_check uses total)."""
    return _model(
        slug="mixtral-8x22b",
        total_params_b=141.0,
        active_params_b=39.0,
        n_layers=56,
        n_kv_heads=8,
    )


# Reusable GPU fixtures from CP catalog (real vram values).
def _l40s() -> GpuCatalogRow:
    return _gpu("l40s", 48)


def _h100_sxm() -> GpuCatalogRow:
    return _gpu("h100", 80)


# Reusable quant fixtures from M01 seed (real bit counts).
def _fp16() -> Quantization:
    return _quant("fp16", 16, 16)


def _fp8() -> Quantization:
    return _quant("fp8", 8, 8)


def _int4() -> Quantization:
    return _quant("int4", 4, 8)


# ---------------------------------------------------------------- Slice 1


def test_llama_70b_fp8_l40s_does_not_fit() -> None:
    """Spec slice 1: weights alone (70.6 GB at FP8) exceed L40S
    48 GB. `fits=False` with `blocking_reasons` naming `weights`.

    Concrete math:
      weight_gb = 70.6 * 8/8 = 70.6
      framework_overhead = max(2, 0.15 * 70.6) = 10.59
      KV (FP8, ctx=4096, batch=1, 8 kv_heads, head_dim=128, 80 layers):
        2 * 80 * 8 * 128 * 4096 * 1 * 1 / 1e9 = 0.67 GB
      total = 70.6 + 10.59 + 0.67 = 81.86 GB
      available = 48 GB
      → fits=False, headroom = -33.86 GB
    """
    result = compute_fit(
        model=_llama_3_3_70b(),
        gpu=_l40s(),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    assert result.fits is False
    assert result.weight_gb == pytest.approx(70.6, rel=1e-9)
    assert result.framework_overhead_gb == pytest.approx(10.59, rel=1e-3)
    assert result.available_gb == 48
    assert result.headroom_gb < 0
    # Blocking reasons must NAME the weights as the dominant cause —
    # 70.6 GB alone is already > 48 GB.
    assert any("weight" in r.lower() for r in result.blocking_reasons)
    # Sufficiency caveat is MANDATORY on every result (spec §
    # critique round 4: fits=True is necessary not sufficient).
    assert result.sufficiency_caveat
    assert "necessary" in result.sufficiency_caveat
    # FitResult is a Pydantic instance per spec § Public surface.
    assert isinstance(result, FitResult)


# ---------------------------------------------------------------- Slice 2


def test_qwen3_coder_30b_int4_l40s_fits_with_positive_headroom() -> None:
    """Spec slice 2. INT4 quantization brings 30B weights down to
    15 GB. L40S 48 GB easily accommodates.

      weight_gb = 30 * 4/8 = 15
      overhead = max(2, 0.15 * 15) = 2.25
      KV (INT4 → kv_cache_bits_default=8, so 1 byte per element):
        2 * 80 * 8 * 128 * 4096 * 1 * 1 / 1e9 = 0.67
      total = 17.92, available 48 → fits with headroom 30.08
    """
    result = compute_fit(
        model=_qwen3_coder_30b(),
        gpu=_l40s(),
        quant=_int4(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    assert result.fits is True
    assert result.weight_gb == pytest.approx(15.0, rel=1e-9)
    assert result.framework_overhead_gb == pytest.approx(2.25, rel=1e-3)
    assert result.headroom_gb > 25  # plenty of room
    assert result.blocking_reasons == []


# ---------------------------------------------------------------- Slice 3


def test_qwen3_coder_30b_fp8_l40s_fits_with_tight_headroom() -> None:
    """Spec slice 3. FP8 doubles weight footprint vs INT4 but still
    fits on a single L40S.

      weight_gb = 30 * 8/8 = 30
      overhead = max(2, 0.15 * 30) = 4.5
      KV (FP8, 1 byte): same 0.67
      total = 35.17, available 48 → fits, headroom 12.83
    """
    result = compute_fit(
        model=_qwen3_coder_30b(),
        gpu=_l40s(),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    assert result.fits is True
    assert result.weight_gb == pytest.approx(30.0, rel=1e-9)
    assert result.headroom_gb == pytest.approx(12.83, rel=0.01)


# ---------------------------------------------------------------- Slice 4


def test_qwen3_coder_30b_fp8_l40s_long_ctx_high_batch_does_not_fit() -> None:
    """Spec slice 4: weights+overhead fit but KV explodes at
    ctx=128k batch=8, pushing the total past available. The
    blocking reason MUST name 'KV cache' so the LLM caller can
    tell the user 'try a shorter context' vs 'try a smaller model'.

      weight_gb = 30, overhead = 4.5
      KV (FP8): 2 * 80 * 8 * 128 * 131072 * 8 * 1 / 1e9 = 171.8
      total = 206.3, available 48 → fits=False, blocking on KV
    """
    result = compute_fit(
        model=_qwen3_coder_30b(),
        gpu=_l40s(),
        quant=_fp8(),
        tp_size=1,
        batch_size=8,
        context_length=131072,
    )

    assert result.fits is False
    # The blocking reason must specifically name KV cache as the
    # exceeded dimension — weights+overhead = 34.5 GB fits in 48 GB,
    # so the cause is KV explosion, not weights. The LLM caller uses
    # this distinction to suggest 'try a shorter context' rather
    # than 'try a smaller model'.
    assert any("kv cache" in r.lower() for r in result.blocking_reasons)
    # And NOT a 'weights exceed VRAM' diagnosis (which is the
    # specific phrase the weight-exceeded branch emits).
    assert not any("weights exceed vram" in r.lower() for r in result.blocking_reasons)


# ---------------------------------------------------------------- Slice 5


def test_tp_size_multiplies_available_vram() -> None:
    """Spec slice 5 narrative is 'Llama-3.3-70B FP8 H100 SXM ctx=32k
    batch=1 → fits'. The spec's own formulas don't produce that
    outcome on a SINGLE H100 (70.6 + 10.59 overhead + 5.37 KV =
    86.6 GB, available 80 GB → doesn't fit). The narrative
    implicitly assumes tp_size>1 OR a different overhead formula.

    What this test pins instead is the underlying contract: tp_size
    multiplies available VRAM linearly (spec § Formulas:
    `available_gb = gpu.vram_gb x tp_size`). With tp_size=2,
    available becomes 160 GB and Llama-3.3-70B FP8 at ctx=32k
    fits with ~73 GB headroom. The narrative discrepancy is filed
    as a spec note for the M06 close-out PR — formulas remain
    authoritative."""
    result = compute_fit(
        model=_llama_3_3_70b(),
        gpu=_h100_sxm(),
        quant=_fp8(),
        tp_size=2,
        batch_size=1,
        context_length=32768,
    )

    assert result.fits is True
    assert result.available_gb == 160  # 80 GB * tp_size=2
    assert result.headroom_gb > 0
    assert result.assumptions["tp_size"] == 2


def test_single_h100_does_not_fit_llama_70b_fp8_at_32k_per_formula() -> None:
    """Companion to the slice 5 pin: explicitly assert the formula
    output on single H100 so the spec-narrative mismatch is
    documented as a regression test, not just a comment.

      weight 70.6 + overhead 10.59 + KV (FP8, ctx=32k, batch=1) 5.37
      = 86.56 GB > 80 GB available → fits=False
    """
    result = compute_fit(
        model=_llama_3_3_70b(),
        gpu=_h100_sxm(),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=32768,
    )
    assert result.fits is False, (
        "spec slice 5 narrative says this fits, but the spec's own "
        "formulas produce 86.6 > 80. Formula is authoritative; the "
        "narrative is a spec erratum to be filed at close-out."
    )


# ---------------------------------------------------------------- Slice 6


def test_deepseek_v3_mla_kv_cache_at_least_10x_smaller_than_equivalent_gqa() -> None:
    """Spec slice 6 + acceptance criterion 4: MLA collapses K/V into
    a low-rank latent. For DeepSeek-V3 (kv_lora_rank=512,
    qk_rope_head_dim=64, n_layers=61, n_kv_heads=128, head_dim=128),
    the MLA cache must be at least 10x smaller than the equivalent
    standard_gqa cache for the same model dimensions.

    MLA: 61 * (512 + 64) * ctx * batch * 1 = 35136 * ctx * batch
    GQA: 2 * 61 * 128 * 128 * ctx * batch * 1 = 1998848 * ctx * batch
    Ratio: 1998848 / 35136 ≈ 56.9x smaller. Comfortably ≥10.
    """
    mla_model = _deepseek_v3()
    # Same model with kv_cache_strategy="standard_gqa" — pure
    # counterfactual to compute the ratio.
    gqa_model = _model(
        slug="deepseek-v3-as-gqa",
        total_params_b=671.0,
        n_layers=61,
        n_kv_heads=128,
        kv_cache_strategy="standard_gqa",
    )

    mla = compute_fit(
        model=mla_model,
        gpu=_gpu("h100x8", 640),  # 8 * 80
        quant=_fp8(),
        tp_size=1,  # gpu fixture already has aggregated vram
        batch_size=1,
        context_length=32768,
    )
    gqa = compute_fit(
        model=gqa_model,
        gpu=_gpu("h100x8", 640),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=32768,
    )

    ratio = gqa.kv_cache_gb / mla.kv_cache_gb
    assert ratio >= 10, (
        f"MLA kv_cache_gb {mla.kv_cache_gb:.3f} must be ≥10x smaller "
        f"than GQA equivalent {gqa.kv_cache_gb:.3f}; got ratio={ratio:.2f}"
    )
    assert mla.assumptions["kv_cache_strategy"] == "mla"


# ---------------------------------------------------------------- Slice 7


def test_moe_memory_uses_total_params_not_active() -> None:
    """Spec slice 7 + common pitfall: Mixtral 8x22B has
    total_params_b=141 and active_params_b=39. ALL experts must be
    loaded into VRAM; only compute uses active. fit_check must
    compute weight_gb against total_params_b, not active.

      With total_params_b=141 at FP8: weight_gb = 141
      If wrongly using active_params_b=39: weight_gb = 39
    """
    result = compute_fit(
        model=_mixtral_8x22b(),
        gpu=_gpu("h100x4", 320),  # 4 * 80
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    assert result.weight_gb == pytest.approx(141.0, rel=1e-9), (
        f"MoE memory math MUST use total_params_b (141 GB at FP8), "
        f"NOT active_params_b (would be 39). Got {result.weight_gb}."
    )
    # And it fits comfortably on 4xH100 (320 GB).
    assert result.fits is True


# ---------------------------------------------------------------- Slice 8


def test_framework_overhead_floor_2gb_kicks_in_for_small_models() -> None:
    """Spec slice 8 narrative: "7B FP16 returns
    framework_overhead_gb=2.0, not 0.15 * 14 = 2.1 (floor kicks
    in)". The spec narrative is internally inconsistent — its own
    formula `max(2.0, 0.15 * weight_gb)` evaluates to
    `max(2.0, 2.1) = 2.1` for 7B FP16, so the floor does NOT kick
    in there. The narrative example would only hold if the
    formula were `min(2.0, ...)`.

    This is the same class of spec-narrative-vs-spec-formula
    mismatch as slice 5 (Llama-3.3-70B FP8 single H100 → "fits"
    despite the formulas saying 86.6 > 80). Formula is
    authoritative; both narrative discrepancies get filed as spec
    errata at close-out.

    What this test pins instead is the underlying behavior: the
    floor IS at 2 GB, and for any model where 15% < 2 GB it
    engages. 3B FP8 → weights = 3 GB → 15% = 0.45 GB → floor
    wins. Companion test `_15pct_when_above_floor` pins the
    other branch with 7B FP16.
    """
    small_model = _model(slug="tiny-3b", total_params_b=3.0)
    result = compute_fit(
        model=small_model,
        gpu=_l40s(),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    # Floor wins (2.0 > 0.15 * 3 = 0.45).
    assert result.framework_overhead_gb == 2.0
    assert result.assumptions["overhead_floor_gb"] == 2.0


def test_framework_overhead_15pct_when_above_floor() -> None:
    """Companion: above ~13.3 GB weights, 15% beats the floor.
    7B FP16 = 14 GB weights → 0.15 * 14 = 2.1 > 2.0 → 15% wins."""
    seven_b = _model(slug="seven-b", total_params_b=7.0)
    result = compute_fit(
        model=seven_b,
        gpu=_l40s(),
        quant=_fp16(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )

    assert result.weight_gb == pytest.approx(14.0, rel=1e-9)
    assert result.framework_overhead_gb == pytest.approx(2.1, rel=1e-3)
    assert result.assumptions["overhead_pct"] == 0.15


# ============================================================ Property tests


def _baseline_args() -> dict[str, Any]:
    """A fits=True baseline (30B INT4 on L40S)."""
    return dict(
        model=_qwen3_coder_30b(),
        gpu=_l40s(),
        quant=_int4(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )


def _nofit_args() -> dict[str, Any]:
    """A fits=False baseline (Llama-3.3-70B FP8 on single L40S).

    Exists so the algebraic-identity property tests exercise the
    fits=False branch too — that path has its own `headroom_gb`
    sign (negative) and populated `blocking_reasons` that a
    fits=True-only sweep would miss."""
    return dict(
        model=_llama_3_3_70b(),
        gpu=_l40s(),
        quant=_fp8(),
        tp_size=1,
        batch_size=1,
        context_length=4096,
    )


@pytest.mark.parametrize("args_fn", [_baseline_args, _nofit_args])
def test_property_total_required_equals_sum_of_parts(
    args_fn: Any,
) -> None:
    """Spec § Property tests: `total_required_gb == weight_gb +
    kv_cache_gb + framework_overhead_gb` exactly (numerical
    identity, not approx). Sweeps both fits=True and fits=False so
    a future refactor that off-by-ones `headroom_gb` on the
    no-fit branch can't slip past."""
    result = compute_fit(**args_fn())
    assert result.total_required_gb == (
        result.weight_gb + result.kv_cache_gb + result.framework_overhead_gb
    )


@pytest.mark.parametrize("args_fn", [_baseline_args, _nofit_args])
def test_property_fits_equals_total_required_le_available(
    args_fn: Any,
) -> None:
    """Spec § Property tests: `fits == (total_required_gb <=
    available_gb)`. Both branches checked."""
    result = compute_fit(**args_fn())
    assert result.fits == (result.total_required_gb <= result.available_gb)


@pytest.mark.parametrize("args_fn", [_baseline_args, _nofit_args])
def test_property_headroom_equals_available_minus_required(args_fn: Any) -> None:
    """Algebraic identity for `headroom_gb`: must be exactly
    `available_gb - total_required_gb`, NEGATIVE on the no-fit
    branch. Catches a future refactor that mistakenly takes
    `max(0, ...)` and loses the magnitude of the shortfall."""
    result = compute_fit(**args_fn())
    assert result.headroom_gb == result.available_gb - result.total_required_gb


def test_property_sufficiency_caveat_nonempty_on_every_result() -> None:
    """Spec acceptance criterion 4: caveat populated REGARDLESS of
    `fits` value. Pin across a few fits/no-fit combinations."""
    for tp in (1, 2, 8):
        for batch in (1, 4):
            for ctx in (2048, 32768):
                args = _baseline_args() | {
                    "tp_size": tp,
                    "batch_size": batch,
                    "context_length": ctx,
                }
                result = compute_fit(**args)
                assert result.sufficiency_caveat, f"empty caveat at tp={tp} batch={batch} ctx={ctx}"
                assert "necessary" in result.sufficiency_caveat


def test_property_assumptions_echoes_required_constants() -> None:
    """Spec acceptance criterion 3: `assumptions` echoes the
    documented constants so the LLM can disclose them."""
    result = compute_fit(**_baseline_args())
    for key in ("kv_bytes", "overhead_pct", "overhead_floor_gb", "tp_size", "kv_cache_strategy"):
        assert key in result.assumptions, f"missing assumption: {key}"


# ============================================================ Degenerate inputs


@pytest.mark.parametrize("bad_value", [0, -1, -10])
def test_batch_size_zero_or_negative_rejected(bad_value: int) -> None:
    args = _baseline_args() | {"batch_size": bad_value}
    with pytest.raises(ValueError, match="positive"):
        compute_fit(**args)


@pytest.mark.parametrize("bad_value", [0, -1, -10])
def test_context_length_zero_or_negative_rejected(bad_value: int) -> None:
    args = _baseline_args() | {"context_length": bad_value}
    with pytest.raises(ValueError, match="positive"):
        compute_fit(**args)


@pytest.mark.parametrize("bad_value", [0, -1])
def test_tp_size_zero_or_negative_rejected(bad_value: int) -> None:
    args = _baseline_args() | {"tp_size": bad_value}
    with pytest.raises(ValueError, match="positive"):
        compute_fit(**args)


def test_model_with_none_total_params_rejected() -> None:
    """ADR-010: null total_params should be routed to
    requires_measurement by M07 BEFORE reaching fit_check. If a
    null slips through, raise loudly — don't fabricate a weight."""
    null_model = _model(slug="unmeasured", total_params_b=None)  # type: ignore[arg-type]
    args = _baseline_args() | {"model": null_model}
    with pytest.raises(ValueError, match="total_params_b"):
        compute_fit(**args)


# ============================================================ Purity test


def test_purity_compute_fit_does_not_touch_filesystem_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec acceptance criterion 2: compute_fit is pure (no I/O, no
    DB, no globals). Monkeypatch `os.environ`, `open`, AND
    `builtins.__import__` (the entry point for any dynamic import
    inside the function body); assert the function still returns
    the same value byte-for-byte.

    If a future refactor adds an `os.environ.get(
    "FRAMEWORK_OVERHEAD_OVERRIDE")` read, a config-file lookup, or
    a lazy `import some_optional_dep` inside compute_fit, this test
    goes red. Spec § Acceptance criterion 2 names environ + file
    I/O + sys.modules explicitly; the `__import__` trap is the
    load-bearing hook for the third — but it has to be scoped
    tightly around the call site, because pytest's own teardown
    machinery lazily imports things and would trip a globally
    installed trap.
    """
    import builtins
    import os

    args = _baseline_args()
    expected = compute_fit(**args)

    monkeypatch.setattr(os, "environ", {})  # wipe env; pure fn ignores it

    real_open = builtins.open
    real_import = builtins.__import__

    def _no_open(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("compute_fit must not open files")

    import_attempts: list[str] = []

    def _trap_import(name: str, *a: Any, **k: Any) -> Any:
        import_attempts.append(name)
        return real_import(name, *a, **k)

    # Scope the trap manually — install, call, restore — so pytest's
    # post-test fixture-teardown lazy imports don't trigger it.
    builtins.open = _no_open  # type: ignore[assignment]
    builtins.__import__ = _trap_import  # type: ignore[assignment]
    try:
        actual = compute_fit(**args)
    finally:
        builtins.open = real_open  # type: ignore[assignment]
        builtins.__import__ = real_import  # type: ignore[assignment]

    # Same bytes-for-bytes output despite I/O being booby-trapped.
    assert actual.model_dump() == expected.model_dump()
    # And no runtime imports happened inside compute_fit.
    assert import_attempts == [], f"compute_fit performed runtime imports: {import_attempts!r}"
