"""Pure-math VRAM fit verdict — `compute_fit` and `FitResult`.

NO I/O, NO globals, NO DB. Pydantic-typed inputs and output; every
formula sourced from Inference Engineering §3.2 and pinned in
spec/M06 § Formulas. Callers (M08 cost-cells join, M09 fit_check
MCP tool, M09 find_cheapest_deployment) supply already-loaded
catalog objects; this module never reaches for the network or
filesystem.

Per spec § Goal — the function returns a structured `FitResult`
with `weight_gb`, `kv_cache_gb`, `framework_overhead_gb`,
`headroom_gb`, `blocking_reasons`, and a MANDATORY
`sufficiency_caveat`. Never a bare bool: `fits=True` is necessary
but NOT sufficient (kernel coverage, latency, communication
overhead, driver compatibility are all out of scope for this
math), and the caveat exists to disclose that to the LLM caller
verbatim per the trust contract in spec/SHARED.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.pricing.projections import GpuCatalogRow

# Framework-overhead constants pinned per spec § Formulas; calibrated
# against real measurements per the spec preamble. Exposed via
# `FitResult.assumptions` on every result so the LLM caller can
# disclose them rather than treating them as opaque magic numbers.
_OVERHEAD_PCT = 0.15
_OVERHEAD_FLOOR_GB = 2.0


# Spec § Goal: `sufficiency_caveat` is MANDATORY on every result —
# fits=True is necessary, not sufficient. Echoed verbatim into
# `trust_envelope.caveats` by M08/M09 callers; defining it as a
# module constant keeps the wording in one place so we never drift
# out of sync with the spec's exact prose.
_SUFFICIENCY_CAVEAT = (
    "Fit check estimates VRAM sufficiency only. It does not guarantee "
    "acceptable latency, kernel support for the chosen quantization, "
    "tensor-parallel communication efficiency, or provider runtime "
    "compatibility (driver, CUDA, framework version). fits=True is "
    "necessary but not sufficient for a usable rental."
)


class FitResult(BaseModel):
    """Output of `compute_fit`. Every numerical field plus the
    mandatory `sufficiency_caveat` makes up the trust-contract
    surface this module owns.

    Uses `extra="forbid"` (not `extra="ignore"`). CLAUDE.md
    invariant 2 mandates `extra="ignore"` for UPSTREAM projections
    (`Model`, `GpuCatalogRow`, `AaModelRow` — schemas we don't
    control and that evolve). FitResult is an OWNED output type
    we construct ourselves, in the same category as our seed
    schemas (`Quantization`, `WorkloadProfile`, `TrackedModelRow`,
    `GpuSupplement`) which all use `extra="forbid"` so a typo in
    construction fails loudly instead of silently dropping a
    field. Forbid here is the conservative choice for the trust-
    contract surface — M08/M09 callers serialize this into trust
    envelopes, and a quietly-dropped field would be a real bug.
    """

    model_config = ConfigDict(extra="forbid")

    fits: bool
    weight_gb: float
    kv_cache_gb: float
    framework_overhead_gb: float
    total_required_gb: float
    available_gb: float
    headroom_gb: float
    blocking_reasons: list[str] = Field(default_factory=list)
    # Constants echoed back so the LLM can disclose them. Spec
    # acceptance criterion 3: "assumptions echoes exact constants
    # used (kv_bytes, overhead_pct=0.15, overhead_floor_gb=2.0,
    # tp_size, kv_cache_strategy)".
    assumptions: dict[str, Any] = Field(default_factory=dict)
    sufficiency_caveat: str = _SUFFICIENCY_CAVEAT


def compute_fit(
    *,
    model: Model,
    gpu: GpuCatalogRow,
    quant: Quantization,
    tp_size: int,
    batch_size: int,
    context_length: int,
) -> FitResult:
    """Pure VRAM-fit verdict per spec/M06 § Formulas.

    Returns a `FitResult` for every input-domain combination. Only
    raises `ValueError` on degenerate inputs (`batch_size <= 0` or
    `context_length <= 0`) — those are mathematical undefined-domain
    cases, not 'doesn't fit' cases. `tp_size <= 0` is the same
    category. `total_params_b is None` is also a degenerate case —
    M07's ADR-010 routes null-params models to `requires_measurement`
    upstream, but if we reach here without it, raise rather than
    fabricate.
    """
    if batch_size <= 0 or context_length <= 0 or tp_size <= 0:
        raise ValueError(
            f"batch_size, context_length, tp_size must all be positive; got "
            f"batch_size={batch_size}, context_length={context_length}, "
            f"tp_size={tp_size}"
        )
    if model.total_params_b is None:
        raise ValueError(
            f"model {model.slug!r} has total_params_b=None; cannot compute "
            "VRAM weight footprint. Upstream callers (M07) should route "
            "null-params models to requires_measurement before reaching here."
        )

    weight_gb = model.total_params_b * quant.bits_per_weight / 8.0

    framework_overhead_gb = max(_OVERHEAD_FLOOR_GB, _OVERHEAD_PCT * weight_gb)

    kv_cache_gb = _kv_cache_gb(
        model=model,
        quant=quant,
        batch_size=batch_size,
        context_length=context_length,
    )

    total_required_gb = weight_gb + kv_cache_gb + framework_overhead_gb
    available_gb = float(gpu.vram_gb * tp_size)
    fits = total_required_gb <= available_gb
    headroom_gb = available_gb - total_required_gb

    blocking_reasons: list[str] = []
    if not fits:
        # Distinguish weights-too-big from KV-too-big so the LLM
        # caller can explain to the user which dimension to relax.
        if weight_gb + framework_overhead_gb > available_gb:
            blocking_reasons.append(
                f"weights exceed VRAM: weights + overhead = "
                f"{weight_gb + framework_overhead_gb:.1f} GB > "
                f"{available_gb:.1f} GB available"
            )
        else:
            # Weights+overhead fit; the KV cache pushes it over.
            blocking_reasons.append(
                f"KV cache exceeds headroom: weights + overhead = "
                f"{weight_gb + framework_overhead_gb:.1f} GB, plus KV cache "
                f"{kv_cache_gb:.1f} GB at ctx={context_length} batch={batch_size} "
                f"= {total_required_gb:.1f} GB > {available_gb:.1f} GB available"
            )

    return FitResult(
        fits=fits,
        weight_gb=weight_gb,
        kv_cache_gb=kv_cache_gb,
        framework_overhead_gb=framework_overhead_gb,
        total_required_gb=total_required_gb,
        available_gb=available_gb,
        headroom_gb=headroom_gb,
        blocking_reasons=blocking_reasons,
        assumptions={
            "kv_bytes": quant.kv_cache_bits_default / 8.0,
            "overhead_pct": _OVERHEAD_PCT,
            "overhead_floor_gb": _OVERHEAD_FLOOR_GB,
            "tp_size": tp_size,
            "kv_cache_strategy": model.kv_cache_strategy,
        },
    )


def _kv_cache_gb(
    *,
    model: Model,
    quant: Quantization,
    batch_size: int,
    context_length: int,
) -> float:
    """KV cache size per spec § Formulas. Branches on
    `model.kv_cache_strategy` — standard_gqa (Llama/Qwen/Mistral/
    Phi/Gemma), mla (DeepSeek-V3), or sliding_window (some Mistral
    variants).
    """
    kv_bytes = quant.kv_cache_bits_default / 8.0

    if model.kv_cache_strategy == "mla":
        # DeepSeek MLA: collapses K and V into a single low-rank
        # latent + a small rope projection. Dramatically smaller
        # than the equivalent standard_gqa for the same n_kv_heads
        # (the M03 fixture has n_kv_heads=128, but only the
        # latent + rope dims are actually cached). Validate the
        # required raw_config keys upfront — direct `raw_config[
        # "kv_lora_rank"]` would raise bare KeyError, which
        # `sync_all_tracked`-style batch consumers and M07/M09's
        # MCP layer don't catch; converting to ValueError naming
        # the slug + missing fields gives MCP callers a useful
        # diagnostic instead of an opaque traceback.
        missing = [k for k in ("kv_lora_rank", "qk_rope_head_dim") if k not in model.raw_config]
        if missing:
            raise ValueError(
                f"model {model.slug!r} has kv_cache_strategy='mla' but "
                f"raw_config is missing required key(s) {missing!r}. "
                "MLA models need both kv_lora_rank and qk_rope_head_dim "
                "from HF config.json — confirm the M03 sync captured them."
            )
        kv_lora_rank = int(model.raw_config["kv_lora_rank"])
        qk_rope_head_dim = int(model.raw_config["qk_rope_head_dim"])
        return (
            model.n_layers
            * (kv_lora_rank + qk_rope_head_dim)
            * context_length
            * batch_size
            * kv_bytes
        ) / 1e9

    if model.kv_cache_strategy == "sliding_window":
        # `KvCacheStrategy` advertises this value but `Model`
        # doesn't carry a typed `sliding_window_size` field yet
        # (it lives in `raw_config["sliding_window"]` for the
        # Mistral variants that use it). Spec § Formulas wants
        # `effective_ctx = min(ctx, model.sliding_window_size)`
        # clamping; without the field plumbed, fall-through to
        # standard_gqa would silently over-estimate KV cache and
        # produce wrong `fits` verdicts. Raise loudly until M07
        # (or a follow-up) plumbs the missing field, so the
        # failure mode is "unsupported model" rather than
        # "wrong-answer budget_to_plan".
        raise NotImplementedError(
            f"model {model.slug!r} has kv_cache_strategy='sliding_window' "
            "but compute_fit doesn't yet plumb `sliding_window_size` from "
            "HF config.json into the Model projection. Tracked for follow-up; "
            "until then this combination is unsupported."
        )

    # standard_gqa: 2 * layers * kv_heads * head_dim * ctx * batch * kv_bytes
    # The leading 2 is K + V tensors.
    return (
        2
        * model.n_layers
        * model.n_kv_heads
        * model.head_dim
        * context_length
        * batch_size
        * kv_bytes
    ) / 1e9
