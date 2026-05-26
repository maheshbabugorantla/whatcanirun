# M06 — fit_check Pure Function

**Status:** ⬜ Not started
**Effort:** 4h (6h realistic with MLA + MoE branches)
**Dependencies:** M01, M03
**Unblocks:** M08, M09 (fit_check is its own MCP tool + used by find_cheapest_deployment)

> Read [`SHARED.md`](SHARED.md) first. Math anchored in Inference Engineering book §3.2.

---

## Goal

Pure function: given a model, GPU, quantization, tensor-parallel size, batch, and context — return a `FitResult` with weight bytes, KV cache bytes, framework overhead, headroom, and a `sufficiency_caveat`. Never just a bool. The sufficiency caveat is mandatory on every result (`fits=True` is necessary, NOT sufficient).

No I/O. No DB. No globals. Property-test verified.

---

## Scope

### Public surface (`src/whatcanirun/inference/fit_check.py`)

```python
def compute_fit(
    model: Model,
    gpu: Gpu,
    quant: Quantization,
    tp_size: int,
    batch_size: int,
    context_length: int,
) -> FitResult:
    """Pure VRAM-fit verdict. Returns FitResult; never raises for input-domain values."""


class FitResult(BaseModel):
    fits: bool
    weight_gb: float
    kv_cache_gb: float
    framework_overhead_gb: float
    total_required_gb: float
    available_gb: float
    headroom_gb: float
    blocking_reasons: list[str]
    assumptions: dict[str, Any]   # kv_bytes, overhead_pct, tp_size, kv_cache_strategy

    # MANDATORY. Per critique-round-4 edit: fits=True is necessary, not sufficient.
    # Echoed into trust_envelope.caveats by callers.
    sufficiency_caveat: str = (
        "Fit check estimates VRAM sufficiency only. It does not guarantee acceptable "
        "latency, kernel support for the chosen quantization, tensor-parallel "
        "communication efficiency, or provider runtime compatibility (driver, CUDA, "
        "framework version). fits=True is necessary but not sufficient for a usable rental."
    )
```

### Formulas

**Weights (all families):**
```
weight_gb = total_params_b × bits_per_weight / 8
```

**KV cache — standard GQA (Llama, Qwen, Mistral, Phi, Gemma):**
```
kv_bytes = quant.kv_cache_bits_default / 8
kv_cache_gb = (2 × n_layers × n_kv_heads × head_dim × ctx × batch × kv_bytes) / 1e9
```
The `2` is for K and V tensors.

**KV cache — MLA (DeepSeek-V3 family):**
```
kv_cache_gb_mla = (n_layers × (kv_lora_rank + qk_rope_head_dim) × ctx × batch × kv_bytes) / 1e9
```
MLA collapses K and V into a single low-rank latent. KV cache is dramatically smaller than equivalent GQA.

**KV cache — sliding window (some Mistral variants):**
```
effective_ctx = min(ctx, model.sliding_window_size)
kv_cache_gb = (2 × n_layers × n_kv_heads × head_dim × effective_ctx × batch × kv_bytes) / 1e9
```

**Framework overhead — 15% of weights, floored at 2 GB:**
```
framework_overhead_gb = max(2.0, 0.15 × weight_gb)
```
CUDA context + activation buffers + small communication buffers. Calibrated against real measurements.

**Total + verdict:**
```
total_required_gb = weight_gb + kv_cache_gb + framework_overhead_gb
available_gb = gpu.vram_gb × tp_size
fits = total_required_gb <= available_gb
headroom_gb = available_gb - total_required_gb
```

**MoE handling:**
- For memory: use `total_params_b` (all experts must be in VRAM)
- For compute (M07's concern): use `active_params_b`

---

## Vertical slices (8 TDD cycles)

Each cycle: write ONE failing test, write ONE impl change, get green, commit.

1. **Llama-3.3-70B FP8 single L40S ctx=4k batch=1** → does NOT fit (`blocking_reasons` contains `"weights exceed VRAM (35 GB > 48 GB available with overhead)"`)
2. **Qwen-3-Coder-30B INT4 single L40S ctx=4k batch=1** → fits, positive headroom
3. **Qwen-3-Coder-30B FP8 single L40S ctx=4k batch=1** → fits with tight headroom (~5GB)
4. **Qwen-3-Coder-30B FP8 single L40S ctx=128k batch=8** → does NOT fit (`blocking_reasons` contains `"KV cache exceeds headroom"`)
5. **Llama-3.3-70B FP8 H100 SXM ctx=32k batch=1** → fits
6. **DeepSeek-V3 FP8 8×H100 ctx=32k batch=1** → fits with MLA branch verified (KV << standard GQA equivalent — assert `kv_cache_gb` is ≥10× smaller than the same model would compute under standard_gqa)
7. **Mixtral 8x22B FP8 4×H100** → fits using `total_params_b` (memory), not `active_params_b`
8. **Framework overhead floor:** 7B FP16 returns `framework_overhead_gb=2.0`, not `0.15 × 14 = 2.1` (floor kicks in)

---

## Property tests

- For all inputs: `total_required_gb == weight_gb + kv_cache_gb + framework_overhead_gb` (numerical identity, not approx)
- For all inputs: `fits == (total_required_gb <= available_gb)`
- For all inputs: `FitResult` is a Pydantic instance with all fields populated; `sufficiency_caveat` is non-empty
- For batch_size or context_length = 0: ValueError raised with clear message (degenerate case)

---

## Acceptance criteria

- [ ] All 8 TDD cycles green; written red-then-green-then-refactor, not horizontally
- [ ] `compute_fit` is pure (no I/O, no DB, no globals) — enforced by property test that monkeypatches `os.environ`, `sys.modules`, file I/O and confirms output unchanged
- [ ] `assumptions` echoes exact constants used (kv_bytes, overhead_pct=0.15, overhead_floor_gb=2.0, tp_size, kv_cache_strategy) so the LLM can disclose them
- [ ] `sufficiency_caveat` populated on EVERY FitResult, regardless of `fits` value
- [ ] MLA branch produces KV cache ≥10× smaller than standard_gqa for DeepSeek-V3
- [ ] `uv run pytest tests/inference/test_fit_check.py -ra` green

---

## Common pitfalls

- **MoE memory math.** All experts must be loaded. Use `total_params_b`, not `active_params_b`. The latter is only for compute math in M07.
- **Sliding window misapplication.** Apply only when `model.kv_cache_strategy == "sliding_window"`. Don't blanket-apply to Mistral — Mistral-Large doesn't slide.
- **Tensor-parallel rounding.** `tp_size=2` does NOT mean exactly half the weights per GPU — communication overhead exists. v1 ignores this; `gpu.vram_gb × tp_size` is upper bound.
- **Pedantic batch=0 cases.** Reject explicitly, don't silently divide-by-something-undefined.

---

## When done

Commit:
> `M06: fit_check pure function with MLA + MoE + sliding-window branches`

Mark M06 ✓ in `INDEX.md`. Continue with M07.
