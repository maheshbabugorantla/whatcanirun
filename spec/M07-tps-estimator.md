# M07 — tps_estimator (5-tier provenance)

**Status:** ⬜ Not started
**Effort:** 4h (6h realistic)
**Dependencies:** M03, M04 (optional), M10 (Tier 1b cells)
**Unblocks:** M08, M09

> Read [`SHARED.md`](SHARED.md) first. ADR-010 (single-stream only) is load-bearing.

---

## Goal

Return a `TpsEstimate` with explicit provenance via 5 tiers. Each tier has a fixed confidence value. Refuse honestly when none apply (`requires_measurement`).

**Critical:** v1 NEVER returns `source="own_measured"`. Only v2 (with GuideLLM) earns that. v1's hand-curated cells are `source="public_benchmark_anchor"` at confidence 0.80.

---

## Scope

### Public surface (`src/whatcanirun/inference/tps_estimator.py`)

```python
def estimate_tps(
    model: Model,
    gpu: Gpu,
    quant: Quantization,
    batch_size: int,
    context_length: int,
    bench_cells: list[BenchmarkCell],
    aa_observations: list[AaModelRow] | None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
) -> TpsEstimate:
    """Returns a single decode-TPS estimate with provenance. Pure function."""


class TpsEstimate(BaseModel):
    value: float | None
    source: Literal[
        "own_measured",                       # v2 only — reproducible GuideLLM run
        "public_benchmark_anchor",            # v1 M10 — hand-curated from blogs/articles
        "provider_anchor",                    # AA median_output_tokens_per_second
        "bandwidth_heuristic_single_stream",  # book §2.4.2
        "requires_measurement",               # explicit refusal
    ]
    confidence: float                   # 0.95 / 0.80 / 0.7 / 0.6 / 0.0
    anchor_detail: str | None
    source_url: str | None              # populated for public_benchmark_anchor
    refusal_reason: str | None          # only when source = requires_measurement
```

### Decision tree (5 tiers, in order)

1. **Tier 1a — `own_measured` (confidence=0.95):** v2 ONLY. Exact `(gpu, model, quant, tp_size, batch≤bucket, ctx≤bucket)` row in `bench_cells` with `source="own_measured"`. **v1 never reaches this branch.**

2. **Tier 1b — `public_benchmark_anchor` (confidence=0.80):** v1's default. Exact match in `bench_cells` with `source="public_benchmark_anchor"`. Real numbers from external sources (Spheron, MorphLLM, vLLM blogs); methodology unverified. `source_url` populated for audit.

3. **Tier 2 — `provider_anchor` (confidence=0.7):** AA has `median_output_tokens_per_second` for this model AND `batch_size==1`. For reasoning models, AA row must match the requested `reasoning_effort`. Caveat:
   > "AA reports a serving aggregate across providers; specific GPU and batch are not modeled."

4. **Tier 3 — `bandwidth_heuristic_single_stream` (confidence=0.6):** `batch_size == 1` AND no measured/AA data. Formula:
   ```python
   weights_bytes_per_token = total_params_b * 1e9 * bits_per_weight / 8
   peak_tps = gpu.memory_bandwidth_gb_s * 1e9 / weights_bytes_per_token
   value = peak_tps * KERNEL_EFFICIENCY_SINGLE_STREAM   # 0.75
   ```
   `KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75` anchored to verified anchors:
   - Llama-3.3-70B FP8 H100 SXM batch=1: heuristic 35.9 tok/s, real ~35 tok/s ✓
   - Llama-3.1-8B BF16 H100 SXM batch=1: heuristic 157 tok/s, real ~100 tok/s (±50%, acceptable for single-stream)

5. **Tier 4 — `requires_measurement` (confidence=0.0):** `batch_size > 1` AND no measured/anchor row. Returns:
   ```python
   TpsEstimate(
       value=None,
       source="requires_measurement",
       confidence=0.0,
       refusal_reason="batched throughput not modeled by heuristic. "
                      "Submit a benchmark cell, switch to batch=1 single-stream "
                      "estimate, or accept that this combination cannot be priced honestly."
   )
   ```

### Tier ordering

When multiple tiers match, lower number wins. When both Tier 1a and Tier 1b match: 1a wins. When both Tier 1b and Tier 2 match: 1b wins (we trust an anchor specific to (gpu, model, quant, batch, ctx) over an aggregate observation).

---

## Vertical slices (8 TDD cycles)

1. **Tier 1 match** — TDD: `(H100, llama-3-3-70b, fp8, batch=1)` with an `own_measured` row → returns measured value, confidence=0.95
2. **Tier 1b match** — same query with only a `public_benchmark_anchor` row → returns anchor value, confidence=0.80, `source_url` populated
3. **Tier 2 fall-through** — same query, no measured/anchor, AA has observation → returns AA value, confidence=0.7
4. **Tier 3 fall-through** — same query, no measured/AA → returns heuristic 35.9 tok/s, confidence=0.6
5. **Tier 4 refusal** — `(H100, llama-3-3-70b, fp8, batch=32)` no anchor → `value=None`, `source="requires_measurement"`
6. **Tier ordering** — both 1a and 1b match: 1a wins. Both 1b and 2 match: 1b wins.
7. **Reasoning effort dimension** — `(H100, gpt-oss-120b, fp8, batch=1, reasoning_effort="high")` AND AA has all three rows → returns `-high` row, not `-low`
8. **Confidence values exact** — 0.95 / 0.80 / 0.7 / 0.6 / 0.0 — enforced (no fudge factors)

---

## Acceptance criteria

- [ ] All 8 TDD cycles green
- [ ] No `TpsEstimate.value` is non-None without a populated `source`
- [ ] Heuristic constant `KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75` is named, with citation comment to anchor verification
- [ ] `source_url` populated on every `public_benchmark_anchor` return
- [ ] v1 NEVER returns `source="own_measured"` — `seeds/benchmark_cells.parquet` validated to contain only `public_benchmark_anchor` rows (this is a fast test that scans the seed file)
- [ ] `estimate_tps` is pure (no I/O); property-tested
- [ ] Reasoning effort dimension respected when AA has the variant rows
- [ ] Carry-forward from M01 retired: backfill `fp8_tflops_dense` (and `fp4_tflops_dense` where applicable) in `seeds/gpus_supplement.yaml` for the 7 GPUs left null at M01 ship (h100nvl, b100, b200, gb200, gb300, l40, mi300x). See [M01 carry-forward](#m01-carry-forward) below.
- [ ] Carry-forward from M04 retired: expand `seeds/tracked_models.yaml` from 3 → 30 rows AND keep `seeds/aa_slug_mapping.yaml` in lockstep (each new `cp_slug` MUST have a matching mapping row, empty `aa_slugs: []` is fine for documented absences). Pick a reasoning model with all three AA variants (`-low` / `-medium` / `-high`) to satisfy the "Reasoning effort dimension respected" criterion against live data, not just unit tests. See [M04 carry-forward](#m04-carry-forward) below.

---

## M01 carry-forward

M01 shipped `seeds/gpus_supplement.yaml` with 7 of 12 rows carrying `fp8_tflops_dense: null` (and `fp4_tflops_dense: null` for the Blackwell rows) because the dense per-GPU value was not directly cited on the vendor page we could read at capture time. M07's bandwidth heuristic does not currently use these fields, so the trust contract holds — Tier 3 only reads `memory_bandwidth_gb_s` from ComputePrices. But:

- Any compute-bound regime check (peak `tps_compute = fp_tflops * 1e12 / (2 * total_params * bits_per_weight / 8)`) added during M07 implementation MUST treat null as "compute bound unknown — fall back to bandwidth-only" rather than crashing or silently substituting a default.
- Before M07 merges, backfill the 7 deferred rows from the canonical vendor whitepapers below. Update `seeds/gpus_supplement.yaml` directly; the round-trip test in `tests/catalog/test_round_trip.py` guards against silent default drift.

| slug | whitepaper / brief to consult |
|---|---|
| `h100nvl` | NVIDIA H100 NVL product brief (94GB / 400W variant) |
| `b100` | NVIDIA Blackwell architecture whitepaper, B100 SKU table |
| `b200` | NVIDIA Blackwell architecture whitepaper, B200 SKU table |
| `gb200` | NVIDIA Grace-Blackwell GB200 whitepaper, per-GPU dense |
| `gb300` | NVIDIA Blackwell Ultra whitepaper (when published) |
| `l40` | NVIDIA L40 datasheet PDF (not the L40S page) |
| `mi300x` | AMD Instinct MI300X datasheet PDF |

---

## M04 carry-forward

M04 shipped `seeds/tracked_models.yaml` with 3 of the target 30 rows (llama-3-3-70b, deepseek-v3, mixtral-8x22b — one per family the HF fixtures cover). `seeds/aa_slug_mapping.yaml` carries the matching 3 rows. The 30-row target was always M07's responsibility: M07's Tier-2 routing needs the populated mapping for every model it ranks, and the unit tests for `resolve_aa_slug` use synthetic data because no real reasoning model in the M04-shipped 3 rows has the full effort-variant set.

Before M07 merges:

1. **Expand `seeds/tracked_models.yaml` to ~30 rows.** Add the remaining models from the families we already track (Llama 3.1 variants, Qwen 2.5 / Qwen 3 variants, Mistral / Mixtral other sizes, Phi, Gemma) plus any that landed in CP since the 2026-05-26 capture. Each new row requires:
   - A captured HF `config.json` fixture under `tests/fixtures/` (offline test policy — adding a row without the fixture breaks `test_first_sync_fetches_from_hf_then_caches`-style coverage).
   - Verified `total_params_b` / `active_params_b` from the model card or safetensors index.

2. **Keep `seeds/aa_slug_mapping.yaml` in lockstep.** Every new `cp_slug` in `tracked_models.yaml` MUST get a corresponding mapping row. Empty `aa_slugs: []` + `investigation_note` is the documented-absence shape (Llama-3.3-70B-style — though the M04 investigation found that one). The existing `test_every_tracked_model_has_an_aa_mapping_row` regression test enforces this.

3. **Pick a reasoning model with all three AA variants for the end-to-end test.** M04's spec example was `gpt-oss-120b` with `-low` / `-medium` / `-high`, but the 2026-05-27 AA capture has only `gpt-oss-120b-low` + base — no `-medium` / `-high` rows. Real candidates that DO have all three on AA today: `gpt-5-5-low/-medium/-high`, `deepseek-v4-pro-non-reasoning/-high`, `nova-2-0-omni-reasoning-low/-medium`. Pick one, add it to `tracked_models.yaml` + `aa_slug_mapping.yaml`, and use it for the "Reasoning effort dimension respected when AA has the variant rows" acceptance test. The synthetic-data unit tests already pin the `resolve_aa_slug` logic; this is the live-data smoke test on top.

4. **Re-capture the AA fixture if it's stale by the time M07 lands.** `tests/fixtures/aa_models_2026-05-27.json` is timestamped; the AA Intelligence Index ships new `evaluations` keys every few releases. If `test_all_525_rows_in_fixture_project_without_error` (M04) still passes against the fresh capture, the schema is stable enough; otherwise the projection may need new optional fields.

The M04 `AA_ATTRIBUTION_STRING` constant (shipped via PR #8) is the canonical source M07's TrustEnvelope construction MUST import for the `sources[].license_attribution` field on any cost cell derived from AA data — do NOT retype the string.

---

## Common pitfalls

- **Don't scale heuristic with batch.** Verified ~6× wrong at batch=128. `batch>1` falls through to Tier 4, period.
- **Don't conflate Tier 1a and Tier 1b.** Same `bench_cells` table; different `source` field. A test that loads M10 seeds and asserts no row has `source="own_measured"` keeps this clean.
- **AA reasoning-effort dimension is in the slug suffix.** Match against the curated mapping from M04, not the model's name.

---

## When done

Commit:
> `M07: tps_estimator 5-tier provenance with bandwidth heuristic + AA Tier 2`

Mark M07 ✓ in `INDEX.md`. Continue with M08.
