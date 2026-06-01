# M10 — Benchmark Seeds (Public Sources)

**Status:** ✓ Partially shipped — verification tooling landed (PR #17, `33ce718`); Tier 1b cell curation **deferred to v2 M17** (PR #19, `<this-pr>`).
**Effort:** 6h estimated, ~7h actual on the verification tooling alone.
**Dependencies:** M00
**Unblocks (revised):** v2 M17 GuideLLM-measured cells will reuse PR #17's V1 sanity-check tool + V2 merge tool + `seeds/benchmark_cells.sources/` archive convention.
**Parallel-safe:** yes

> Read [`SHARED.md`](SHARED.md) first. ADR-006 is load-bearing.

---

## Deferral preamble (2026-05-31)

The Tier 1b cell-curation work this spec describes proved infeasible for v1.
Three independent reasons emerged during PR-β scoping:

1. **Public benchmark sources don't publish the data shape `BenchmarkCell` expects.**
   The schema below wants steady-state per-stream decode-TPS at a specific
   `(gpu, model, quant, tp, batch, ctx)` op-point. Public benchmark blogs
   instead publish aggregate-throughput-at-concurrency numbers (e.g.
   "2,400 tok/s across 100 concurrent requests on H100 SXM5") and
   commonly conflate prefill into the per-token figure. Extracting our
   shape requires per-stream isolation work the source authors didn't do.

2. **Source URLs rot faster than the spec assumed.** The spec's named
   Spheron H100/H200 article 404s as of 2026-05-30. The replacement
   article from the same publisher reports single-stream numbers
   (120 tok/s for Llama-3.3-70B FP8 on H100 SXM5) that are ~3.4×
   above the bandwidth-physics ceiling — likely prefill-amortized
   measurements, but unauditable as steady-state decode.

3. **Even paid first-principles sources don't anchor cells.** The
   *Inference Engineering* book by Kiely (2026), explicitly named in
   the spec source list, is a textbook that **teaches** the
   bandwidth-heuristic methodology we already implement (`KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75` from §2.4.2) and publishes GPU spec
   tables we already cross-check against ComputePrices. It does not
   publish per-cell TPS measurements.

What landed regardless:

- **`seeds/gpu_catalog_snapshot.yaml`** — 66 GPU rows projected from
  ComputePrices, cross-validated against Kiely 2026 §3.2 tables.
  Useful for V1 sanity-check determinism + any future cell additions.
- **`scripts/m10/sanity_check_cells.py` (V1) + `merge_candidate_to_parquet.py` (V2)**
  — 11 sanity-check functions + atomic merge tool. Reusable for v2's
  `own_measured` cells.
- **`seeds/benchmark_cells.sources/`** archive convention + README.
- **`@pytest.mark.network`** URL accessibility test (Slice C).
- **`BenchmarkCell` schema** unchanged. **`Source` Literal**
  retains `public_benchmark_anchor` — v2 may revive Tier 1b.

What v1 ships with instead:

- Tier 2 (AA `provider_anchor`, confidence 0.7) for models AA tracks.
- Tier 3 (`bandwidth_heuristic`, confidence 0.6) for everything else.
- The trust contract is preserved: confidence is honestly reported
  at the lower tier rather than fabricated at 0.80.

The rest of this spec is **archived as historical context** — its
"20–30 cells from public sources" goal is no longer the active v1
target. v2's M17 will replace this work with GuideLLM-measured cells
that own_measured cells (Tier 1a, confidence 0.95) will populate.

---

## Goal

20–30 hand-curated benchmark cells from public sources, committed to `seeds/benchmark_cells.parquet`. Every row tagged `source="public_benchmark_anchor"` with `source_url`. These bootstrap `tps_estimator` Tier 1b at confidence 0.80.

No GPU rental in v1. No GuideLLM runs. Just careful curation from existing public benchmarks.

---

## Scope

### Schema

```python
class BenchmarkCell(BaseModel):
    model_config = ConfigDict(extra="forbid")  # our own data; strict

    # Op-point identifiers
    gpu_slug: str
    model_slug: str
    quant_slug: str
    tp_size: int
    batch_size: int
    context_length: int

    # Measured numbers
    decode_tps: float
    prefill_tps: float | None
    ttft_ms: float | None

    # Engine details
    engine: Literal["vllm", "sglang", "tensorrt_llm", "tgi", "other"]
    engine_version: str
    measured_at: date

    # Provenance — v1 ALWAYS public_benchmark_anchor
    source: Literal["public_benchmark_anchor", "own_measured"]   # v1 enforces NEVER own_measured
    source_url: str
    notes: str                                                    # methodology summary in 1–2 sentences
```

### Source list (curated; live-verified)

Targets ~20-30 cells covering the matrix:
**GPUs:** H100 SXM, H200 SXM, L40S, A100 80GB SXM
**Models:** Llama-3.3-70B, Llama-3.1-8B, Qwen-3-Coder-30B, Mistral-Large, DeepSeek-V3
**Quants:** FP16/BF16, FP8, INT4
**Op-points:** batch=1 (always), plus a few batch=8 and batch=32 where the source is rigorous

Verified sources:

1. **Inference Engineering book (Kiely 2026)** — §2.4 single-stream examples, §3.2 H100 quoted figures. Anchor for ~5 cells.
2. **vLLM official benchmarks** — release notes and blog posts. Anchor for ~5 cells.
3. **SGLang release benchmarks** — same. Anchor for ~3 cells.
4. **Spheron H100/H200 article (2026)** — comprehensive Llama-3.3-70B FP8 numbers across batches. Anchor for ~5 cells.
5. **MorphLLM tokens-per-second article** — concurrency-aware numbers across providers. Anchor for ~5 cells.
6. **Joshua8.ai Blackwell RTX PRO 6000 article** — Llama-3.1-8B INT4/FP4 numbers. Anchor for ~2 cells.

### Example row

```python
BenchmarkCell(
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
    source_url="https://www.spheron.network/...",
    notes="Single H100 SXM, FP8 quantization (e4m3), batch=1, ctx=4096. vLLM 0.6.x with paged_attention.",
)
```

---

## Vertical slices

1. **Slice A: BenchmarkCell schema** — TDD: extra field rejected; required field missing rejected; `source="own_measured"` rejected with explicit "v1 cannot create own_measured cells" error in validator.
2. **Slice B: Parquet write/read round-trip** — TDD: 1 cell written, read back, equals original (pyarrow round-trip).
3. **Slice C: Source-URL accessibility test** — TDD: every committed URL responds 200 (run as a separate `@pytest.mark.network` test, gated to off in CI; manual check before release).
4. **Slice D: Seed compilation** — fill in 20-30 cells from the source list. One PR per source for clean git history.
5. **Slice E: tps_estimator integration** — TDD: a seeded cell's exact op-point query returns the seeded value via Tier 1b, confidence=0.80.

---

## Acceptance criteria

- [ ] `seeds/benchmark_cells.parquet` has ≥20 rows covering the matrix above.
- [ ] Every row has `source="public_benchmark_anchor"` — NO row claims `own_measured` (enforced by Pydantic validator AND a separate test that reads the parquet and asserts).
- [ ] Every row's `source_url` is publicly accessible (manual verification recorded in commit message).
- [ ] `notes` field on every row summarizes methodology in 1–2 sentences.
- [ ] tps_estimator returns Tier 1b values at confidence 0.80 for seeded op-points.
- [ ] Schema validation rejects any future PR attempting to add `source="own_measured"` to v1 seeds.

---

## Common pitfalls

- **Mixing sources without methodology check.** Some "benchmark" blog posts don't specify batch size or engine version. Skip those rows. Better to have 18 good cells than 30 questionable ones.
- **Stale numbers.** A cell from 2024 with vLLM 0.4 is less applicable to 2026 stacks. Prefer recent sources (2025+) and document `engine_version`.
- **Cross-pollinated GPUs.** "H100" in a benchmark might mean H100 SXM or PCIe. They have different bandwidth. Read carefully; if unclear, skip.
- **Tempting "own_measured" for verified anchors.** The whole point of this milestone is honesty. A public source that aligns with your bandwidth heuristic is still a public source, not an own measurement.

---

## When done

Commit:
> `M10: benchmark seeds — N cells from public sources, all tagged public_benchmark_anchor`

Mark M10 ✓ in `INDEX.md`. Continue with M11.
