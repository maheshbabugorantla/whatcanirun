# ComputePrices data quality notes

Issues we've encountered with the live ComputePrices API data while
building whatcanirun. None block v1 functionality (Tier 3 only reads
`memory_bandwidth_gbps`, which is correct in all sampled rows), but
they're worth knowing if you build on the same data.

Discovered while cross-checking `seeds/gpu_catalog_snapshot.yaml` (66
rows projected from the 2026-05-26 CP fixture) against published
authoritative tables in Kiely 2026, *Inference Engineering* §3.2
"GPU Architecture Generations."

---

## 1. `specs.fp8_tflops` is mislabeled as FP16

CP's `fp8_tflops` field reports a value that is **half** of the
published FP8 dense compute number on every GPU we cross-checked.
Kiely's tables explicitly label these as FP8 dense; CP's value
matches the FP16 dense number instead.

| GPU | CP `fp8_tflops` | Kiely §3.2 FP8 dense | Kiely FP16 dense |
|---|---|---|---|
| H100 SXM | 990 | 1,979 | 989 ¹ |
| H200 | 990 | 1,979 | 989 ¹ |
| L4 | 121 | 242 | 121 |
| B200 | 4,500 | ~5,000 | (not separately tabulated; consistent with the ~2× FP8/FP16 pattern) |

¹ "An H100 GPU in FP16 can perform 989 teraFLOPS of dense
computation against 3.35 TB/s of memory bandwidth" — Kiely 2026
§2.4.1.

**v1 impact:** none. The TPS-estimator Tier 3 bandwidth heuristic
reads `specs.memory_bandwidth_gbps` only, and that value matches
Kiely's tables exactly. No code path reads `fp8_tflops`.

**Future-impact risk:** any downstream consumer that interprets
`fp8_tflops` as FP8 compute will under-predict FP8 throughput by 2×.
Worth flagging to CP upstream and/or aliasing the field in our
projection if we ever rank by compute.

## 2. NVIDIA Blackwell B300 missing from the catalog

Kiely §3.2.3 documents B300 as a 288 GB / up to 8 TB/s Blackwell
sibling of B200. The 2026-05-26 CP fixture has no B300 row. Sampling
the live CP API would confirm whether it has landed since.

**v1 impact:** none. The B300 isn't widely deployed for inference
yet; absence doesn't block any existing workflow. v1's catalog
coverage is what users can actually rent today.

**Future-impact risk:** the moment a CP provider lists B300 prices,
the gpu_slug join (used by fit_check + tps_estimator + cost_cells)
will fail to resolve. Watch for this when CP next refreshes.

---

## Reporting upstream

Both items are CP-side data quality issues, not whatcanirun bugs.
The right place to flag them is the ComputePrices issue tracker /
their API support channel. This doc exists so future contributors
who hit the same surprises see them once and don't need to re-debug.

## How we found these

Cross-validated `seeds/gpu_catalog_snapshot.yaml` (one-line `uv run
python` script that loads the YAML through `GpuCatalogRow` and prints
a 5-column table for the GPUs Kiely tabulates) against Kiely's
published spec tables. All `vram_gb` and `memory_bandwidth_gbps`
values agreed exactly; only `fp8_tflops` and missing-GPU-row issues
surfaced. The check is repeatable on any future snapshot refresh.
