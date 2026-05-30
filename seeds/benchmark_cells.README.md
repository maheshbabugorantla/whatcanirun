# benchmark_cells.parquet

20-30 hand-curated benchmark cells from public sources. Built in M10 — see `spec/M10-benchmark-seeds.md`.

This file is committed as Parquet (binary). The CSV-form schema is documented here for readers:

| Column | Type | Notes |
|---|---|---|
| gpu_slug | str | Joins ComputePrices |
| model_slug | str | Joins tracked_models |
| quant_slug | str | Joins quantizations |
| tp_size | int | Tensor-parallel rank count |
| batch_size | int | |
| context_length | int | |
| decode_tps | float | Output tokens/sec |
| prefill_tps | float \| null | Input tokens/sec |
| ttft_ms | float \| null | Time to first token |
| engine | enum | vllm \| sglang \| tensorrt_llm \| tgi \| other |
| engine_version | str | |
| measured_at | date | |
| source | enum | **v1: always `public_benchmark_anchor`** — never `own_measured` |
| source_url | str | Publicly accessible URL |
| notes | str | Methodology summary, 1-2 sentences |

**The Pydantic schema rejects `source="own_measured"` in v1.** This is enforced by validator, not just convention. v2's M17 introduces own_measured cells via GuideLLM runs.

## Methodology triage TODO

Two cells were flagged by the M10 PR-α bandwidth-heuristic prototype as
having decode_tps values that diverge >50% from the single-stream
bandwidth prediction (after MoE adjustment, where applicable). Re-review
the source URLs and either update or remove during Slice D Phase 1:

- **`(h100, mistral-7b, fp8, batch=1)`** — actual 140.5 tok/s; predicted
  ~359 tok/s on H100 SXM at fp8. Actual is 61% below ideal. Re-check the
  Mistral deployment-guide source for batch>1 disclosure or post-decode
  overhead that the cell may have absorbed unintentionally.
- **`(l40s, phi-4, bf16)`** — actual 52 tok/s; predicted ~23 tok/s on L40S
  GDDR6 at bf16. Actual is 2.25× ideal single-stream. Almost certainly
  the source reported batch>1 throughput or used speculative decoding;
  reclassify the cell or drop it.

These cells pass schema validation today; the heuristic check will run
as a warning (exit 1) once V1 ships, so they remain in the parquet but
get surfaced on every future sanity-check run until reconciled.
