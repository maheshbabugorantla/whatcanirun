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
