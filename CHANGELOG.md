# Changelog

All notable changes to whatcanirun are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

Date filled in at tag-cut (M12 Slice G). First public release. v1 ships as a clone-install repo for power users
on a host with Python 3.12 + `uv` (or Docker). PyPI publication and MCP
registry submissions are deferred to v2 once the tool surface stabilizes
through real usage; see
[`spec/M12-release.md`](spec/M12-release.md) § "Deferred to v2" for the
rationale.

### Added — MCP server surface (M09)

**Numerical tools** — every response carries a `TrustEnvelope`:

- `fit_check(model_slug, gpu_slug, quant_slug, tp_size=1, batch_size=1,
  context_length=4096)` — pure-math VRAM verdict with weight / KV-cache
  / framework-overhead breakdown, headroom, blocking reasons, mandatory
  `sufficiency_caveat`.
- `find_cheapest_deployment(model_slug, quant_slug=None, batch_size=1,
  context_length=4096, region=None, top_n=10)` — ranked list of cost
  cells across providers, GPUs, quants. Per-row envelope contract:
  every list element carries its own envelope.
- `compare_deployment_modes(model_slug, gpu_slug, quant_slug,
  batch_size, context_length, workload_profile_slug)` — side-by-side
  cloud-GPU-rental vs. hosted-API-token economics at one op-point.
- `budget_to_plan(budget_usd, model_slug, workload_profile_slug=None,
  quant_slug=None, top_n=3)` — ranked plan with `hours_available`,
  `est_total_prompts`, `est_wallclock_minutes` per row.

**Catalog tools** — envelope-exempt by design (no synthesized number):

- `list_catalog()` — GPUs, providers, tracked models, quantizations,
  workload profiles.
- `resolve_model(model_slug, hf_repo_id)` — persists a `(slug, repo_id)`
  mapping and triggers HF sync; backs the unknown-model elicitation
  flow.

**Resources:**

- `cost-cells://current` — full cost-cell table as Parquet
  (DuckDB-materialized — ADR-014).
- `cost-cells://provenance` — JSON document with upstream source
  attributions, license terms, audit links.

**Prompt:**

- `/benchmark-on-budget` — guided template for the headline budget
  question.

### Added — release tooling (M12)

- `whatcanirun-mcp prefetch` — synchronous warmup CLI; runs
  `HfModelSync.sync_all_tracked` and `load_runtime_deps()`
  with per-source stderr progress so the cold-cache 1-3s delay is an
  observable operator step instead of a hidden first-call surprise.
- `whatcanirun-mcp --version` — prints the package version; used by
  the clean-machine smoke gate.
- `scripts/install_host_uv.sh` — canonical host-uv install path
  (`git clone` → `uv sync` → `prefetch` → release-gate test → MCP
  client config block with the absolute repo path substituted).
- `Dockerfile` + `scripts/run_mcp_docker.sh` — Docker fallback path
  with `python:3.12-slim` runtime, named cache volume
  (`whatcanirun-cache:/var/cache/whatcanirun`), and env-var passthrough
  for the three optional upstream keys.
- `tests/release/test_stdio_install.py` (`@pytest.mark.release`) —
  release gate driving the installed binary over stdio with FastMCP's
  `Client` + `StdioTransport`; asserts the trust-envelope invariants
  on every numerical response.

### Added — pre-M09 milestones

- **M00 — Bootstrap.** Project scaffolding, FastMCP skeleton, CI gate,
  mattpocock skills workflow.
- **M01 — Catalog supplements.** `seeds/gpus_supplement.yaml` covers
  CP gaps (fp8 TFLOPS, KV-cache flags, form factor, MLA-vs-GQA family);
  `seeds/quantizations.yaml` defines the 10 quant slugs.
- **M02 — ComputePrices client.** Async httpx client for `/gpus`,
  `/gpu-prices`, `/llm-models`; latest+snapshots cache layout;
  exponential-backoff retries on 429/5xx; snapshot fallback per
  [ADR-013](docs/ADRs/ADR-013-snapshot-fallback.md).
- **M03 — Hugging Face model sync.** `HfModelSync.sync_model` and
  `sync_all_tracked`; raw config persisted byte-identical before
  projection per [ADR-015](docs/ADRs/ADR-015-raw-projection-pattern.md);
  architecture-family detection with MLA support for DeepSeek.
- **M04 — Artificial Analysis optional client.** AA `/api/v2/llms/models`
  client gated on `AA_API_KEY`; provider_anchor (Tier 2) throughput
  source; AA attribution string surfaces in every dependent
  `TrustEnvelope.sources` entry and in
  `cost-cells://provenance`.
- **M05 — Workload profile seeds.** Three v1 profiles
  (`code_completion`, `chat_assistant`, `batch_eval`) covering the
  representative `(avg_input_tokens, avg_output_tokens)` shapes.
- **M06 — `fit_check`.** Pure-math VRAM verdict returning `FitResult`
  with weight / KV-cache / framework-overhead breakdown, headroom,
  blocking reasons, and mandatory `sufficiency_caveat`.
- **M07 — `tps_estimator`.** Four-tier TPS provenance ladder
  (`own_measured` reserved for v2, `public_benchmark_anchor`,
  `provider_anchor`, `bandwidth_heuristic_single_stream`,
  `requires_measurement`); per
  [ADR-010](docs/ADRs/ADR-010-tps-single-stream.md) the heuristic is
  single-stream only.
- **M08 — Cost cells join layer.** Tool-call path is plain Python
  list/dict joins over in-memory caches; DuckDB reserved for the
  `cost-cells://current` resource materialization
  ([ADR-014](docs/ADRs/ADR-014-duckdb-resource-only.md)). Split
  enforced by an AST grep test.
- **M09 — MCP server.** Tool / resource / prompt surface above; full
  trust-envelope wiring; three-case unknown-model dispatcher
  (Case 1 lazy-sync, Case 2 partial-cell, Case 3 elicitation);
  compaction-survival hooks.
- **M10 — Benchmark seeds (partial-ship).** Verification tooling
  (V1 sanity-check + V2 merge + Slice C URL test) and the
  `gpu_catalog_snapshot.yaml` landed. Tier 1b public-benchmark cell
  curation was REMOVED from v1 — see § "Known limitations" below.
- **M11 — Tests + golden-path + docs.** Release-gating golden-path
  test (`test_golden_path_v1_release_gate`); per-client install docs
  ([`docs/MCP.md`](docs/MCP.md)); trust-contract deep-dive
  ([`docs/TRUST.md`](docs/TRUST.md)); 15-ADR documentation tree;
  synthesized PRD ([`docs/PRD.md`](docs/PRD.md)).

### Added — locked ADRs

[`docs/ADRs/`](docs/ADRs/) documents the 15 architectural decisions
that survive into v1:

- [ADR-001](docs/ADRs/ADR-001-computeprices-canonical.md) —
  ComputePrices is the canonical pricing + GPU catalog source.
- [ADR-002](docs/ADRs/ADR-002-huggingface-canonical-architecture.md) —
  Hugging Face is the canonical architecture source.
- [ADR-003](docs/ADRs/ADR-003-aa-optional-enrichment.md) — Artificial
  Analysis is optional enrichment.
- [ADR-004](docs/ADRs/ADR-004-trust-envelope-required.md) —
  TrustEnvelope is required on every numerical response.
- [ADR-005](docs/ADRs/ADR-005-gpu-supplement-yaml.md) — GPU-supplement
  YAML covers CP schema gaps.
- [ADR-006](docs/ADRs/ADR-006-benchmark-cells-parquet.md) — Benchmark
  cells (v2-future) ship as CC-BY-4.0 Parquet on HF Datasets.
- [ADR-007](docs/ADRs/ADR-007-stdio-transport.md) — v1 transport is
  stdio only.
- [ADR-008](docs/ADRs/ADR-008-v1-stack.md) — v1 stack is FastMCP +
  Pydantic + httpx + DuckDB-on-files (no Django, no SQL DB).
- [ADR-009](docs/ADRs/ADR-009-v2-stack.md) — v2 stack adds
  FastAPI + Postgres for the remote HTTP transport.
- [ADR-010](docs/ADRs/ADR-010-tps-single-stream.md) — TPS heuristic
  is single-stream only.
- [ADR-011](docs/ADRs/ADR-011-observability-v2-only.md) — No
  centralized observability target in v1; Python logging in clients,
  stderr captured by MCP client.
- [ADR-012](docs/ADRs/ADR-012-auth-email-otp.md) — v2 auth is
  email-OTP → bearer API key (no OAuth).
- [ADR-013](docs/ADRs/ADR-013-snapshot-fallback.md) — Snapshot
  fallback when ComputePrices is unreachable.
- [ADR-014](docs/ADRs/ADR-014-duckdb-resource-only.md) — DuckDB only
  for resource generation; tool calls use Python joins.
- [ADR-015](docs/ADRs/ADR-015-raw-projection-pattern.md) — Raw +
  projection storage; evolving schemas typed as
  `dict[str, Any]` / `dict[str, float | None]`.

### Fixed

- `httpx.AsyncClient` instantiations in CP, AA, and HF clients now
  pass `follow_redirects=True`. Surfaced by the M12 release-gate
  test: CP's Vercel host added a www → apex 308 permanent redirect
  that httpx does not follow by default, causing every cold-start
  CP fetch to crash the server-side handler.

### Attributions

v1 inference plans depend on data from:

- **[ComputePrices](https://www.computeprices.com/)** —
  GPU rental + hosted-LLM-API pricing. Anonymous reads with email-
  requested `COMPUTEPRICES_API_KEY` for the 5k/hr tier.
- **[Hugging Face](https://huggingface.co/)** — model architecture
  via `config.json`. Optional `HF_TOKEN` for gated repos.
- **[Artificial Analysis](https://artificialanalysis.ai/)** — quality
  scores and per-provider throughput aggregates. Requires
  `AA_API_KEY` (free tier); attribution string surfaces on every
  AA-derived `TrustEnvelope.sources` entry per AA's API terms.
- **[Kiely 2026 *Inference Engineering*](https://www.inferenceengineering.com/)** —
  bandwidth-heuristic methodology citation in
  [`docs/TRUST.md`](docs/TRUST.md). Provides Tier-3 throughput
  formulas; book itself is not bundled.

### Known limitations

- **Tier 1b `public_benchmark_anchor` cells removed from v1.** The
  public benchmark source landscape (blogs, vendor whitepapers,
  paid books) does not publish steady-state per-stream decode-TPS in
  the shape v1 required — sources publish aggregate-throughput-at-
  concurrency, URLs rot fast, and methodology-only references like
  Kiely 2026 teach the heuristic rather than measurements. v1's
  throughput confidence ceiling is Tier 2 (AA `provider_anchor`,
  0.70) for AA-tracked models and Tier 3
  (`bandwidth_heuristic_single_stream`, 0.60) otherwise. Tier 1b is
  NOT tied to a specific v2 milestone — reviving it would need a
  separate decision and a fresh source landscape.
- **Tier 1a `own_measured` cells deferred to v2.** v2's M17 unlocks
  Tier 1a via GuideLLM-measured benchmark publishing.
- **No PyPI artifact in v1.** Power-user clone-install only; PyPI
  + MCP registry submissions are v2 work.
- **No remote HTTP transport.** Stdio only per
  [ADR-007](docs/ADRs/ADR-007-stdio-transport.md); v2 adds HTTP with
  bearer-token auth per
  [ADR-012](docs/ADRs/ADR-012-auth-email-otp.md).
- **Single-stream TPS heuristic only.** `batch_size > 1` returns
  `requires_measurement` unless a measured benchmark cell exists per
  [ADR-010](docs/ADRs/ADR-010-tps-single-stream.md). Verified ~6×
  wrong at `batch=128` with linear scaling.
- **No rentability modeling.** The server models pricing, not stock
  availability — reflected on every cost-cell row's
  `availability_caveat`.

### License

- Project: **MIT** — declared in `pyproject.toml`. See
  [`LICENSE`](LICENSE).
- Benchmark *dataset* (v2-future, on HF Datasets): **CC-BY-4.0** per
  [ADR-006](docs/ADRs/ADR-006-benchmark-cells-parquet.md).

[0.1.0]: https://github.com/maheshbabugorantla/whatcanirun/releases/tag/v0.1.0
