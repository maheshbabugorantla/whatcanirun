# spec/SHARED.md — Shared Design Contract

This document is read **before any milestone**. Every M{NN}-*.md spec assumes you've internalized what's here. Conflicts between a milestone spec and this document resolve in favor of this document.

---

## Strategic context

The product answers one question: *"I have $X to spend on running model Y — which GPU, on which provider, for how long, on how many prompts?"*

No public tool answers this today. Components exist (ComputePrices for prices, HuggingFace for architecture, Artificial Analysis for quality/throughput) but nothing joins them into a budget-aware plan with VRAM fit checking, workload-aware token estimation, and honest confidence labeling.

That join — plus the trust envelope that backs it — is the product.

---

## The Trust Contract (the product's spine)

Anyone on the user spectrum gets the same response shape. Hobbyist learning LLM inference for the first time, or a power user cutting repetitive work. The output never lies, never bluffs, and never hides what it assumed. Persona-handling happens in the LLM client *interpreting* the response, not in the server *producing* it.

### TrustEnvelope shape

Every numerical tool response carries:

```python
class TrustEnvelope(BaseModel):
    sources: list[Source]                                # Each upstream that contributed a number
    confidence: float                                    # min(confidence_breakdown.values())
    confidence_breakdown: dict[ConfidenceDomain, float]  # Per-domain, weakest-link semantics
    assumptions: dict[str, Any]                          # What was held fixed
    caveats: list[str]                                   # What we explicitly do NOT model
    freshness: dict[str, datetime]                       # Per-source last-updated timestamps
    verify_links: list[str]                              # URLs the user can audit upstream

ConfidenceDomain = Literal[
    "pricing",              # ComputePrices data freshness + provider catalog completeness
    "fit_check",            # accuracy of model architecture + GPU VRAM specs
    "throughput",           # TpsEstimate.confidence for this cell
    "model_architecture",   # HF config.json freshness + family extraction success
    "gpu_specs",            # ComputePrices catalog completeness + supplement YAML coverage
    "workload_assumption",  # how grounded the assumed (avg_input_tokens, avg_output_tokens)
                            # are — user-elicited or argument-supplied = 0.95; silent default
                            # = 0.2; omit the key when no workload assumption was made.
                            # Only populated by tools that synthesize derived counts from a
                            # workload (e.g. budget_to_plan's est_total_prompts).
    "freshness",            # weakest-link cache age across all contributing sources
]

class Source(BaseModel):
    name: Literal["computeprices", "huggingface", "artificial_analysis",
                  "own_measured_benchmark", "public_benchmark_anchor",
                  "bandwidth_heuristic", "datasheet_yaml"]
    detail: str
    last_updated: datetime
    license_attribution: str | None
```

### Rollup semantics — explicit, not hand-wavy

`confidence` is `min(confidence_breakdown.values())`. Weakest-link by design.

A plan with `pricing=0.9, fit_check=0.85, throughput=0.0` (requires_measurement) has top-level `confidence=0.0` — and the breakdown shows the LLM client *exactly* which domain is the problem so it can tell the user "I can show you the cheapest GPU and confirm it fits, but I don't have a benchmark for this batch size."

### Calibration

- `own_measured_benchmark` (v2 only): 0.95 — reproducible GuideLLM methodology required
- `public_benchmark_anchor` (v1 M10): 0.80 — real numbers from blogs/articles, methodology unverified
- `provider_anchor` (AA): 0.7
- `bandwidth_heuristic_single_stream`: 0.6 — batch=1 only
- `requires_measurement`: 0.0 — no number returned at all
- `datasheet_yaml` for GPU specs domain: 0.99 — facts
- `workload_assumption` domain:
  - user explicitly supplied custom `avg_input_tokens` + `avg_output_tokens`: 1.0
  - user elicited a `workload_profile_slug` (interactive flow) OR client passed it as a tool argument: 0.95
  - server fell back to a silent default profile: 0.2  ← intentionally low so `confidence = min(...)` forces the LLM client to relay that the prompt count is hearsay
  - tool returned no derived prompt count (e.g. `find_cheapest_deployment`): omit the `workload_assumption` key entirely

**No throughput number returns confidence > 0.95 unless it came from `own_measured_benchmark`.** GPU specs (different domain) can reach 0.99.

### Staleness policy — freshness decays confidence

Breakpoints calibrated to actual upstream refresh cadences:

```python
def freshness_confidence(source: str, age: timedelta) -> float:
    if source == "computeprices":           # CP refreshes ~hourly
        if age < timedelta(hours=2):  return 0.95
        if age < timedelta(hours=24): return 0.75
        return 0.4
    elif source == "artificial_analysis":   # AA refreshes ~8×/day; our cache 6h
        if age < timedelta(hours=12): return 0.95
        if age < timedelta(hours=72): return 0.75
        return 0.4
    elif source == "huggingface":           # config.json rarely changes
        if age < timedelta(days=30):  return 0.95
        return 0.80
    elif source == "datasheet_yaml":
        return 0.99                         # manufacturer facts don't decay
    elif source == "public_benchmark_anchor":  # blog posts get stale
        if age < timedelta(days=90):  return 0.85
        if age < timedelta(days=365): return 0.70
        return 0.45
```

---

## ADRs (locked architectural decisions)

| ADR | Decision | Rationale |
|---|---|---|
| **ADR-001** | ComputePrices is canonical for GPU $/hr, LLM API $/M-token pricing, GPU base catalog | Live-verified: 71 providers, 66 GPUs with bandwidth+VRAM, hourly refresh, free 5k/hr with email-requested key |
| **ADR-002** | Hugging Face Hub is canonical for model architecture (`config.json` + safetensors metadata) | Free, unlimited public configs; no aggregator covers `n_layers`/`n_kv_heads`/`head_dim` |
| **ADR-003** | Artificial Analysis is **optional** enrichment | Free 1k/day tier verified to cover 14/16 open-weight models with populated TPS; v1 must function without it |
| **ADR-004** | Trust envelope on every numerical response | Single design principle handling any-persona usage; enforced in code |
| **ADR-005** | GPU `fp8_tflops_dense`, KV cache flags, form factor, MLA-vs-GQA flag in 12-row supplemental YAML | Not in ComputePrices schema; book §5.1.1 and datasheet facts |
| **ADR-006** | Benchmark cells published as Parquet on Hugging Face Datasets under CC-BY-4.0 | Strategic moat artifact; v1 seeds 20–30 from public sources; v2 adds GuideLLM-measured cells |
| **ADR-007** | v1 transport: stdio only. v2 adds Streamable HTTP with bearer token for Claude Code/Desktop. Claude.ai web out of scope. | Bearer sufficient for the two main MCP clients; Claude.ai OAuth currently has upstream bugs |
| **ADR-008** | v1 stack: FastMCP 2.x + Pydantic v2 + httpx + DuckDB-on-files. No Django, no SQL DB. | Stdio fast-start; self-host has zero infrastructure |
| **ADR-009** | v2 stack: Django 5.x + DRF + Postgres + Redis + Celery on Render | Standard production stack; reintroduce only when persistence + auth + scheduled jobs are load-bearing |
| **ADR-010** | TPS heuristic restricted to single-stream (batch=1). For batch>1, return `requires_measurement` unless measured benchmark exists. | Verified: linear batch scaling is 6× wrong in compute-bound regimes |
| **ADR-011** | Observability: Logfire (free 10M spans/mo) + Sentry (free 5K errors/mo) in v2 only. v1 logs to stderr. | Self-hosted v1 has no central observability surface |
| **ADR-012** | Auth in v2: email-OTP → bearer API key via Resend (free 3k/mo). No OAuth in v1 or v2. | Sufficient for Claude Code/Desktop; one shot to identify and gate quota |
| **ADR-013** | When ComputePrices unreachable, serve last-good local snapshot with `freshness` reflecting staleness; never fail tool calls outright | Single upstream dependency mitigated by 30-day rolling local cache |
| **ADR-014** | Cost-cells query layer is plain Python list joins for tool calls; DuckDB ONLY for `cost-cells://current` resource generation | Python for testable business logic; DuckDB for declarative resource materialization |
| **ADR-015** | **Raw + Projection storage pattern.** Full upstream response stored verbatim; Pydantic models pluck currently-known fields with `extra='ignore'`. Nested objects whose schema is undocumented or evolving (`evaluations`, `pricing`, `specs`, HF `config.json`) typed as `dict[str, Any]` or `dict[str, float \| None]`, never narrow-typed. | Upstream schemas DO change. AA's `evaluations` has 16+ fields where docs showed 10. ComputePrices adds new sub-objects per release. HF `config.json` varies per family. Narrow-typed Pydantic = breakage on every upstream release. |

---

## Domain Glossary

Used verbatim in code, tests, commit messages, PR titles, issue bodies.

- **Cost cell** — `(gpu, provider, model, quant, deployment_mode, batch, ctx) → (hourly_usd, decode_tps, cost_per_m_output_usd, trust_envelope)`. Atomic output unit.
- **Trust envelope** — `TrustEnvelope` Pydantic model. Required on every numerical tool response.
- **Confidence domain** — One of: `pricing`, `fit_check`, `throughput`, `model_architecture`, `gpu_specs`, `workload_assumption`, `freshness`. Top-level `confidence` is `min(confidence_breakdown.values())`. `workload_assumption` is only populated on responses that synthesize derived counts from a workload (e.g. `BudgetPlanRow.est_total_prompts`); omitted entirely otherwise. See the `ConfidenceDomain` Literal above for the per-domain semantics and the Calibration section for value ranges.
- **Deployment mode** — `cloud_gpu_rental`, `hosted_api_token`. (v2 adds `on_prem` with `tco_treatment` subfield.) The earlier 5-mode taxonomy is deprecated.
- **Op-point** — `(batch_size, context_length)` tuple.
- **Fit check** — Pure-math VRAM verdict. Returns `FitResult` with `weight_gb`, `kv_cache_gb`, `framework_overhead_gb`, `headroom_gb`, `blocking_reasons`, `sufficiency_caveat`. Never just a bool.
- **TPS source** — `own_measured | public_benchmark_anchor | provider_anchor | bandwidth_heuristic_single_stream | requires_measurement`.
- **Workload profile** — `(avg_input_tokens, avg_output_tokens)` seed. v1 ships 3: `code_completion`, `chat_assistant`, `batch_eval`.
- **Plan** — Ranked list of cost cells for a budget, with `hours_available`, `est_total_prompts`, `est_wallclock_minutes`, all under one trust envelope.

---

## Project layout

```
whatcanirun/
├── pyproject.toml
├── uv.lock
├── src/whatcanirun/
│   ├── server.py              # FastMCP entry point
│   ├── catalog/               # GPU, Model, Quant, WorkloadProfile (M01, M03, M05)
│   ├── pricing/               # ComputePrices, AA clients (M02, M04)
│   ├── inference/             # fit_check, tps_estimator (M06, M07)
│   ├── plan/                  # budget_planner, cost_cells join (M08)
│   ├── trust/                 # TrustEnvelope + per-tool builders
│   └── mcp_tools/             # The 6 MCP tool definitions (M09)
├── seeds/
│   ├── gpus_supplement.yaml
│   ├── quantizations.yaml
│   ├── workload_profiles.yaml
│   ├── tracked_models.yaml
│   ├── aa_slug_mapping.yaml
│   └── benchmark_cells.parquet
├── tests/
├── docs/
│   ├── PRD.md
│   ├── MCP.md
│   ├── TRUST.md
│   └── ADRs/
├── spec/
│   ├── INDEX.md
│   ├── SHARED.md              ← this file
│   └── M00..M12-*.md
├── CLAUDE.md
└── .claude/skills-lock.json
```

---

## Open decisions (require resolution before v2, not blocking for v1)

1. **Persistence migration shape (M13).** v1 is DuckDB-on-files. v2 needs Postgres for auth/quota. Two paths:
   - Path A: Migrate cost-cells layer to Postgres too (uniform store). +2h.
   - Path B: Keep cost cells in DuckDB-on-files, only auth in Postgres. Saves I/O. +0h, slight complexity.

2. **Benchmark dataset license.** ADR-006 proposes CC-BY-4.0. Confirm vs MIT or CC0.

3. **Hugging Face namespace** for the benchmark dataset. Personal account or new org?

4. **ComputePrices contact.** Email `api@computeprices.com` before v1 ships, requesting:
   - A free 5k/hr API key
   - Attribution language confirmation
   - Heads-up on planned schema changes (their roadmap mentions `/api/v1/llm-benchmarks` in v1.2)

---

## Verification status (locked in v2.1)

Confirmed live as of 25 May 2026:

- **ComputePrices `/api/v1/gpus`:** 66 GPUs; bandwidth+VRAM+architecture+fp16_tflops verified against datasheets. Missing `fp8_tflops` everywhere — supplement YAML required.
- **ComputePrices `/api/v1/llm-models`:** 214 models, no architecture fields. HF sync remains required.
- **AA `/api/v2/data/llms/models`:** 524 models in free-tier response. Curated slug mapping covers 14/16 tracked open-weight models. `median_output_tokens_per_second` confirmed populated for open-weight rows. Llama-3.3-70B notably absent under that slug — investigate during M04. Reasoning models have effort-level dimension (`-low`/`-medium`/`-high`).
- **AA `evaluations` schema:** 16+ fields where docs showed 10. NEW fields like `aime_25`, `lcr`, `terminalbench_hard`, `tau2`, `ifbench` confirmed present. Justifies the `dict[str, float | None]` typing per ADR-015.
- **Bandwidth heuristic:** verified within ±50% of real anchors at batch=1 single-stream. Verified ~6× overestimate at batch=128. v1 restricts heuristic to batch=1 only.
- **vLLM benchmark CLI:** vLLM docs themselves recommend GuideLLM for production benchmarking. v2 uses GuideLLM.

---

## Effort estimate

- **v1 implementation-only:** ~54h
- **v1 ship-ready** (with debugging, docs, clone-install testing on a fresh host, broken-slug edge cases, flaky upstream responses, benchmark-source validation, golden-path hardening): **~75–110h**

Plan for the upper bound. Pick the lower bound only if you're full-time for a week.

---

## Cost (verified)

- **v1 monthly:** $0–$1.25 (just an optional domain name)
- **v2 monthly minimum:** ~$22 (Render Starter + Postgres Basic + cron)
- **v2 monthly comfortable:** ~$72
- **Year-1 projection:** $200–$320 total

---

## The trust contract is the product

Every shortcut that compromises honesty in tool output destroys the only thing that differentiates this from a weekend GPU price comparison site. When in doubt: surface the caveat, lower the confidence, expose the assumption.
