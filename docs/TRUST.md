# The Trust Contract

The trust contract is the product. Everything else — the catalog
sync, the cost-cell join, the MCP wire format — is plumbing in
service of one promise: **the server's numerical output never
lies, never bluffs, and never hides what it assumed.**

Hobbyist learning LLM inference for the first time, or a power
user cutting repetitive work — same response shape. Persona
handling happens in the LLM client *interpreting* the response,
not in the server *producing* it.

This document explains what the contract is, why it's structured
the way it is, what the per-domain confidence values actually
mean, how freshness decays them, the 4-tier throughput provenance
ladder, the things the server explicitly does not model, and the
attribution due to every upstream.

If you just want the one-line summary: every numerical tool
response carries a `trust_envelope` with weakest-link confidence,
audit links, and an enumerated list of caveats. The server cannot
return a number it can't source.

---

## TrustEnvelope shape

```python
class TrustEnvelope(BaseModel):
    sources: list[Source]                                # which upstreams contributed
    confidence: float                                    # min(confidence_breakdown.values())
    confidence_breakdown: dict[ConfidenceDomain, float]  # per-domain, weakest-link semantics
    assumptions: dict[str, Any]                          # what was held fixed
    caveats: list[str]                                   # what we explicitly do NOT model
    freshness: dict[str, datetime]                       # per-source last-updated timestamps
    verify_links: list[str]                              # URLs the user can audit upstream
```

Every numerical tool — `fit_check`, `find_cheapest_deployment`,
`compare_deployment_modes`, `budget_to_plan` — returns rows that
each carry their own envelope. Catalog-fact tools
(`list_catalog`, `resolve_model`) are envelope-exempt because
they don't synthesize a number — they return identifiers.

---

## Weakest-link rollup, not an average

`confidence = min(confidence_breakdown.values())`. Always. Never
an average, never a weighted sum.

A plan with `pricing=0.9, fit_check=0.85, throughput=0.0`
(throughput unknowable, source = `requires_measurement`) has
top-level `confidence = 0.0`. The breakdown shows the LLM client
exactly which domain is the problem — so it can tell the user
"I can show you the cheapest GPU and confirm the model fits in
VRAM, but I don't have a benchmark for this batch size."

Averaging would let a strong pricing number paper over a missing
throughput number. The weakest link is the point.

---

## The confidence domains

Six domains can appear in a cost-cell envelope; a seventh
(`workload_assumption`) is conditional.

### `pricing`

The freshness of ComputePrices pricing data plus provider catalog
completeness. ComputePrices refreshes hourly; if our snapshot is
younger than 2h, this domain reads 0.95. Beyond 24h, it drops to
0.4. ADR-013 covers the case where CP is unreachable — the
snapshot is served with `freshness.computeprices` reflecting how
stale the bytes are, and this domain's confidence follows.

### `fit_check`

How sure the pure-math VRAM verdict is. The check itself is
deterministic — it's the inputs (model architecture from HF, GPU
VRAM from CP supplemented by datasheet YAML) whose confidence
flows through. A `fits=True` from `fit_check` is necessary, not
sufficient: every `FitResult` carries `sufficiency_caveat`
spelling out which kernel-acceptance, framework, or driver
question the math doesn't answer.

### `throughput`

The TPS estimate's confidence, sourced from the
`TpsEstimate.source` tier (see § 4-tier provenance ladder below).
No throughput number ever exceeds 0.95 in v1 because the highest
tier (`own_measured`) is v2 territory; the published seed
(`public_benchmark_anchor`, 0.80) was deferred to v2's M17 after
the M10 audit determined public sources don't publish per-stream
decode-TPS in our cell shape.

### `model_architecture`

The Hugging Face `config.json` freshness, plus how cleanly we
extracted the architecture family (Llama, DeepSeek-MLA, Mistral,
Qwen, etc.). Config files rarely change after release, so
freshness here decays slowly (0.95 within 30 days, 0.80 beyond).
Family-extraction failure (an unknown `architectures` array)
routes the model to the unknown-model elicitation flow rather
than degrading silently.

### `gpu_specs`

ComputePrices catalog completeness plus the 12-row supplement
YAML (ADR-005). Manufacturer datasheet facts — VRAM, memory
bandwidth, `fp8_tflops_dense`, form factor — don't decay, so the
datasheet-derived rows hold 0.99. The supplement is what closes
ComputePrices' gaps (CP's reported `fp8_tflops` mislabels FP16
dense values on several SKUs; see
[`CP-DATA-QUALITY.md`](CP-DATA-QUALITY.md)).

### `freshness`

The weakest-link cache age across every source that contributed
to the response. If pricing is 1h fresh but the AA enrichment was
fetched 48h ago, `freshness` collapses to whichever curve is
worst. This is the domain that lets the LLM client say "this
answer is built on a 2-day-old snapshot" without having to
introspect every source individually.

### `workload_assumption` *(conditional)*

Only populated by tools that synthesize a derived prompt count
from a workload profile — `budget_to_plan`'s `est_total_prompts`,
`est_wallclock_minutes`. Omit the key entirely on responses
that don't synthesize a workload-dependent number (e.g.
`find_cheapest_deployment`).

Calibration:

- User explicitly supplied custom `avg_input_tokens` +
  `avg_output_tokens`: **1.0**
- User elicited a `workload_profile_slug` or the client passed
  one as a tool argument: **0.95**
- Server fell back to a silent default profile: **0.2** —
  intentionally low so the `min(...)` rollup forces the LLM
  client to relay that the prompt count is hearsay

---

## The 4-tier throughput provenance ladder

Throughput is the easiest number in this whole stack to lie
about. The estimator commits to a tier per cell and never
upgrades silently.

| Tier | `TpsEstimate.source` | Confidence | When it fires |
|---|---|---|---|
| 1a | `own_measured` *(v2)* | 0.95 | Reproducible GuideLLM run with methodology disclosed. **v1 never returns this** — we have no own benchmarks yet. |
| 2 | `provider_anchor` | 0.70 | AA enrichment surfaced a `median_output_tokens_per_second` for the model. |
| 3 | `bandwidth_heuristic_single_stream` | 0.60 | Pure arithmetic: `bandwidth_gbps / weights_bytes_per_token * 0.75`. Batch=1 only. |
| 4 | `requires_measurement` | 0.00 | None of the above applied. Server returns the cell with throughput = None and confidence = 0. |

Tier 1b (`public_benchmark_anchor`, 0.80) was scoped for M10
public-source seeds but **deferred to v2's M17** during the M10
audit — public benchmark blogs do not publish per-stream
steady-state decode-TPS in the shape our cell schema requires,
and curated seeds rotted faster than we could maintain them.
The validator still rejects `own_measured` rows in v1 so the
ladder can't be silently raised.

The 0.75 efficiency factor in Tier 3 follows Kiely 2026,
*Inference Engineering*, §2.4.2 "LLM Inference Bottlenecks" —
the load-bearing assumption being that LLM decode is
memory-bound at low-to-medium batch sizes. **For batch > 1, the
estimator returns Tier 4 (`requires_measurement`)** rather than
guessing — verified that linear batch scaling overestimates by
~6× in compute-bound regimes (ADR-010).

---

## Freshness policy

Decay curves calibrated to actual upstream refresh cadences:

| Source | Fresh | Aging | Stale |
|---|---|---|---|
| ComputePrices (hourly refresh) | < 2h → 0.95 | < 24h → 0.75 | ≥ 24h → 0.40 |
| Artificial Analysis (~8×/day) | < 12h → 0.95 | < 72h → 0.75 | ≥ 72h → 0.40 |
| Hugging Face `config.json` | < 30d → 0.95 | — | ≥ 30d → 0.80 |
| Datasheet YAML | always 0.99 | — | — (manufacturer facts don't decay) |
| Public benchmark anchor *(v2)* | < 90d → 0.85 | < 365d → 0.70 | ≥ 365d → 0.45 |

The minimum across all contributing sources lands in the
`freshness` confidence domain. The raw timestamps are also
exposed via `TrustEnvelope.freshness` so the caller can render
"based on a snapshot from 3 hours ago" without re-deriving the
math.

---

## What the server explicitly does NOT model

Surfaced verbatim on every relevant response in the `caveats`
list, and consolidated in the `cost-cells://provenance` resource:

- **Provider rentability or stock availability.** We model
  pricing, not whether the SKU is in stock at the listed price.
- **Real-time latency / time-to-first-token.** v1 reports
  steady-state throughput only.
- **Kernel-level acceptance of the chosen quantization on the
  chosen GPU.** `fit_check` proves the bytes fit; whether vLLM
  or your chosen runtime accepts the kernel is on you.
- **Tensor-parallel communication efficiency across
  heterogeneous links.** Single-GPU and homogeneous multi-GPU
  only.
- **Provider runtime compatibility** (CUDA / driver / framework
  versions). The cell tells you the math; the runtime contract
  is between you and the provider.
- **Batch > 1 throughput in v1.** Linear batch scaling is ~6×
  wrong in compute-bound regimes (ADR-010). For batch > 1, the
  estimator returns `requires_measurement` unless an own-measured
  cell exists (v2).
- **On-prem TCO.** v2 work, gated on usage signal. v1 ships
  `cloud_gpu_rental` + `hosted_api_token` modes only.
- **Reserved-instance or spot preemption probability.** On-demand
  pricing only.
- **Hosted-API rate limits or per-key quota caps.**

Hosted-API responses additionally **do not carry** the
`fit_check`, `model_architecture`, or `gpu_specs` confidence
domains — those domains don't apply to remote APIs where you
don't see the underlying GPU. Hosted-API envelopes use exactly
three domains: `pricing`, `throughput`, `freshness` (plus the
conditional `workload_assumption`).

---

## Attribution and licenses

Pretty much every number in this product comes from someone
else's work. Per-source attribution is also served live at
`cost-cells://provenance`.

- **ComputePrices** (https://www.computeprices.com) — GPU $/hr,
  LLM API $/M-token, GPU base catalog. Used under ComputePrices'
  Terms of Service.
- **Hugging Face Hub** (https://huggingface.co) — model
  architecture (`config.json` + safetensors metadata). Public
  configs only; license terms per-repo on Hugging Face.
- **Artificial Analysis** (https://artificialanalysis.ai) —
  optional; provider-anchored throughput. Used under AA's Free
  Tier terms.
- **Datasheet YAML** — manufacturer-published GPU facts curated
  from datasheets and Kiely 2026 §5.1.1. MIT
  (project-controlled).

Citations are also injected into `TrustEnvelope.verify_links` on
each response so the LLM client can hand the audit URL straight
to the user.

---

## Methodology citations

The TPS-estimator's Tier 3 bandwidth heuristic
(`KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75`, applied as
`predicted_tps = bandwidth_gbps / weights_bytes_per_token * 0.75`)
follows Kiely 2026, *Inference Engineering*, §2.4.2 "LLM
Inference Bottlenecks." The book teaches the arithmetic-intensity
analysis that justifies treating LLM decode as memory-bound at
low-to-medium batch sizes — the load-bearing assumption for
Tier 3's single-stream estimates.

Recommended reading for understanding the heuristic + the
GPU-spec tables ComputePrices reports:

- Kiely, Philip. *Inference Engineering.* 2026.
  Specifically §2.4 (Calculating Inference Bottlenecks) and
  §3.2 (GPU Architecture Generations).
