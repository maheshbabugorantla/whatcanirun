# whatcanirun — Product Requirements

The public-facing PRD. Synthesized from
[`spec/SHARED.md`](../spec/SHARED.md),
[`spec/INDEX.md`](../spec/INDEX.md), and the per-milestone specs.
The internal specs remain the source of truth for milestone
acceptance criteria; this document is the reader-facing framing.

---

## What problem this solves

LLM inference users face a recurring question with no clean
answer:

> *"I have $X to spend on running model Y — which GPU, on which
> provider, for how long, on how many prompts?"*

The pieces exist:

- **ComputePrices** aggregates GPU rentals and hosted-API pricing.
- **Hugging Face** is canonical for model architecture.
- **Artificial Analysis** publishes provider throughput numbers.

But nothing joins them into a budget-aware plan that VRAM-checks
the model, estimates throughput at the right provenance tier, and
labels every number with honest confidence. That join — plus the
trust envelope behind it — is the product.

## Who it's for

- **Hobbyists** learning LLM inference for the first time, who
  need the "what fits in my budget?" question answered without
  spending a weekend reading benchmark blog posts.
- **Power users** cutting repetitive work who want machine-readable
  per-row trust data so they can build their own dashboards or
  compare cells programmatically.

Same response shape for both personas. The LLM client decides how
verbose to be; the server decides what's true.

## The trust contract is the product

Every numerical response carries a structured `TrustEnvelope`:

- **`sources`** — every upstream that contributed
- **`confidence`** — weakest-link rollup (`min(breakdown.values())`)
- **`confidence_breakdown`** — per-domain confidence values
- **`assumptions`** — what was held fixed
- **`caveats`** — what we explicitly do NOT model
- **`freshness`** — per-source last-updated timestamps
- **`verify_links`** — URLs the user can audit upstream

Full deep-dive in [`TRUST.md`](TRUST.md). One-line summary: the
server cannot return a number it can't source, and never bluffs
about how sure it is.

## v1 surface

The stdio MCP server exposes:

**Numerical tools** — each row carries its own `TrustEnvelope`:

- **`fit_check(model_slug, gpu_slug, quant_slug, tp_size=1,
  batch_size=1, context_length=4096)`** — pure-math VRAM verdict
  with weight/KV-cache/overhead breakdown, headroom, blocking
  reasons, sufficiency caveat.
- **`find_cheapest_deployment(model_slug, quant_slug=None,
  batch_size=1, context_length=4096, region=None, top_n=10)`** —
  ranked list of cost cells across providers, GPUs, quants.
  `region` is accepted but a no-op in v1 (CP doesn't structure
  region per gpu-price row).
- **`compare_deployment_modes(model_slug, gpu_slug, quant_slug,
  batch_size, context_length, workload_profile_slug)`** —
  side-by-side cloud-GPU-rental vs hosted-API-token economics
  at one op-point, conditioned on a workload profile.
- **`budget_to_plan(budget_usd, model_slug,
  workload_profile_slug=None, quant_slug=None, top_n=3)`** —
  ranked plan with `hours_available`, `est_total_prompts`,
  `est_wallclock_minutes` per row.

**Catalog tools** — facts only, no envelope:

- **`list_catalog`** — enumerate known GPU SKUs, providers, tracked
  models.
- **`resolve_model(model_slug, hf_repo_id)`** — persist a
  `(slug, repo_id)` mapping and sync architecture from HF for
  unknown-model elicitation.

**Resources:**

- **`cost-cells://current`** — full cost-cell table as Parquet.
- **`cost-cells://provenance`** — JSON document with upstream
  source attributions, license terms, and audit links.

**Prompt:**

- **`/benchmark-on-budget`** — guided template for the headline
  question.

## v1 milestones

Tracked in [`spec/INDEX.md`](../spec/INDEX.md). Status as of this
PRD:

| # | Milestone | Status |
|---|---|---|
| M00 | Bootstrap | ✓ |
| M01 | Catalog supplements | ✓ |
| M02 | ComputePrices client | ✓ |
| M03 | Hugging Face model sync | ✓ |
| M04 | Artificial Analysis optional client | ✓ |
| M05 | Workload profile seeds | ✓ |
| M06 | `fit_check` | ✓ |
| M07 | `tps_estimator` | ✓ |
| M08 | Cost cells join layer | ✓ |
| M09 | MCP server | ✓ |
| M10 | Benchmark seeds (public sources) | ✓ (partial — see below) |
| M11 | Tests + golden-path + docs | ✓ |
| M12 | Release (clone-install) | in flight |

### M10 partial-ship note

M10's verification tooling and the GPU catalog snapshot landed,
but Tier 1b `public_benchmark_anchor` cell curation was removed
from v1 — Tier 1b is not tied to a specific v2 milestone and
reviving it would need a separate decision and a fresh source
landscape. v2's M17 is the unlock for Tier 1a
(`own_measured`) via GuideLLM-measured cells, not Tier 1b. The
public benchmark source landscape proved infeasible for the
cell shape v1 required:

- Public benchmark blogs publish aggregate-throughput-at-concurrency,
  not per-stream steady-state decode-TPS.
- Source URLs rot fast (M10's URL test caught multiple 404s
  shortly after curation).
- Paid first-principles sources (Kiely 2026 *Inference
  Engineering*) teach methodology, not measurements.

v1's throughput confidence ceiling is Tier 2
(AA `provider_anchor`, 0.70) for AA-tracked models and Tier 3
(`bandwidth_heuristic_single_stream`, 0.60) otherwise. The trust
contract holds because the confidence values report this honestly.

## v2 — what's gated on usage signal

v2 is not a fixed roadmap. Each milestone ships when v1 usage
data justifies it. The current trigger table:

| Trigger | v2 work it unlocks |
|---|---|
| ≥3 GitHub issues asking for "use this in Claude.ai web" | Remote HTTP transport + auth (see [ADR-007](ADRs/ADR-007-stdio-transport.md), [ADR-012](ADRs/ADR-012-auth-email-otp.md)) |
| ≥5 issues citing wrong/stale prices for a specific provider | Corrections API + provider scrape-health surface |
| `tps_source=requires_measurement` hit by >30% of `budget_to_plan` calls | GuideLLM-based own-measured benchmark publishing (unlocks Tier 1a) |
| ≥10 issues asking for on-prem TCO or reserved-cloud comparison | Port on_prem + reserved_cloud cost math |
| Benchmark dataset gets ≥100 downloads/month on HF | Weekly automated GuideLLM publishing pipeline |
| ComputePrices Enterprise tier exposes `/api/v1/llm-benchmarks` at reasonable cost | Replace own-benchmarks track with CP Enterprise integration |

## What's explicitly out of scope

These are durable scoping decisions, not "TODOs." They're out
because they conflict with the product's spine or aren't
load-bearing yet.

- **Claude.ai web custom connector** — blocked on upstream Claude.ai
  OAuth bugs ([ADR-007](ADRs/ADR-007-stdio-transport.md)).
- **TimescaleDB / pricing time-series** — ComputePrices already owns
  the history.
- **Live GuideLLM benchmark runs in v1** — v2 work
  ([ADR-006](ADRs/ADR-006-benchmark-cells-parquet.md)).
- **AA Pro / Premium Insights** — opaque pricing; defer until
  AA-anchored throughput is shown load-bearing.
- **`recommend_stack` tool** — gated on data quality being
  routing-load-bearing.
- **`cost_for_workload` tool** — the LLM client composes this
  from `budget_to_plan` + workload arithmetic.
- **`generate_cost_report` tool** — Claude is already a markdown
  renderer.
- **More than 3 WorkloadProfiles in v1** — defer to usage signals.
- **More than 1 MCP prompt in v1** — `/benchmark-on-budget` is the
  headline.
- **Provider rentability or stock availability modeling** — we
  model pricing, not whether the SKU is in stock at the listed
  price. Reflected on every cost-cell row's
  `availability_caveat`.

## How to install

See [`MCP.md`](MCP.md) for per-client configuration blocks
(Claude Desktop, Claude Code, Cursor, Cline) and troubleshooting.

## License

MIT — declared in `pyproject.toml`. The benchmark *dataset*
(published as Parquet on Hugging Face Datasets) ships under
CC-BY-4.0 per ADR-006.

## References

- [`TRUST.md`](TRUST.md) — the trust contract in detail.
- [`MCP.md`](MCP.md) — installation per client.
- [`ADRs/`](ADRs/) — the 15 locked architectural decisions.
- [`../spec/SHARED.md`](../spec/SHARED.md) — internal design
  contract (canonical source for v1 acceptance criteria).
- [`../spec/INDEX.md`](../spec/INDEX.md) — milestone tracker.
