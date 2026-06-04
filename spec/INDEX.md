# spec/INDEX.md — Milestone Tracker

Read [`SHARED.md`](SHARED.md) first. Then this index. Then the specific milestone you're working on.

---

## v1 Milestones (stdio, self-hosted, free public APIs only)

Critical path = 8 weekends at ~6h each (~54h implementation-only). Plan for **75–110h ship-ready** with debugging, docs, clone-install testing on a fresh host, and broken-slug edge cases. M04, M05, M10 parallelize. v1 ships as a clone-install repo (host-uv + Docker); PyPI publication is deferred to v2 (see [`M12-release.md`](M12-release.md) § Deferred to v2).

| # | Spec | Effort | Critical path | Status |
|---|---|---|---|---|
| M00 | [Bootstrap](M00-bootstrap.md) | 4h | ✓ | ✓ |
| M01 | [Catalog supplements](M01-catalog-supplements.md) | 1h | ✓ | ✓ |
| M02 | [ComputePrices client](M02-computeprices-client.md) | 4h | ✓ | ✓ |
| M03 | [Hugging Face model sync](M03-hf-model-sync.md) | 4h | ✓ | ✓ |
| M04 | [Artificial Analysis optional client](M04-aa-optional-client.md) | 4h | parallel | ✓ |
| M05 | [Workload profile seeds](M05-workload-profiles.md) | 1h | parallel | ✓ |
| M06 | [fit_check](M06-fit-check.md) | 4h | ✓ | ✓ |
| M07 | [tps_estimator](M07-tps-estimator.md) | 4h | ✓ | ✓ |
| M08 | [Cost cells join layer](M08-cost-cells.md) | 3h | ✓ | ✓ |
| M09 | [MCP server](M09-mcp-server.md) | 10h | ✓ | ✓ |
| M10 | [Benchmark seeds (public sources)](M10-benchmark-seeds.md) | 6h | parallel | ✓ ¹ |
| M11 | [Tests + golden-path + docs](M11-tests-docs.md) | 6h | ✓ | ✓ |
| M12 | [Release (clone-install)](M12-release.md) | 3h | ✓ | ✓ |
| **Total** | | **~54h impl / 75–110h ship-ready** | | |

¹ M10 partially shipped — verification tooling + gpu_catalog snapshot landed in PR-α (PR #17, squash `33ce718`), but Tier 1b public_benchmark_anchor cell curation was deferred to v2's M17 after the public benchmark source landscape proved infeasible for the cell shape this spec required. Public benchmark blogs don't publish steady-state per-stream decode-TPS (they publish aggregate-throughput-at-concurrency), source URLs rot fast (Spheron's M10-cited article 404s; replacement-article numbers fail bandwidth physics), and even paid first-principles sources (Kiely 2026 *Inference Engineering*) teach the bandwidth-heuristic methodology rather than publishing measured numbers. See `spec/M10-benchmark-seeds.md` preamble for the full rationale. v1's confidence ceiling is Tier 2 (AA provider_anchor, 0.7) for AA-tracked models and Tier 3 (bandwidth_heuristic, 0.6) otherwise; trust contract preserved by honest confidence reporting.

---

## v2 Trigger conditions (gated on v1 usage signals)

v2 is *not* a fixed roadmap. It's a set of conditional milestones that ship when v1 usage data justifies them.

| Trigger | v2 work this unlocks |
|---|---|
| ≥3 GitHub issues asking for "use this in Claude.ai web" | Remote HTTP transport + auth |
| ≥5 issues citing wrong/stale prices for a specific provider | Corrections API + provider scrape-health |
| `tps_source=requires_measurement` hit by >30% of `budget_to_plan` calls | GuideLLM-based measured benchmark publishing |
| ≥10 issues asking for on-prem TCO or reserved cloud comparison | Port v1-repo's on_prem + reserved_cloud math |
| Public dataset (M10 seeds) gets ≥100 downloads/month on HF | Weekly automated GuideLLM publishing pipeline |
| ComputePrices Enterprise tier adds `/api/v1/llm-benchmarks` from AA at reasonable cost | Replace own-benchmarks track with CP Enterprise integration |

---

## How to use this spec

1. `/setup-matt-pocock-skills` once at repo bootstrap (M00).
2. Pick the next ⬜ milestone in critical path order.
3. `/to-prd spec/M{NN}-*.md` → publishes a `ready-for-agent` issue.
4. `/to-issues #<issue-num>` → breaks into vertical-slice sub-issues.
5. For each sub-issue: `/tdd` discipline, one red-green-refactor cycle per behavior.
6. Commit per cycle. PR per milestone.
7. Update this index's Status column when merged.

---

## What's out of scope (permanently or until-proven-needed)

- **Claude.ai web custom connector** — OAuth 2.1 + RFC 9728 PRM + DCR. Currently has Claude.ai-side bugs. Re-evaluate in 6 months.
- **TimescaleDB / pricing time-series** — ComputePrices owns the history.
- **Live GuideLLM benchmark runs in v1** — public-source anchors only.
- **AA Pro / Premium Insights** — opaque pricing. Defer until AA-anchored TPS is shown load-bearing.
- **`recommend_stack` tool** — gated on quality data being routing-load-bearing.
- **`cost_for_workload` tool** — inverse of `budget_to_plan`; the LLM composes.
- **`generate_cost_report` tool** — Claude is already a markdown renderer.
- **More than 3 WorkloadProfiles in v1** — defer to usage signals.
- **More than 1 MCP prompt in v1** — `/benchmark-on-budget` is the headline.
