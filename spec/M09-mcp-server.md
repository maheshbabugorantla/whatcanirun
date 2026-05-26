# M09 — MCP Server (5 tools, 2 resources, 1 prompt)

**Status:** ⬜ Not started
**Effort:** 10h (12h realistic with prompt + instructions polish)
**Dependencies:** M01–M08
**Unblocks:** M11 (golden-path test), M12 (release)

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

A FastMCP server exposing the public product surface. Stdio transport. Five tools, two resources, one prompt. The `FastMCP.instructions` string is the single most important piece of prose in the whole project — it's what makes the LLM client speak in trust-contract-respecting voice without further training.

---

## Public surface

### Tools

1. **`list_catalog()`** → `{gpus, models, quantizations, workload_profiles, providers}`.
   One-call dropdown helper for clients building UIs.

2. **`find_cheapest_deployment(model_slug, quant_slug?, batch_size=1, context_length=4096, region?, top_n=10)`** → ranked `list[CostCell]`.
   The basic price-comparison tool. No budget; just "what's cheapest for this op-point?"

3. **`compare_deployment_modes(model_slug, gpu_slug, quant_slug, batch_size, context_length, workload_profile_slug)`** → `DeploymentComparison`.
   Side-by-side of `cloud_gpu_rental` vs `hosted_api_token` for this op-point, with the inference-engineering-book §7.4.2 break-even framing.

4. **`fit_check(model_slug, gpu_slug, quant_slug, tp_size, batch_size, context_length)`** → `FitResult` with trust envelope.
   Standalone wrapper over M06. Always populates `sufficiency_caveat`.

5. **`budget_to_plan(budget_usd, model_slug, workload_profile_slug?, quant_slug?, top_n=3)`** → ranked `list[BudgetPlanRow]`.
   **The headline tool.** Each row:
   ```python
   class BudgetPlanRow(BaseModel):
       cost_cell: CostCell
       hours_available: float                 # budget_usd / hourly_usd
       est_total_prompts: int                 # using workload profile
       est_total_output_tokens: int
       est_wallclock_minutes: float
       cost_per_m_output_usd: float
       trust_envelope: TrustEnvelope          # includes availability_caveat
   ```

### Resources

1. **`cost-cells://current`** — Parquet materialization of all current cost cells. Re-rendered when **any** contributing cache invalidates (CP / AA / HF / YAML hash). Resource carries `generated_at` + per-source freshness map.

2. **`cost-cells://provenance`** — JSON document. Every data source named with attribution string, every ADR linked, the "what we DO NOT model" list, license declarations. The single document anyone can audit to decide whether to trust this server.

### Prompts

1. **`/benchmark-on-budget`** — guided workflow. Takes `budget_usd` and optional `model_slug`. Chains `list_catalog` (if model missing) → `fit_check` × candidate GPUs → `budget_to_plan`. Useful for first-time users who don't know which catalog item maps to their idea.

---

## The FastMCP.instructions string

This is the prose the LLM client reads at every tool call. It defines speaking voice. Draft:

```
This server returns inference cost/fit/throughput plans for LLM workloads.

Every numerical tool output includes a `trust_envelope` carrying:
- sources (each upstream that contributed a number)
- confidence_breakdown (per-domain: pricing, fit_check, throughput, model_architecture, gpu_specs, freshness)
- assumptions (what was held fixed)
- caveats (what we explicitly do NOT model)
- freshness (per-source last-updated timestamps)
- verify_links (URLs the user can audit upstream)

When relaying tool output to the user:
1. Always relay `sources`, the WORST domain in `confidence_breakdown`, and `caveats` verbatim. Do not paraphrase caveats; they are precise legal/factual disclaimers.
2. When `confidence_breakdown.throughput == 0.0`, the server is refusing to estimate that combination. Explain why (the `tps_estimate.refusal_reason` field tells you exactly).
3. When `fit_result.fits == True`, ALSO surface `fit_result.sufficiency_caveat` — fits=True is necessary but not sufficient.
4. When `pricing_type == "spot"`, mention that to the user. Spot pricing has preemption risk.
5. ALWAYS mention `availability_caveat` on CostCell results. We do not model rentability, only pricing.

Adapt explanation depth to the user's apparent experience. A first-time renter needs the caveats spelled out; a power user needs them present but compact. Either way: never strip the envelope, never hide a caveat, never round a confidence value upward.

This server is designed to be honest, not optimistic. When two numbers disagree, surface both. When a number is unknown, say so. The user's trust is the product.
```

---

## Vertical slices

1. **Slice A: FastMCP server skeleton** — `whatcanirun-mcp` starts, advertises capabilities, responds to `initialize` over stdio.
2. **Slice B: list_catalog** — TDD: returns all 5 catalog lists with non-zero entries (assumes M01, M03, M05 ran).
3. **Slice C: fit_check tool** — TDD: wraps M06 + builds TrustEnvelope for fit_check + model_architecture + gpu_specs domains.
4. **Slice D: find_cheapest_deployment** — TDD: returns ranked list, top row is cheapest, all rows have trust_envelope.
5. **Slice E: compare_deployment_modes** — TDD: returns both cloud_gpu_rental and hosted_api_token rows for the same op-point.
6. **Slice F: budget_to_plan** — TDD: `budget_to_plan(budget_usd=20, model_slug="qwen-3-coder-30b")` returns 3 rows ranked by `cost_per_m_output_usd`, each with populated `est_total_prompts` derived from default workload.
7. **Slice G: cost-cells://current resource** — TDD: resource is materialized as Parquet, contains all cells, `generated_at` populated, refreshed when CP cache invalidates.
8. **Slice H: cost-cells://provenance resource** — TDD: contains AA attribution, CP disclaimer, ADR list, "what we do NOT model" section.
9. **Slice I: /benchmark-on-budget prompt** — TDD: prompt template references the three tools in order and includes example arguments.
10. **Slice J: TrustEnvelope builders** — `src/whatcanirun/trust/builders.py` with one per tool. Confidence breakdown computed correctly per domain, `confidence = min(breakdown.values())` enforced.
11. **Slice K: Instructions string** — wired into FastMCP, exposed via standard protocol.

---

## Acceptance criteria

- [ ] `whatcanirun-mcp` starts, completes MCP handshake, advertises capabilities.
- [ ] All 5 tools callable; smoke-tested via fixtures (no live network in CI).
- [ ] Every tool response has a populated `trust_envelope` with all 6 domains present in `confidence_breakdown`.
- [ ] `confidence == min(confidence_breakdown.values())` enforced by a property test on TrustEnvelope construction.
- [ ] `budget_to_plan` golden path: `(budget_usd=20, model_slug="qwen-3-coder-30b")` returns ≥3 BudgetPlanRow, sorted ASC by `cost_per_m_output_usd`.
- [ ] `cost-cells://current` materializes Parquet with `generated_at` and per-source freshness.
- [ ] `cost-cells://provenance` contains AA attribution and ComputePrices disclaimer verbatim.
- [ ] `/benchmark-on-budget` chains the right tools when invoked from Claude Desktop.
- [ ] `FastMCP.instructions` string present, length-checked (rough sanity bound — not empty, not multi-thousand-word).
- [ ] Claude Desktop config in `docs/MCP.md` works end-to-end against stdio transport (manual test, recorded in commit message).

---

## Common pitfalls

- **TrustEnvelope construction divergence.** Every tool builds its own envelope. If one forgets a domain, the rollup is wrong. The builders module should expose a single function per tool to enforce uniformity.
- **Resources are not tools.** `cost-cells://current` is fetched as a resource (cached, addressable); not invoked as a tool. Don't accidentally implement it as a tool.
- **Instructions string overflow.** Some clients truncate at 4K chars. Keep the instructions tight; long-form details belong in `docs/TRUST.md`.
- **stdio transport buffering.** Test with `claude --mcp-server stdio whatcanirun-mcp` in the sandbox before declaring victory.

---

## When done

Commit:
> `M09: MCP server with 5 tools, 2 resources, 1 prompt, trust-contract instructions`

Mark M09 ✓ in `INDEX.md`. Continue with M10 (parallel) or M11.
