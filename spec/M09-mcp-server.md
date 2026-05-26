# M09 — MCP Server (6 tools, 2 resources, 1 prompt)

**Status:** ⬜ Not started
**Effort:** 10h (12h realistic with prompt + instructions polish)
**Dependencies:** M01–M08
**Unblocks:** M11 (golden-path test), M12 (release)

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

A FastMCP server exposing the public product surface. Stdio transport. Six tools, two resources, one prompt. The `FastMCP.instructions` string is the single most important piece of prose in the whole project — it's what makes the LLM client speak in trust-contract-respecting voice without further training.

---

## Public surface

### Tools

1. **`list_catalog()`** → `{gpus, models, quantizations, workload_profiles, providers}`.
   One-call dropdown helper for clients building UIs.

2. **`find_cheapest_deployment(model_slug, quant_slug?, batch_size=1, context_length=4096, region?, top_n=10)`** → `list[CostCell]` (the normal path) **OR** `UnknownModelResponse` (Case 3 of [Unknown model handling](#unknown-model-handling)).
   The basic price-comparison tool. No budget; just "what's cheapest for this op-point?"

3. **`compare_deployment_modes(model_slug, gpu_slug, quant_slug, batch_size, context_length, workload_profile_slug)`** → `DeploymentComparison` (the normal path) **OR** `UnknownModelResponse` (Case 3, AND Case 2 — see "Tool-by-tool Case 2 behavior" below).
   Side-by-side of `cloud_gpu_rental` vs `hosted_api_token` for this op-point, with the inference-engineering-book §7.4.2 break-even framing.

4. **`fit_check(model_slug, gpu_slug, quant_slug, tp_size, batch_size, context_length)`** → `FitResult` with trust envelope (the normal path) **OR** `UnknownModelResponse` (Case 3, AND Case 2 — see "Tool-by-tool Case 2 behavior" below).
   Standalone wrapper over M06. Always populates `sufficiency_caveat`.

5. **`budget_to_plan(budget_usd, model_slug, workload_profile_slug?, quant_slug?, top_n=3)`** → `list[BudgetPlanRow]` (the normal path) **OR** `UnknownModelResponse` (Case 3 of [Unknown model handling](#unknown-model-handling)) **OR** `WorkloadElicitationResponse` (when `workload_profile_slug` is omitted — see [Workload assumption handling](#workload-assumption-handling)).
   **The headline tool.** Each row:
   ```python
   class BudgetPlanRow(BaseModel):
       cost_cell: CostCell
       hours_available: float | None          # budget_usd / hourly_usd; null for hosted_api_token
       est_total_prompts: int                 # grounded in the active workload profile
       est_total_output_tokens: int           # est_total_prompts × workload.avg_output_tokens
       est_wallclock_minutes: float | None    # null when throughput is requires_measurement
       cost_per_m_output_usd: float
       trust_envelope: TrustEnvelope          # `confidence_breakdown["workload_assumption"]`
                                              # populated; `assumptions["workload_profile"]`
                                              # names the profile this row is conditioned on.
   ```
   Every `est_total_prompts` / `est_total_output_tokens` figure is conditioned on a workload profile — the trust envelope names which one explicitly via `confidence_breakdown["workload_assumption"]` and `assumptions["workload_profile"]`. See [Workload assumption handling](#workload-assumption-handling) for why this can never be silently defaulted.


6. **`resolve_model(model_slug, hf_repo_id)`** → `ResolveModelResult`.
   Persists the `(model_slug, hf_repo_id)` mapping to `~/.config/whatcanirun/user_models.yaml` and triggers `HfModelSync.sync_model(hf_repo_id)`. Used by MCP clients after they receive an `UnknownModelResponse` and elicit the `hf_repo_id` from the user.

   ```python
   class ResolveModelResult(BaseModel):
       model_slug: str
       hf_repo_id: str
       status: Literal["resolved", "sync_failed", "not_found_on_hf"]
       hf_revision_sha: str | None          # populated when status == "resolved"
       error_detail: str | None             # populated when status != "resolved"
   ```

   Keeping `resolve_model` as its own tool (rather than threading `hf_repo_id_hint` through every model-taking tool) keeps the other tool signatures stable and lets MCP clients schema-validate them without a union type per arg.

   `ResolveModelResult` deliberately does NOT carry a `trust_envelope`. Per `spec/SHARED.md`, the trust envelope wraps **numerical** tool outputs; `resolve_model` returns a status + diagnostic, no numbers. The follow-up call to `budget_to_plan` / `find_cheapest_deployment` / etc. is where the trust envelope appears, and `freshness["huggingface"]` on that envelope reflects the just-completed sync (`hf_revision_sha` matches the value returned here on the resolved path).

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
- confidence_breakdown (per-domain: pricing, fit_check, throughput, model_architecture, gpu_specs, workload_assumption, freshness — `workload_assumption` appears only on responses that synthesize derived counts from a workload profile, e.g. `BudgetPlanRow.est_total_prompts`; omit it entirely when no workload was assumed)
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
6. When `confidence_breakdown.workload_assumption` is present, ALWAYS surface the assumed workload profile from `assumptions["workload_profile"]` (e.g. "this estimate assumes ~500 input + ~200 output tokens per prompt; if your prompts differ, the count scales accordingly"). A `workload_assumption` value < 0.5 means the server fell back to a default profile rather than the user picking — call that out and offer the elicitation alternatives.

Adapt explanation depth to the user's apparent experience. A first-time renter needs the caveats spelled out; a power user needs them present but compact. Either way: never strip the envelope, never hide a caveat, never round a confidence value upward.

This server is designed to be honest, not optimistic. When two numbers disagree, surface both. When a number is unknown, say so. The user's trust is the product.
```

---

## Unknown model handling

When a user calls a tool for a model whatcanirun doesn't fully model, dispatch depends on which cache state applies. Affects `budget_to_plan`, `find_cheapest_deployment`, `compare_deployment_modes`, and `fit_check`. **All three cases are handled at the M09 layer — the underlying caches stay simple.**

### Case 1 — In the merged tracked-models set, config not yet synced locally

Lazy-sync transparently via `HfModelSync.sync_model(repo_id)` (M03). Tool then proceeds as normal. `trust_envelope.freshness["huggingface"]` reflects the just-completed sync (per-source timestamp, not per-domain — `freshness` is keyed by upstream, not by `confidence_breakdown` domain).

No user-visible difference from a warm-cache request other than ~1s latency on first call.

### Case 2 — Known to ComputePrices (catalog + prices), NOT in our tracked-models set

CP knows the model's hosted-API pricing; we don't have architecture data. Return **partial** `CostCell` rows with `deployment_mode="hosted_api_token"`. The complete field map (consult M08's CostCell schema for the union — `gpu_slug`, `quant_slug`, `tp_size` are nullable, see M08 for the rationale):

Identifiers:
- `gpu_slug = None` (no GPU — hosted API).
- `quant_slug = None` (provider's choice, not disclosed by CP).
- `tp_size = None` (tensor parallelism is provider-internal).
- `provider_slug` populated from the `LlmPriceRow.provider_slug` of the chosen quote.
- `model_slug` is the requested slug (echoed back).
- `batch_size`, `context_length` echoed from the tool's request (still relevant for `est_total_prompts` math even though they don't drive hosted-API price).
- `deployment_mode = "hosted_api_token"`.

Pricing:
- `hourly_usd = None` (no GPU rental).
- `pricing_type = None` (M08's `pricing_type` is the GPU `on_demand|spot` enum; LLM hosted-API has its own `standard|batch` enum which travels in `trust_envelope.assumptions["llm_pricing_tier"]` instead, to avoid the enum collision).
- `price_per_m_input_usd`, `price_per_m_output_usd` populated from CP's `/api/v1/llm-prices` (the model's existence is confirmed via `/api/v1/llm-models`; the dollar values come from the prices endpoint — M02's `LlmPriceRow` projection).

Throughput + fit:
- `decode_tps = None`.
- `tps_estimate.source = "requires_measurement"`, `tps_estimate.confidence = 0.0`, `tps_estimate.refusal_reason` cites the missing architecture data.
- `fit_result = None`.
- `cost_per_m_output_usd_self_hosted = None`.

Availability + trust:
- `availability_modeled = False`; default `availability_caveat` applies.
- `trust_envelope.confidence_breakdown["model_architecture"] = 0.0`.
- `trust_envelope.freshness["computeprices"]` populated; `freshness["huggingface"]` absent because no HF data was consumed.
- `trust_envelope.caveats` includes verbatim:
  > "Architecture data not available for this model — only hosted-API pricing is shown. Fit-check and self-hosted throughput are not estimated. To enable full analysis, add an entry to your local `~/.config/whatcanirun/user_models.yaml` with the model's Hugging Face `repo_id`."

The partial answer is honest — the user gets actionable pricing AND an explicit, named gap they can close themselves.

#### Tool-by-tool Case 2 behavior

Case 2's "partial CostCell" path is honest for tools that quote token economics (where hosted-API pricing IS the answer the user wants). It is dishonest for tools that fundamentally require architecture data — returning a degenerate `DeploymentComparison` or `FitResult` would invite the LLM client to relay something the server can't actually defend.

| Tool | Case 2 behavior | Why |
|---|---|---|
| `find_cheapest_deployment` | partial `list[CostCell]` per the field map above (hosted_api_token rows only) | Token-only prices are usefully comparable across providers; cloud_gpu_rental rows are unavailable but explicitly absent |
| `budget_to_plan` | partial `list[BudgetPlanRow]` wrapping the hosted_api_token CostCells | Same logic — token economics drives the budget math |
| `compare_deployment_modes` | **`UnknownModelResponse`** (Case 2 collapses to Case 3 for this tool) | Its whole purpose is to compare cloud_gpu_rental vs hosted_api_token; without architecture data, the cloud side is impossible. A `DeploymentComparison` with `cloud_gpu_rental=None` would obscure the failure mode |
| `fit_check` | **`UnknownModelResponse`** (Case 2 collapses to Case 3 for this tool) | Fit-checking requires architecture by definition |

This is why `compare_deployment_modes` and `fit_check`'s signature lines (Public surface §3 and §4) cite both Case 2 and Case 3 as `UnknownModelResponse` triggers, while `find_cheapest_deployment` and `budget_to_plan` cite only Case 3.

### Case 3 — In NEITHER CP nor our tracked-models set (genuinely unknown)

Interactive. The tool returns a structured `UnknownModelResponse` **instead of the normal tool result payload** (which varies — `list[BudgetPlanRow]` for `budget_to_plan`, `list[CostCell]` for `find_cheapest_deployment`, `DeploymentComparison` for `compare_deployment_modes`, `FitResult` for `fit_check`):

```python
class UnknownModelResponse(BaseModel):
    requested_model_slug: str
    status: Literal["unknown_model"] = "unknown_model"
    elicit_field: Literal["hf_repo_id"] = "hf_repo_id"
    elicit_prompt: str = (
        "I don't have this model in my catalog yet. If you can share the "
        "Hugging Face repo_id (e.g. `meta-llama/Llama-3.3-70B-Instruct`), "
        "I'll fetch its config and add it for this and future requests. "
        "If the model isn't on a public Hugging Face repo, I won't be able "
        "to estimate fit or throughput for it."
    )
    suggested_followups: list[str] = Field(
        default_factory=lambda: [
            "list_catalog (to see what models are already supported)",
            "budget_to_plan with a publicly tracked model_slug",
        ]
    )
```

The MCP client surfaces `elicit_prompt` to the user. Two outcomes:

1. **User supplies a repo_id** → client invokes the dedicated `resolve_model(model_slug, hf_repo_id)` tool (see Public surface §6). On success, the client re-invokes the original tool with the same `model_slug` and gets a normal result (Case 1 path).

2. **User can't supply a repo_id** (private model, doesn't know, etc.) → MCP client stops here. There is no second tool call — `UnknownModelResponse` is an elicitation, not a refusal that travels back through the tool. The client relays the `elicit_prompt`'s framing to the user and explains that without a Hugging Face `repo_id` whatcanirun has no path to estimating fit or throughput; if the model is hosted elsewhere or is private, the user can file an issue at <https://github.com/maheshbabugorantla/whatcanirun/issues>.

   Note that `UnknownModelResponse` deliberately does NOT carry a `trust_envelope`. Per the trust contract in `spec/SHARED.md`, only **numerical** tool responses must carry one (it wraps numbers with sources/confidence/caveats; an elicitation has no numbers to wrap). The "named gap" the trust contract requires is carried by `elicit_prompt` text and the absence of any numerical result — both are honest signals to the client that whatcanirun cannot help.

## Workload assumption handling

`budget_to_plan` synthesizes prompt-count and wallclock estimates by conditioning on a workload profile (`avg_input_tokens`, `avg_output_tokens` from M05). The number changes a lot with the assumed token counts — `chat_assistant` at 500/200 vs `batch_eval` at 2000/100 yields wildly different `est_total_prompts` for the same `budget_usd`. **The spec treats a missing workload as the same kind of problem as a missing model: the server elicits it rather than silently guessing.**

### When the client omits `workload_profile_slug`

`budget_to_plan` returns a `WorkloadElicitationResponse` instead of `list[BudgetPlanRow]`:

```python
class WorkloadElicitationResponse(BaseModel):
    requested_model_slug: str
    status: Literal["workload_required"] = "workload_required"
    elicit_field: Literal["workload_profile_slug"] = "workload_profile_slug"
    elicit_prompt: str = (
        "To estimate prompt counts for your budget, I need to know what kind "
        "of workload these prompts represent. Pick one:\n"
        "- code_completion: short prompts (≈100 in, ≈50 out)\n"
        "- chat_assistant: medium prompts (≈500 in, ≈200 out)\n"
        "- batch_eval:    long prompts (≈2000 in, ≈100 out)\n"
        "If none of those fit, ask me for `find_cheapest_deployment` instead — "
        "it returns $/M figures so you can do the math against your own "
        "token distribution."
    )
    available_profiles: list[str] = Field(
        default_factory=lambda: ["code_completion", "chat_assistant", "batch_eval"]
    )
    suggested_followups: list[str] = Field(
        default_factory=lambda: [
            "budget_to_plan with workload_profile_slug='chat_assistant' for a starting estimate",
            "find_cheapest_deployment (returns $/M figures, no prompt-count synthesis)",
        ]
    )
```

The MCP client surfaces `elicit_prompt` to the user. On reply:

1. **User picks one of the three profile slugs** → client re-invokes `budget_to_plan(..., workload_profile_slug=<picked>)`. The retry returns normal `BudgetPlanRow`s with `confidence_breakdown["workload_assumption"] = 0.95` and `assumptions["workload_profile"] = <picked>` recording exactly what was assumed.

2. **User wants prices without a profile** → client switches to `find_cheapest_deployment` (which returns CostCells with `$/M_input`, `$/M_output`, no prompt-count synthesis, and no `workload_assumption` in the trust envelope's breakdown). The user does the multiplication themselves.

`WorkloadElicitationResponse` does NOT carry a `trust_envelope` — same logic as `UnknownModelResponse` and `ResolveModelResult`. It's an elicitation, not a numerical output; there are no numbers to wrap.

### Why a silent default isn't acceptable

The earlier draft of M09 had `workload_profile_slug?` as optional with an implicit default. That hides hearsay: the user gets `est_total_prompts = 22000` without knowing the number was computed against a `chat_assistant`-shaped prompt distribution they may or may not match. With the `workload_assumption` confidence domain now landed in `spec/SHARED.md`, a silent default would set `confidence_breakdown["workload_assumption"] = 0.2` and drag the top-level confidence to 0.2 by min-rollup — at which point the FastMCP instructions string would force the LLM client to relay the low score anyway. Eliciting up-front is the same answer, expressed in the API instead of as a runtime quality signal.

### Tool-by-tool workload requirement

| Tool | Needs workload? | Behavior when missing |
|---|---|---|
| `find_cheapest_deployment` | no — returns CostCells with $/M figures | n/a |
| `compare_deployment_modes` | yes — signature already requires `workload_profile_slug` | hard schema error from the MCP layer; no elicitation needed |
| `fit_check` | no — purely architectural | n/a |
| `budget_to_plan` | yes — synthesizes prompt counts | `WorkloadElicitationResponse` |

### The merged tracked-models set

Throughout this section "the tracked-models set" means the union of:

- `seeds/tracked_models.yaml` (project-controlled, committed to the repo)
- `~/.config/whatcanirun/user_models.yaml` (per-user, runtime-accumulated, NOT committed)

The merging is M03's responsibility — see `spec/M03-hf-model-sync.md` § "User-extension file" for the loader contract. M09's `resolve_model` tool is what appends to `user_models.yaml`; M03's loader is what reads both files at sync time and surfaces a single combined list to M09's dispatcher.

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
12. **Slice L: Unknown-model dispatcher + `resolve_model` tool** — `find_cheapest_deployment` / `budget_to_plan` / `compare_deployment_modes` / `fit_check` route by case:
    - in tracked-models set, not cached → lazy-sync via M03 (Case 1)
    - in CP catalog + prices, not in tracked-models set → partial CostCell with `deployment_mode="hosted_api_token"`, `model_architecture=0.0` confidence, LLM `pricing_tier` carried in `assumptions` (Case 2)
    - in neither → return `UnknownModelResponse` (Case 3, first call)
    - `resolve_model(model_slug, hf_repo_id)` → persists to `~/.config/whatcanirun/user_models.yaml` + invokes M03 sync; subsequent calls go through Case 1.
    Tested with stubbed HF responses; no live network in CI.
13. **Slice M: Workload-assumption dispatcher** — `budget_to_plan` returns `WorkloadElicitationResponse` when `workload_profile_slug` is omitted; populates `confidence_breakdown["workload_assumption"] = 0.95` and `assumptions["workload_profile"]` on the retry; trust envelope construction adds the new domain (SHARED.md update). Tests: missing slug → elicitation; supplied slug → normal `BudgetPlanRow`s with the domain populated; `find_cheapest_deployment` omits the key from `confidence_breakdown` (no derived prompt count).

---

## Acceptance criteria

- [ ] `whatcanirun-mcp` starts, completes MCP handshake, advertises capabilities.
- [ ] All 6 tools callable; smoke-tested via fixtures (no live network in CI).
- [ ] Every **numerical** tool response (and every CostCell / BudgetPlanRow / FitResult / DeploymentComparison contained in one) has a populated `trust_envelope` covering every applicable domain in `confidence_breakdown` (`workload_assumption` is required only on responses that synthesize a derived count from a workload — i.e. `BudgetPlanRow`s). Non-numerical responses (`UnknownModelResponse`, `ResolveModelResult`, `WorkloadElicitationResponse`) do not carry one — per `spec/SHARED.md`, the trust envelope wraps numbers; an elicitation or a status/diagnostic response has none to wrap.
- [ ] `budget_to_plan` called without `workload_profile_slug` returns a `WorkloadElicitationResponse`, never a `BudgetPlanRow` with a silent default. The retry with `workload_profile_slug` set populates `confidence_breakdown["workload_assumption"] = 0.95` and `assumptions["workload_profile"]` on every row.
- [ ] `confidence == min(confidence_breakdown.values())` enforced by a property test on TrustEnvelope construction.
- [ ] Unknown-model dispatcher covers all three cases (lazy-sync, partial-answer, interactive elicitation) with named caveats. `resolve_model(model_slug, hf_repo_id)` persists user-supplied pairs to `~/.config/whatcanirun/user_models.yaml`, not `seeds/tracked_models.yaml`.
- [ ] M03's `sync_all_tracked()` reads from BOTH `seeds/tracked_models.yaml` AND `~/.config/whatcanirun/user_models.yaml` (when present) — the merged-loader contract is part of M03's surface, not M09's; see `spec/M03-hf-model-sync.md` § "User-extension file".
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
> `M09: MCP server with 6 tools, 2 resources, 1 prompt, trust-contract instructions`

Mark M09 ✓ in `INDEX.md`. Continue with M10 (parallel) or M11.
