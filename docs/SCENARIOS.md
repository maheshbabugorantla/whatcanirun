# Validation scenarios for whatcanirun

Eight end-to-end scenarios you can walk through manually in a real MCP
client (Claude Desktop, Claude Code, Cursor, Cline) to validate that a
running install behaves as expected — not just that it responds, but
that it relays the trust contract honestly.

> These scenarios are **prescriptive**, not test-validated. They tell
> you what to ask, what tool path Claude should walk, and what to
> check in the response. The release-gate test (`pytest -m release`)
> verifies the server side of every contract; these scenarios verify
> the *client-side* behaviour the server's `INSTRUCTIONS` string is
> designed to elicit. If a scenario fails because the client
> paraphrased a caveat or hid a confidence value, that's a real bug
> worth filing — the server cannot make the LLM client behave; it can
> only give the client material to behave honestly with.

## How to use this doc

For each scenario:

1. Wire your client to the server per [`docs/MCP.md`](MCP.md).
2. Paste the **"Question to ask"** into the chat verbatim.
3. Observe which tool(s) Claude calls (most clients surface this).
4. Walk the **"What to verify"** checklist against the response Claude
   produces in chat — NOT against the raw tool output, which is the
   server's contract (already covered by the release-gate test).
5. If any checkbox fails, that's the bug — capture the chat
   transcript and the tool-call inspector output (Claude Code shows
   raw tool I/O on demand; Claude Desktop has a developer toggle).

The scenarios are ordered by complexity — start with #1, work down.

---

## Scenario 1 — Budget → plan (the headline)

**Question to ask:**

> "I have $50 to spend running Llama 3.3 70B for chat. What can I
> actually run?"

**Expected tool path:** `budget_to_plan(budget_usd=50.0,
model_slug="llama-3-3-70b", workload_profile_slug="chat_assistant")`.

If Claude calls `budget_to_plan` without a workload profile slug, the
server returns a `WorkloadElicitationResponse` and Claude should
follow up by asking you to pick one (Scenario 6 covers this branch).

**What to verify:**

- [ ] Ranked rows by `cost_per_m_output_usd` (cheapest first)
- [ ] Each row mentions `hours_available`, `est_total_prompts`, and
      `est_wallclock_minutes` — those are the answer to "what can I
      actually run?"
- [ ] Claude states the workload assumption verbatim, naming
      `chat_assistant`'s `(avg_input_tokens, avg_output_tokens)` shape
      from the `assumptions["workload_profile"]` field. Per the
      `INSTRUCTIONS` string rule 6, this is mandatory when
      `workload_assumption` appears in `confidence_breakdown`.
- [ ] Claude names the WORST domain in the confidence breakdown (most
      commonly `throughput` at 0.60 or 0.70 depending on AA coverage).
      If Claude says "highly confident" without naming the weak
      domain, it's relaying dishonestly.
- [ ] Claude mentions `availability_caveat` per `INSTRUCTIONS` rule 5
      — the server does not model rentability.

---

## Scenario 2 — Fit check (won't fit)

**Question to ask:**

> "Will Mixtral 8x22B fit on a single H100 80GB at fp16, 8k context?"

**Expected tool path:** `fit_check(model_slug="mixtral-8x22b",
gpu_slug="h100", quant_slug="fp16", tp_size=1, batch_size=1,
context_length=8192)`.

**What to verify:**

- [ ] `fit_result.fits = False` (Mixtral 8x22B at fp16 is ~280GB
      weight-only — doesn't fit in 80GB)
- [ ] `fit_result.blocking_reasons` non-empty and names the actual
      shortfall (weight + KV cache + overhead vs. VRAM available)
- [ ] Claude surfaces `sufficiency_caveat` even on a False verdict —
      it's MANDATORY per spec/M06's FitResult contract
- [ ] Claude suggests realistic alternatives (multi-GPU
      `tp_size>1`, quantization, larger SKU) — the response carries
      enough info for Claude to do this without inventing
- [ ] No `workload_assumption` domain in `confidence_breakdown` —
      `fit_check` is pure VRAM math, no workload synthesis (per the
      omit-when-not-synthesized rule)

---

## Scenario 3 — Find cheapest

**Question to ask:**

> "What's the cheapest provider hosting Qwen 2.5 72B right now?"

**Expected tool path:** `find_cheapest_deployment(model_slug=
"qwen-2-5-72b", top_n=10)`.

**What to verify:**

- [ ] Sorted ascending by `cost_per_m_output_usd`
- [ ] Each row carries its OWN `trust_envelope` (per-row contract;
      there is no top-level envelope for the list)
- [ ] Each row mentions `availability_caveat` per `INSTRUCTIONS` rule 5
- [ ] If any row has `pricing_type=spot`, Claude mentions preemption
      risk per `INSTRUCTIONS` rule 4
- [ ] Claude names the freshness of CP pricing — `freshness.
      computeprices` is on every envelope. If CP hasn't refreshed in
      6h+, the freshness confidence domain decays and that should
      surface as "prices may be stale" in Claude's relay

---

## Scenario 4 — Compare deployment modes (rental vs hosted)

**Question to ask:**

> "Should I rent an H100 or use a hosted API for Llama 3.3 70B at
> chat volumes?"

**Expected tool path:** `compare_deployment_modes(model_slug=
"llama-3-3-70b", gpu_slug="h100", quant_slug="fp16", batch_size=1,
context_length=4096, workload_profile_slug="chat_assistant")`.

**What to verify:**

- [ ] Both `rental_economics` (cloud_gpu_rental) and
      `hosted_economics` (hosted_api_token) populated
- [ ] `workload_assumption` PRESENT in `confidence_breakdown` —
      per-prompt cost is workload-derived
- [ ] Claude names the break-even crossover (the rental vs hosted
      decision flips at a workload volume threshold; the response
      contains the data to compute it)
- [ ] Claude surfaces both envelopes' WORST domains, not an average
- [ ] If hosted-API row is missing for `llama-3-3-70b` because no
      tracked provider hosts it, Claude says so — does not paper over

---

## Scenario 5 — Unknown-model elicitation (Case 1: model not tracked)

**Question to ask:**

> "Can I run NousResearch/Hermes-2-Pro-Mistral-7B on an A100?"

**Expected tool path:** `fit_check` is called with an unknown slug →
server returns `UnknownModelResponse` with elicitation prompt →
Claude asks you for the HF repo_id (if not already in the question) →
calls `resolve_model(model_slug="hermes-2-pro-mistral",
hf_repo_id="NousResearch/Hermes-2-Pro-Mistral-7B")` → retries
`fit_check` against the now-synced model.

**What to verify:**

- [ ] Claude correctly recognizes "Hermes-2-Pro-Mistral" isn't a
      tracked slug (don't pre-emptively try `mistral-7b` — Hermes-2
      is a fine-tune, not the base; substring matching would be
      wrong per CLAUDE.md § Don'ts)
- [ ] Claude prompts for the HF repo_id rather than inventing one
- [ ] After `resolve_model` succeeds, the retry `fit_check` returns
      a real `FitResult` with all the trust-envelope invariants
- [ ] The persisted slug → repo_id mapping lands at
      `~/.config/whatcanirun/user_models.yaml` (check on disk after
      the chat)

---

## Scenario 6 — Workload elicitation (Case 2: no workload profile)

**Question to ask:**

> "How many prompts can I run on $20 of Qwen 2.5 7B?"

The question deliberately omits a workload profile — that's what
triggers the elicitation. It also doesn't pin a GPU; `budget_to_plan`
ranks across GPUs and the point here is the workload branch, not the
fit branch.

**Expected tool path:** `budget_to_plan(budget_usd=20.0,
model_slug="qwen-2-5-7b", workload_profile_slug=None)` → server
returns `WorkloadElicitationResponse` with the 3 v1 profiles → Claude
shows the choices → user picks → retry `budget_to_plan` with the
chosen slug.

**What to verify:**

- [ ] Claude presents the 3 v1 workload profiles with their
      `(avg_input_tokens, avg_output_tokens)` shapes:
      `code_completion`, `chat_assistant`, `batch_eval`
- [ ] Claude does NOT pick a default silently — the elicitation flow
      is the trust-contract-correct path
- [ ] After the user picks, the retry succeeds and the response
      `confidence_breakdown.workload_assumption` value is >= 0.95
      (user picked) rather than 0.2 (silent fallback). If 0.2 fires,
      that's a bug — `budget_to_plan` is the elicitation surface
      precisely so the silent default is unreachable

---

## Scenario 7 — Provenance audit

**Question to ask:**

> "Show me the sources behind your throughput estimate."

**Expected tool path:** Claude reads the
`cost-cells://provenance` resource (NOT just relays prior
context).

**What to verify:**

- [ ] Resource is actually read (Claude Code shows resource reads in
      its tool inspector; in Claude Desktop the resource fetch is
      visible in the developer log)
- [ ] Returned JSON's `sources[]` array names ComputePrices, Hugging
      Face, AND Artificial Analysis. AA appears regardless of whether
      `AA_API_KEY` is set — the static document lists every potential
      contributor, and AA's `attribution` text explains it's an
      optional tier the server runs without
- [ ] Each `sources[]` entry has `name`, `url`, `attribution`,
      `role`, and `license` fields. The AA entry's `attribution`
      string is the load-bearing one — AA's free-tier API terms
      require it on every consumer-visible surface
- [ ] Each entry's `url` is a real, resolvable address Claude can
      point the user at for audit (per-row `verify_links` live on
      individual tool responses' trust envelopes — those are
      Scenarios 1-4's surface, not this static-document surface)

---

## Scenario 8 — Honest "I don't know" (batch > 1)

**Question to ask:**

> "What's the per-stream decode TPS for Llama 3.3 70B on an H100 at
> batch=32?"

**Expected tool path:** Any tool that estimates TPS at batch>1 → at
least one response row has `tps_source=requires_measurement`
(per ADR-010: the v1 heuristic is single-stream only).

**What to verify:**

- [ ] Claude does NOT invent a TPS number
- [ ] Claude surfaces `tps_estimate.refusal_reason` per
      `INSTRUCTIONS` rule 2 — names that the v1 heuristic doesn't
      extrapolate batch>1
- [ ] Claude offers batch=1 as the heuristic-supported alternative
- [ ] `confidence_breakdown.throughput == 0.0` — the weakest-link
      rule means the top-level `confidence` is also 0.0
- [ ] No fabricated number with a hedge — the server's contract is
      "refuse, don't bluff"; if Claude bluffs anyway, the
      `INSTRUCTIONS` string in `src/whatcanirun/server.py` may need
      a stronger rule

---

## What scenario failure means

If a scenario passes the server-side gate (release test) but fails in
chat because of how Claude relayed it:

- The `INSTRUCTIONS` string isn't strong enough for that client. Open
  an issue, include the chat transcript + the raw tool output, and
  consider tightening the relevant numbered rule in
  [`src/whatcanirun/server.py`](../src/whatcanirun/server.py).

If a scenario fails because the server returned the wrong shape:

- That's a server bug, not a relay bug. The release-gate test should
  have caught it — open a regression issue against
  `tests/release/test_stdio_install.py` so the gate covers the
  scenario going forward.

If a scenario fails because the tool path Claude took was wrong
(e.g. Claude called `find_cheapest_deployment` when you asked a
budget question that should have routed to `budget_to_plan`):

- The tool docstrings + `INSTRUCTIONS` string aren't disambiguating
  well enough. Same fix path — tighten the prose, file the diff,
  capture the chat transcript as the regression case.

---

## Out of scope for these scenarios

- **Multi-turn conversations** spanning >2 tool calls. The
  unknown-model / workload elicitation flows (Scenarios 5 + 6) are
  the only multi-turn cases the v1 server is designed for. Anything
  longer (chained "now compare to a different GPU" type questions)
  is the LLM client composing on its own — useful, but not part of
  the server's validation surface.
- **Performance benchmarking** of the server itself. Use
  `time uv run whatcanirun-mcp prefetch` for cold-cache cost;
  warm-cache tool-call latency is bounded by the FastMCP roundtrip
  (sub-second) per the release test's 13s-for-8-tests benchmark.
- **Adversarial input** (malformed slugs, path traversal in
  resolve_model, etc.). Those are server-side concerns covered by
  the existing boundary-validation tests in
  `tests/catalog/test_hf_sync.py` and friends — not user scenarios.

## See also

- [`docs/MCP.md`](MCP.md) — per-client install + config
- [`docs/TRUST.md`](TRUST.md) — the trust contract in detail (what
  every envelope field means)
- [`docs/PRD.md`](PRD.md) — product framing (who this is for, what
  problem it solves)
- [`tests/release/test_stdio_install.py`](../tests/release/test_stdio_install.py)
  — the mechanical server-side gate these scenarios layer on top of
