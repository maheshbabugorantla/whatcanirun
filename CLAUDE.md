# CLAUDE.md — Workflow Rules for Claude Code on whatcanirun

This file is read by Claude Code agents (and your local Claude Code sessions) before any action. It sets discipline that protects against the most common solo-builder failure modes: scope creep, horizontal-instead-of-vertical implementation, and silently breaking the trust contract.

---

## What this project is

A self-hosted stdio MCP server that converts `(budget_usd, model)` queries into ranked, source-backed inference plans. Read [`spec/SHARED.md`](spec/SHARED.md) before doing anything substantive. Each milestone has its own spec at `spec/M{NN}-*.md`.

---

## Invariants — do not violate

1. **Trust envelope is mandatory.** Every numerical tool output must include a `TrustEnvelope` Pydantic object with `sources`, `confidence_breakdown`, `assumptions`, `caveats`, `freshness`, `verify_links`. If a number can't be wrapped, do not return it. See `spec/SHARED.md` § "Trust Contract".

2. **Raw + projection storage.** Every upstream API response is persisted to disk verbatim *before* parsing. Pydantic models are projections with `extra="ignore"`. Nested objects whose schema is undocumented (`evaluations`, `pricing`, `specs`, HF `config.json`) typed as `dict[str, Any]` or `dict[str, float | None]` — NEVER narrow-typed. ADR-015.

3. **TPS source enum is sacred.** `own_measured | public_benchmark_anchor | provider_anchor | bandwidth_heuristic_single_stream | requires_measurement`. v1 NEVER returns `own_measured` (we have no own benchmarks yet). Conflating `public_benchmark_anchor` with `own_measured` is dishonest and a trust-contract violation.

4. **fit_check is necessary, not sufficient.** `fits=True` does NOT mean the rental will work well — only that the model fits in VRAM. Every FitResult must populate `sufficiency_caveat`.

5. **No SQL in tool business logic.** DuckDB is reserved for `cost-cells://current` resource materialization. Tool call paths use plain Python list/dict joins over in-memory caches. ADR-014, enforced by grep test.

6. **No Django, no SQL DB.** v1 is FastMCP + Pydantic + httpx + DuckDB-on-files. If you find yourself reaching for Postgres, you're working on v2 prematurely. ADR-008.

7. **Stdio transport only in v1.** No remote HTTP, no auth, no Render. Going-live is v2's problem. ADR-007.

---

## TDD discipline (red → green → refactor, one cycle at a time)

This is the single biggest leverage point for solo work. Vertical slices through one behavior at a time.

**The cycle:**
1. Write ONE failing test that captures ONE behavior.
2. Run it. Confirm it fails for the right reason.
3. Write the MINIMUM code to make it pass.
4. Run all tests. Confirm green.
5. Refactor (only if the code is actually unclean). Re-run.
6. Commit. Move to next behavior.

**What this rules out:**
- Writing five tests, then five implementations. You will lose your place. Do one at a time.
- "Let me just sketch the whole thing first." No. Test → impl → test → impl.
- Refactoring while red. Get green first, refactor second.

**Skill reference:** `tdd` from mattpocock. Read `.claude/skills/tdd/SKILL.md` before any milestone with TDD cycles in its acceptance criteria (M02, M03, M04, M06, M07).

---

## Domain glossary — use these terms exactly

In code, tests, commit messages, PR titles, issue bodies. Consistency reduces cognitive load.

- **Cost cell** — `(gpu, provider, model, quant, deployment_mode, batch, ctx) → (hourly_usd, decode_tps, cost_per_m_output_usd, trust_envelope)`. Atomic output unit.
- **Trust envelope** — Pydantic model carrying source provenance, calibrated confidence by domain, assumptions, caveats, freshness, audit links.
- **Confidence domain** — One of: `pricing`, `fit_check`, `throughput`, `model_architecture`, `gpu_specs`, `freshness`. Top-level `confidence` is `min(confidence_breakdown.values())`.
- **Deployment mode** — `cloud_gpu_rental`, `hosted_api_token`. (v2 adds `on_prem` with `tco_treatment` subfield.)
- **Op-point** — `(batch_size, context_length)` tuple.
- **Fit check** — Pure-math VRAM verdict. Returns FitResult with `weight_gb`, `kv_cache_gb`, `framework_overhead_gb`, `headroom_gb`, `blocking_reasons`, `sufficiency_caveat`. Never just a bool.
- **TPS source** — `own_measured | public_benchmark_anchor | provider_anchor | bandwidth_heuristic_single_stream | requires_measurement`.
- **Workload profile** — `(avg_input_tokens, avg_output_tokens)` seed. v1 ships 3.
- **Plan** — Ranked list of cost cells for a budget, with hours_available, est_total_prompts, est_wallclock_minutes, all under one trust envelope.

---

## Don'ts (these are the failure modes)

- ❌ **Don't reach for Django.** v1 stays minimal. If you think you need an ORM, you don't — DuckDB queries Parquet/JSON directly.
- ❌ **Don't fuzzy-match slugs.** Curated mapping in YAML only. Verified live: `llama-3-1-405b` substring-matches `hermes-4-llama-3-1-405b` which is a Nous Research fine-tune. Wrong answer, silently.
- ❌ **Don't narrow-type evolving schemas.** AA's `evaluations` had 16 fields when docs showed 10. They'll add more. `dict[str, float | None]` always.
- ❌ **Don't scale TPS linearly with batch size.** Verified ~6× wrong at batch=128. v1 heuristic is single-stream only; batch>1 returns `requires_measurement`.
- ❌ **Don't strip the trust envelope** in tool responses. The LLM client decides depth of explanation; the server never decides what to hide.
- ❌ **Don't add OAuth in v1.** Stdio has no auth. ADR-007.
- ❌ **Don't commit secrets.** `COMPUTEPRICES_API_KEY`, `AA_API_KEY`, `HF_TOKEN` go in `.env` (gitignored).

---

## Sandbox

Use the Docker sandbox for any code execution Claude Code does on your behalf:

```bash
docker compose -f compose.claude.yml run --rm claude-code
```

Inside the sandbox you have:
- Read/write access to the repo (mounted at `/workspace`)
- Read access to `~/.claude/skills/` (mattpocock skills)
- API key env vars passed through from your host (`.env`)
- No outbound network except to allowed domains (defined in `compose.claude.yml`)

The sandbox prevents accidental `rm -rf` outside the workspace, accidental commits to your global git config, and accidental network exfiltration of the upstream API keys.

---

## How to use the skills

`.claude/skills-lock.json` pins the mattpocock skill versions. The relevant skills for this project:

- `to-prd` — converts a milestone spec file into a `ready-for-agent` issue body
- `to-issues` — breaks a milestone into vertical-slice sub-issues
- `tdd` — enforces red-green-refactor discipline
- `triage` — categorizes problems before fixing them
- `diagnose` — root-cause analysis for failing tests
- `grill-with-docs` — pushback discipline (challenge assumptions before acting)
- `prototype` — quick proof-of-concept before full implementation
- `zoom-out` — strategic checkpoint when stuck
- `write-a-skill` — for adding new skills as patterns emerge

Standard workflow: `/to-prd spec/M00-bootstrap.md` → issue published → `/to-issues #<num>` → vertical slices → `/tdd` per slice → commit per cycle.

---

## When you're stuck

In order of preference:

1. **Run `/zoom-out`.** Strategic checkpoint. Are you working on the right thing?
2. **Re-read the milestone spec.** Did you drift from acceptance criteria?
3. **Re-read the relevant section of `spec/SHARED.md`.** Did you violate an ADR?
4. **Run `/triage`.** Categorize the blocker before guessing at a fix.
5. **Stop. Commit what you have. Sleep on it.** Solo project, no deadline — quality beats velocity.

---

## Last reminder

The trust contract is the product. Every shortcut that compromises honesty in tool output destroys the only thing that differentiates this project from a one-weekend GPU price comparison site. When in doubt: surface the caveat, lower the confidence, expose the assumption.

---

## Compact Instructions

This file auto-reloads when Claude Code compacts the conversation. Items below must survive compaction — they are the operational invariants the rest of this file expands on, plus pointers to specs that do NOT auto-load.

**Trust contract (non-negotiable):**
- Every numerical tool output carries a `TrustEnvelope` with `sources`, `confidence_breakdown`, `assumptions`, `caveats`, `freshness`, `verify_links`. No exceptions.
- `confidence = min(confidence_breakdown.values())` — weakest-link rule, never an average.
- `workload_assumption` domain appears ONLY on responses that synthesize a derived count from a workload profile; omit the key otherwise.

**Locked ADRs to honor:**
- **ADR-007:** v1 transport is stdio only — no remote HTTP, no auth.
- **ADR-008:** v1 stack is FastMCP + Pydantic + httpx + DuckDB-on-files — no Django, no SQL DB.
- **ADR-010:** TPS heuristic is single-stream only (batch=1); batch>1 returns `requires_measurement` unless a measured benchmark cell exists.
- **ADR-013:** When ComputePrices is unreachable, serve last-good local snapshot — never fail tool calls outright.
- **ADR-014:** Cost-cells tool path is plain-Python list joins; DuckDB is reserved for `cost-cells://current` resource materialization. Enforced by an AST grep test.
- **ADR-015:** Raw + projection storage. Upstream payloads stored verbatim; nested evolving objects typed as `dict[str, Any]` or `dict[str, float | None]`, never narrow-typed.

**TPS source enum (locked, never confuse):**
`own_measured | public_benchmark_anchor | provider_anchor | bandwidth_heuristic_single_stream | requires_measurement`. v1 never returns `own_measured` — we have no own benchmarks yet.

**Spec files that do NOT auto-load** (read them before substantive work in their area):
- [`spec/SHARED.md`](spec/SHARED.md) — § Trust Contract, § Calibration (per-domain confidence values), § Staleness policy (freshness decay curve)
- [`spec/M{NN}-*.md`](spec/INDEX.md) — current milestone acceptance criteria + vertical slice list

**Commit hygiene (per `story-commit` skill):**
- Invoke the `story-commit` skill via the Skill tool before any `git commit -m`.
- Never include `Co-Authored-By` trailers unless the user explicitly asks.
- Never `git add -A` / `git add .` — stage specific files by name.
- Never amend, never force-push, never skip pre-commit hooks.

**Memory rules to consult** at `~/.claude/projects/-workspace/memory/MEMORY.md`:
- `feedback-merge-authorization` — never merge a PR without explicit per-PR "ship it" from the user.
- `feedback-review-before-push` and `feedback-review-during-copilot-wait` — run `/review` + `/security-review` before every push and after every fix commit.
- `feedback-branching` — milestone work goes on a dedicated branch; never on main.
