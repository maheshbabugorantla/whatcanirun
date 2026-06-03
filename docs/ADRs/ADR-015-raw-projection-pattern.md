# ADR-015 — Raw + projection storage pattern for upstream APIs

**Status:** Locked
**Date:** 2026-05 (v2.1 lock-in)

## Decision

Every upstream API response is persisted to disk verbatim *before*
parsing. Pydantic models are projections of the raw response,
configured with `extra="ignore"` so unknown fields don't raise.
Nested objects whose schema is undocumented or evolving
(`evaluations`, `pricing`, `specs`, HF `config.json` family-specific
fields) are typed as `dict[str, Any]` or
`dict[str, float | None]` — **never** narrow-typed.

## Context

Upstream schemas change. Verified live during M02–M04 build-out:

- AA's `evaluations` nested object has 16+ fields where docs
  showed 10 (new ones include `aime_25`, `lcr`,
  `terminalbench_hard`, `tau2`, `ifbench`).
- ComputePrices adds sub-objects per release; `specs` is the
  current undocumented pocket.
- Hugging Face `config.json` varies per family — DeepSeek-MLA
  has `q_lora_rank` and `kv_lora_rank` that Llama models don't;
  Mixtral has `num_local_experts` that nobody else has.

A narrow-typed Pydantic model would break on every upstream
release. Worse: tightly typing nested objects whose schema is
unknown encourages dropping fields that the trust envelope might
later care about.

## Consequences

- Every upstream fetch writes the raw upstream bytes to disk
  *before* the projection runs. Each client owns its layout under
  the user cache directory (XDG `~/.cache/whatcanirun` by
  default); the pattern is "latest" + "snapshots" rather than a
  single fixed path. Current clients:
  - ComputePrices:
    `<cache_dir>/<endpoint>.latest.json` (raw bytes) plus rolling
    snapshots in `<cache_dir>/<endpoint>.snapshots/`
    (src/whatcanirun/pricing/computeprices.py).
  - Artificial Analysis:
    `<cache_dir>/artificial_analysis/models.latest.json` plus
    gzipped snapshots in
    `<cache_dir>/artificial_analysis/models.snapshots/<ISO-8601>.json.gz`
    (src/whatcanirun/pricing/artificial_analysis/client.py).
  - Hugging Face: slug-keyed cache under
    `<cache_dir>/huggingface/` —
    `{slug}.model.json` (projection),
    `{slug}.config.json` (raw config bytes),
    `{slug}.info.json` (raw info-endpoint bytes). One entry per
    slug; the cached file's `hf_repo_id` + `hf_revision_sha` are
    validated on read, and a SHA change triggers a re-sync that
    overwrites in place rather than retaining historical
    revisions.
  New clients should follow the same "raw bytes verbatim before
  any narrowing" rule; the exact directory layout is the
  client's call as long as the bytes survive on disk.
- Projection models inherit a base config with
  `extra="ignore"`, so a new upstream field is invisible to the
  projection but preserved in the raw layer for forensic
  inspection.
- Nested evolving fields are typed loosely. Reviewers should
  treat any attempt to narrow them as a regression — the M03
  HF-sync work has the explicit DeepSeek-MLA-was-a-surprise
  cautionary tale in commit history.
- This is the load-bearing pattern under ADR-013's snapshot
  fallback — the snapshot served is the raw bytes, parsed afresh
  on read.

## Alternatives considered

- **Narrow-typed Pydantic everywhere.** Breaks on every upstream
  schema change. Rejected with evidence.
- **No persistence; parse-and-discard.** Loses the audit trail and
  the snapshot fallback (ADR-013) is impossible.
- **Persist only the projection.** Loses fields the next milestone
  might need; defeats the point of having raw bytes.

## References

- ADR-013 (snapshot fallback reads the raw layer)
- [`../../CLAUDE.md`](../../CLAUDE.md) § Invariant 2 (this is one
  of the load-bearing invariants for trust contract honesty)
