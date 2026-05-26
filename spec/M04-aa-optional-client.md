# M04 — Artificial Analysis Optional Client

**Status:** ⬜ Not started
**Effort:** 4h (6h realistic with slug investigation)
**Dependencies:** M00
**Unblocks:** M07 (provider_anchor Tier 2 in tps_estimator)
**Parallel-safe:** Can run alongside M02, M03, M06

> Read [`SHARED.md`](SHARED.md) first. ADR-003, ADR-015 are load-bearing.

---

## Goal

Optional enrichment from Artificial Analysis. When `AA_API_KEY` is set, ingest the 524-model free-tier response and surface per-model TPS aggregates as Tier-2 anchors in `tps_estimator`. When `AA_API_KEY` is unset, the rest of the system works unchanged — no calls, no caveats, no trust envelope entries naming AA.

---

## Verification status (locked, 25 May 2026)

Live-tested with a real key:
- **524 models** in the free-tier response
- **14/16** tracked open-weight models found by curated slug mapping
- `median_output_tokens_per_second` **confirmed populated** for open-weight rows (e.g., `gpt-oss-120b-low`: 361.84 tok/s)
- Llama-3.3-70B **notably absent** under that exact slug (investigate during this milestone)
- Reasoning models have an **effort-level dimension**: `gpt-oss-120b-low`, `-medium`, `-high` are three distinct rows
- `evaluations` payload has **16+ fields** vs the 10 in docs — including `aime_25`, `lcr`, `terminalbench_hard`, `tau2`, `ifbench`, `hle`

---

## Scope

### Public surface (`src/whatcanirun/pricing/artificial_analysis.py`)

```python
class AaDisabled(Exception):
    """Raised when AA_API_KEY is unset and an AA-only operation is requested."""


class ArtificialAnalysisClient:
    def __init__(self, cache_dir: Path, api_key: str | None = None):
        self.enabled = bool(api_key)

    async def get_models(self) -> list[AaModelRow]:
        """Raises AaDisabled if no key. Otherwise returns the projection."""

    async def get_raw_response(self) -> dict[str, Any]:
        """Full upstream payload. For trust envelope provenance + debugging."""
```

### Pydantic projection (ADR-015 — every field except identifiers is permissive)

```python
class AaModelRow(BaseModel):
    model_config = ConfigDict(extra="ignore")  # AA adds fields per Intelligence Index revision

    # Identifiers (stable)
    id: str                            # AA UUID — stable; primary join key
    slug: str                          # AA slug — may change
    name: str
    model_creator: dict[str, str]      # {id, name, slug}
    release_date: date | None          # undocumented but present on every row

    # Reasoning effort dimension — reasoning models have multiple rows per base model
    # Derived from slug suffix in projection logic; None for non-reasoning models
    reasoning_effort: Literal["low", "medium", "high"] | None

    # Throughput / latency — projected, used by tps_estimator
    median_output_tokens_per_second: float | None
    median_time_to_first_token_seconds: float | None
    median_time_to_first_answer_token: float | None  # post-reasoning latency for thinking models

    # Pricing — undocumented sub-keys exist (cache, batch, tiered)
    pricing: dict[str, float | None]

    # Evaluations — schema explicitly evolving. 16+ fields verified.
    # NEVER narrow-typed. AA Intelligence Index v5/v6 will add more.
    evaluations: dict[str, float | None]

    # Raw payload — full upstream row, verbatim
    raw: dict[str, Any]
```

### `seeds/aa_slug_mapping.yaml` (the curated mapping — CRITICAL)

```yaml
# WHY this is curated and not fuzzy: verified live, fuzzy matching maps
# `llama-3-1-405b` to `hermes-4-llama-3-1-405b` (a Nous fine-tune, NOT vanilla Llama).
# Every mapping must be explicit.

- cp_slug: gpt-oss-120b          # ComputePrices slug (our canonical)
  aa_slugs:                       # AA returns multiple rows for reasoning models
    - aa_slug: gpt-oss-120b-low
      reasoning_effort: low
    - aa_slug: gpt-oss-120b-medium
      reasoning_effort: medium
    - aa_slug: gpt-oss-120b-high
      reasoning_effort: high

- cp_slug: gpt-oss-20b
  aa_slugs:
    - aa_slug: gpt-oss-20b-low
      reasoning_effort: low
    - aa_slug: gpt-oss-20b-medium
      reasoning_effort: medium
    - aa_slug: gpt-oss-20b-high
      reasoning_effort: high

- cp_slug: deepseek-v3
  aa_slugs:
    - aa_slug: deepseek-v3
      reasoning_effort: null

- cp_slug: deepseek-r1
  aa_slugs:
    - aa_slug: deepseek-r1
      reasoning_effort: null

- cp_slug: llama-3-3-70b
  aa_slugs: []                    # AA does NOT track vanilla Llama-3.3-70B; Tier 2 falls through to heuristic
  investigation_note: "Live verification (25 May 2026): no match for 'llama-3-3-70b' or 'Llama 3.3 70B' across 524 rows. Investigate during M04 — may be under a provider-specific qualifier."

- cp_slug: mistral-large
  aa_slugs:
    - aa_slug: mistral-large
      reasoning_effort: null

# ... ~25 more rows. Open the live response with the AA verification script
# from earlier and grep for each CP slug; fill in real AA slugs.
```

### Cache layout

```
~/.cache/whatcanirun/artificial_analysis/
├── models.latest.json              # full raw response, verbatim
└── models.snapshots/
    ├── 2026-05-25T13-00-00Z.json.gz
    └── ...                          # 30-day rolling, pruned
```

### Refresh cadence

Every 6 hours. AA refreshes ~8×/day; 6h cache means we're never more than 3hr stale, well inside their 1k/day budget at 4 calls/day.

### tps_estimator integration (Tier 2)

When `tps_estimator` queries `(model_slug, batch_size=1, reasoning_effort=None|low|med|high)`:

1. Look up `aa_slug_mapping[model_slug]`
2. If found AND matches effort level: return `median_output_tokens_per_second` with:
   - `source="provider_anchor"`
   - `confidence=0.7`
   - `anchor_detail="AA-observed serving aggregate for {aa_slug}: {tps} tok/s"`
3. Caveat populated in trust envelope: `"AA reports a serving aggregate across providers; specific GPU and batch are not modeled."`
4. For `batch_size > 1`: AA Tier 2 does NOT fire. Falls through to Tier 4 (`requires_measurement`).

### Attribution (AA ToS requires)

Every `TrustEnvelope.sources` entry that uses AA data must populate:
```python
license_attribution=(
    "Includes data from Artificial Analysis (https://artificialanalysis.ai/), "
    "used under their free-tier API terms with attribution."
)
```

`cost-cells://provenance` resource (M09) also names AA prominently.

---

## Out of scope

- Per-provider TPS breakdowns (AA Pro feature, opaque pricing).
- AA's Pro/Enterprise API — defer until usage data justifies the cost.
- Auto-detecting new AA model slugs as they're added — manual YAML updates between releases.

---

## Vertical slices

1. **Slice A: AaModelRow with flexible nested objects** — TDD: a payload with new `evaluations` sub-keys is preserved in `raw` and queryable via `evaluations` dict.
2. **Slice B: HTTP client with `respx`** — failing test: `await client.get_models()` returns 524 parsed rows with stubbed upstream.
3. **Slice C: AaDisabled when no key** — failing test: `ArtificialAnalysisClient(api_key=None).get_models()` raises `AaDisabled`.
4. **Slice D: 6h cache + snapshot** — failing test: second call within 6h hits cache.
5. **Slice E: aa_slug_mapping YAML loader** — failing test: a mapping with one reasoning model produces three (cp_slug, aa_slug, effort) tuples.
6. **Slice F: Investigate Llama-3.3-70B mystery** — use the cached snapshot, grep for `llama-3-3` and `llama 3.3` variants. Document findings in `investigation_note` field of the mapping row.
7. **Slice G: Schema-evolution test** — fixture payload with a new `evaluations` sub-key (`gdpval_aa: 0.5`) is preserved.
8. **Slice H: Fallback when AA upstream fails** — TDD: 500 from AA returns cached snapshot if available, else logs warning and returns empty list (does NOT raise to caller).

---

## Acceptance criteria

- [ ] With `AA_API_KEY` unset: no calls made, `trust_envelope.sources` never names AA, all tests pass.
- [ ] With `AA_API_KEY` set: rows ingested, cached 6 hours, raw response persisted.
- [ ] If AA returns 401/429/500: falls back gracefully without failing parent tool call.
- [ ] **Schema-evolution test passes** — ingesting payload with new `evaluations` sub-key (simulating Intelligence Index v5) succeeds; field preserved in `evaluations` dict.
- [ ] Slug mapping covers all 30 tracked CP slugs; entries with `aa_slugs: []` documented (Llama-3.3-70B etc.).
- [ ] Reasoning-effort variants distinct: querying `gpt-oss-120b` with `reasoning_effort="high"` returns the `-high` row, not `-low`.
- [ ] AA attribution string present in every TrustEnvelope source entry that uses AA data.
- [ ] Llama-3.3-70B investigation documented (either: found under unexpected slug X, OR confirmed absent from free tier).

---

## Common pitfalls

- **Don't fuzzy-match.** Substring `llama-3-1-405b` matches `hermes-4-llama-3-1-405b` (Nous fine-tune). Use exact slug lookup only.
- **`evaluations` keys evolve.** Don't narrow-type with Literal — `dict[str, float | None]` always.
- **Reasoning effort is in the slug suffix.** Extract it in projection (regex on suffix) but also let YAML override explicitly when ambiguous.
- **Free tier is per-MODEL aggregate.** Not per-provider. Don't promise users provider-specific TPS from AA free tier.
- **Cache key the response by date, not query.** The endpoint takes no parameters; the cache key is the day.

---

## When done

Commit message:
> `M04: Artificial Analysis optional client + curated slug mapping + reasoning-effort dimension`

Mark M04 as ✓ in `spec/INDEX.md`. Move to M07 (depends on M04 + M06) or continue parallel work on M05/M10.
