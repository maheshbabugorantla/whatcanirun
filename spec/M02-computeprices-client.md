# M02 — ComputePrices Client

**Status:** ⬜ Not started
**Effort:** 4h
**Dependencies:** M00
**Unblocks:** M08 (cost-cells join), M09 (MCP tool surface)

> Read [`SHARED.md`](SHARED.md) first. ADR-001, ADR-013, ADR-015 are load-bearing.

---

## Goal

Async client wrapping the four ComputePrices endpoints we use, with on-disk raw caching, snapshot fallback when upstream is unreachable, and the Raw + Projection storage pattern. Every response is persisted verbatim before parsing, so future field additions never lose data.

---

## Endpoints to wrap

| Endpoint | Cache TTL | Why |
|---|---|---|
| `GET /api/v1/gpu-prices` | 1h | Per-(provider, GPU, pricing_type) prices. The headline data. |
| `GET /api/v1/llm-prices` | 1h | Hosted LLM API per-1M-token pricing. |
| `GET /api/v1/gpus` | 24h | GPU catalog with bandwidth + VRAM + architecture. Rarely changes. |
| `GET /api/v1/llm-models` | 24h | LLM model slug list with context window + modalities. Rarely changes. |

Auth: `Authorization: Bearer cp_live_...` from `COMPUTEPRICES_API_KEY` env var. Falls back to anonymous 60/hr public tier with a logged warning.

---

## Scope

### Public surface (`src/whatcanirun/pricing/computeprices.py`)

```python
class ComputePricesClient:
    """Async client with on-disk caching and snapshot fallback (ADR-013)."""

    def __init__(self, cache_dir: Path, api_key: str | None = None): ...

    async def get_gpu_prices(self) -> list[GpuPriceRow]: ...
    async def get_llm_prices(self) -> list[LlmPriceRow]: ...
    async def get_gpu_catalog(self) -> list[GpuCatalogRow]: ...
    async def get_llm_catalog(self) -> list[LlmCatalogRow]: ...

    # Raw access — for trust envelope provenance + debugging + fields not yet projected
    async def get_raw_response(self, endpoint: str) -> dict[str, Any]: ...
```

### Pydantic projections (per ADR-015 — `extra="ignore"`, nested objects flexible)

```python
class GpuCatalogRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    slug: str
    name: str
    manufacturer: Literal["NVIDIA", "AMD", "Intel"]
    architecture: str | None
    vram_gb: int
    release_date: date | None
    # specs is undocumented at field level — schema gains fields without notice. Keep flexible.
    specs: dict[str, float | int | str | None] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)  # full upstream row, verbatim


class GpuPriceRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    provider: str
    provider_slug: str
    gpu: str
    gpu_slug: str
    vram_gb: int
    gpu_count: int
    price_per_hour_usd: float
    pricing_type: Literal["on_demand", "spot"]
    commitment_months: int | None
    currency: str
    source_url: str
    last_updated: datetime
    raw: dict[str, Any] = Field(default_factory=dict)


class LlmCatalogRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    slug: str
    name: str
    creator: str
    family: str | None
    context_window: int | None
    modalities: list[str]
    knowledge_cutoff: date | None
    raw: dict[str, Any] = Field(default_factory=dict)


class LlmPriceRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    provider: str
    provider_slug: str
    model_slug: str
    price_1m_input_usd: float | None
    price_1m_output_usd: float | None
    pricing_type: Literal["standard", "batch"]
    last_updated: datetime
    raw: dict[str, Any] = Field(default_factory=dict)
```

### Cache layout

```
~/.cache/whatcanirun/computeprices/
├── gpu-prices.latest.json          # raw upstream response, current
├── gpu-prices.snapshots/
│   ├── 2026-05-25T13-00-00Z.json.gz
│   ├── 2026-05-25T14-00-00Z.json.gz
│   └── ...                          # 30-day rolling history
├── llm-prices.latest.json
├── llm-prices.snapshots/...
├── gpus.latest.json
├── gpus.snapshots/...
├── llm-models.latest.json
└── llm-models.snapshots/...
```

### Failure modes (ADR-013)

| Upstream | Action |
|---|---|
| 200 OK | Parse, project, persist raw + projection. Update `latest.json` + snapshot. |
| 429 / 500 / 503 | `tenacity` retry 3× with exponential backoff (1s, 2s, 4s). |
| All retries fail | Load `latest.json` from cache. Return with `freshness.computeprices` reflecting actual file mtime. No exception bubbles to caller. |
| Cache empty AND upstream down | Raise `ComputePricesUnavailable` only here. Caller responsible for trust envelope adjustment. |

### Pricing caveats (must propagate)

Every CostCell that uses ComputePrices data MUST include the following verbatim in `trust_envelope.caveats` (per ComputePrices' own disclaimer):

```
Prices collected from public sources and provider APIs, refreshed regularly,
provided for informational purposes only. Prices may exclude:
- CPU, disk, and network costs
- Regional variations
- Minimum commitments or reservation discounts
- Negotiated enterprise pricing
Always confirm pricing on the provider's official page before purchasing.
```

---

## Vertical slices

1. **Slice A: Pydantic projections** — write each model with one TDD cycle each (extra field tolerated, missing required field rejected, raw preserved).
2. **Slice B: HTTP client with `respx` mock** — failing test: `await client.get_gpu_prices()` returns a parsed list. Stub the upstream with a fixture. Green.
3. **Slice C: On-disk cache** — failing test: second call within TTL returns from disk in <5ms. Implement TTL check. Green.
4. **Slice D: Snapshot persistence** — failing test: after one successful fetch, a timestamped `.json.gz` exists. Implement. Green.
5. **Slice E: Fallback on upstream failure** — failing test: with respx returning 500, client returns latest snapshot. Implement. Green.
6. **Slice F: Pruning** — failing test: snapshots older than 30 days are removed. Implement. Green.
7. **Slice G: Schema-evolution test (ADR-015)** — failing test: payload with an unknown extra field on `specs` succeeds, field preserved in `raw` and queryable. Already covered by Pydantic config but verify end-to-end. Marker: `@pytest.mark.schema_evolution`.

---

## Acceptance criteria

- [ ] Cache hit returns <5ms (measure with `time.perf_counter()`).
- [ ] Cache miss fetches, parses, writes both `latest.json` and a timestamped snapshot.
- [ ] All four endpoints work: `gpu-prices`, `llm-prices`, `gpus`, `llm-models`.
- [ ] Network failure returns latest snapshot with `freshness` populated; no exception.
- [ ] `respx` fixture suite covers: happy path, 429 retry-then-success, 500-then-fallback, malformed JSON.
- [ ] No live HTTP in CI (`COMPUTEPRICES_API_KEY=""` in CI env).
- [ ] Schema-evolution test (`@pytest.mark.schema_evolution`) passes — unknown field on `specs` preserved.
- [ ] Caveat list propagated: ComputePrices' disclaimer included verbatim in trust envelope caveats.
- [ ] `prune_snapshots(older_than=timedelta(days=30))` works and is tested.

---

## Operational note

**Before this milestone is fully shippable, email `api@computeprices.com`** requesting a free 5k/hr API key with the project name and use case. The anonymous 60/hr tier will hit rate limits during testing if you're not careful.

---

## Common pitfalls

- **httpx Async vs sync confusion.** Use `httpx.AsyncClient`. All four methods are `async def`.
- **JSON snapshot growth.** Compressed (`.gz`); pruning runs on every fetch.
- **Pricing type field.** ComputePrices uses `pricing_type` for both GPU rows (`on_demand`/`spot`) and LLM rows (`standard`/`batch`). Different enums; don't conflate.
- **Hourly TTL drift.** If your cache TTL is exactly 1h, two clients may both hit upstream simultaneously after a long idle. Add jitter (e.g., TTL ± 60s).

---

## When done

Commit message:
> `M02: ComputePrices async client + raw cache + snapshot fallback`

Mark M02 as ✓ in `spec/INDEX.md`. Move to M03.
