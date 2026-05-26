# M08 — Cost Cells Join Layer

**Status:** ⬜ Not started
**Effort:** 3h (5h realistic)
**Dependencies:** M01, M02, M03, M06, M07
**Unblocks:** M09

> Read [`SHARED.md`](SHARED.md) first. **ADR-014 is the load-bearing decision:** Python for tool business logic, DuckDB ONLY for resource materialization.

---

## Goal

`query_cost_cells(filters) -> list[CostCell]` for tool calls — implemented in plain Python over in-memory caches. Plus a separate `render_cost_cells_resource() -> bytes` for the `cost-cells://current` MCP resource — implemented with DuckDB.

This split is enforced by a grep test: no SQL strings in the tool call path.

---

## Scope

### CostCell schema

```python
class CostCell(BaseModel):
    # Identifiers
    gpu_slug: str | None             # null for hosted_api_token (no GPU)
    provider_slug: str
    model_slug: str
    quant_slug: str | None           # null for hosted_api_token (provider's choice, not disclosed)
    tp_size: int | None              # null for hosted_api_token (tensor parallelism is provider-internal)
    batch_size: int
    context_length: int
    deployment_mode: Literal["cloud_gpu_rental", "hosted_api_token"]

    # Pricing
    hourly_usd: float | None                                # null for hosted_api_token
    pricing_type: Literal["on_demand", "spot"] | None       # from CP; null for hosted_api_token
    price_per_m_input_usd: float | None
    price_per_m_output_usd: float | None

    # Throughput + fit
    decode_tps: float | None
    tps_estimate: TpsEstimate                               # M07
    fit_result: FitResult | None                            # M06; null for hosted_api_token
    cost_per_m_output_usd_self_hosted: float | None         # derived

    # Availability — we model PRICING, not RENTABILITY
    availability_modeled: bool = False
    availability_caveat: str = (
        "Price source does not guarantee current rentable capacity. Spot pricing "
        "is also subject to preemption and minimum-commitment terms not modeled here."
    )

    trust_envelope: TrustEnvelope                           # M09 builds this
```

### Public surface (`src/whatcanirun/plan/cost_cells.py`)

```python
def query_cost_cells(
    pricing_data: ComputePricesCache,
    gpu_catalog: list[Gpu],
    model_catalog: list[Model],
    bench_cells: list[BenchmarkCell],
    aa_data: list[AaModelRow] | None,
    filters: CostCellFilters,
) -> list[CostCell]:
    """Pure function. Pass caches in, get cells out. No I/O, no DB."""


def render_cost_cells_resource(
    pricing_data: ComputePricesCache,
    gpu_catalog: list[Gpu],
    model_catalog: list[Model],
    bench_cells: list[BenchmarkCell],
    aa_data: list[AaModelRow] | None,
) -> bytes:
    """Materialize ALL current cost cells as Parquet bytes for cost-cells://current resource."""


@dataclass
class CostCellFilters:
    model_slug: str | None = None
    gpu_slug: str | None = None
    provider_slug: str | None = None
    quant_slug: str | None = None
    batch_size: int | None = None
    context_length: int | None = None
    deployment_mode: Literal["cloud_gpu_rental", "hosted_api_token"] | None = None
    only_fits: bool = False
    workload_profile_slug: str | None = None
```

### Derived field math

```python
# Per-million-token self-hosted cost for cloud_gpu_rental rows
# (only computed when fits=True and decode_tps is non-None)
cost_per_m_output_usd_self_hosted = (hourly_usd / 3600) * (1_000_000 / decode_tps)
```

For `hosted_api_token` rows, this field stays None — the hosted price is already per-token.

---

## Architecture: Python-first, DuckDB-second

### Tool call path (Python only)

```python
def query_cost_cells(...) -> list[CostCell]:
    # 1. Apply filters as list comprehensions
    candidate_prices = [p for p in pricing_data.gpu_prices
                        if (filters.gpu_slug is None or p.gpu_slug == filters.gpu_slug)
                        and (filters.provider_slug is None or p.provider_slug == filters.provider_slug)
                        and ...]

    # 2. For each (price, model, quant) combination:
    cells = []
    for price in candidate_prices:
        gpu = gpu_by_slug[price.gpu_slug]
        for model in models_to_consider:
            for quant in quants_to_consider:
                fit = compute_fit(model, gpu, quant, tp_size, batch, ctx)  # M06
                if filters.only_fits and not fit.fits:
                    continue
                tps = estimate_tps(model, gpu, quant, batch, ctx,
                                   bench_cells, aa_data)  # M07
                cells.append(_build_cost_cell(...))

    return cells
```

No SQL anywhere in this path. Total memory footprint is small (~hundreds of rows). List comprehensions are faster than SQL roundtrips at v1 scale and easier to debug.

### Resource render path (DuckDB)

```python
def render_cost_cells_resource(...) -> bytes:
    """ONLY function in this module that uses DuckDB."""
    import duckdb
    con = duckdb.connect(":memory:")
    # Register the in-memory caches as DuckDB views
    con.register("gpu_prices", _to_arrow(pricing_data.gpu_prices))
    con.register("gpu_catalog", _to_arrow(gpu_catalog))
    con.register("model_catalog", _to_arrow(model_catalog))
    con.register("bench_cells", _to_arrow(bench_cells))

    # The one materialization query
    result = con.sql("""
        SELECT
            gp.gpu_slug, gp.provider_slug, gc.vram_gb,
            ...
        FROM gpu_prices gp
        JOIN gpu_catalog gc ON gp.gpu_slug = gc.slug
        LEFT JOIN bench_cells bc ON bc.gpu_slug = gp.gpu_slug AND ...
    """)
    return result.to_parquet()  # bytes
```

---

## Out of scope

- `on_prem_*` and `reserved_cloud` deployment modes — v2 work.
- Multi-region pricing — v2 work (CP exposes it via repeat rows; we'd need provider_region as a separate dimension).
- Real-time availability checks — by design, see availability_modeled flag.

---

## Vertical slices

1. **Slice A: CostCell schema** — TDD: full Pydantic instantiation passes; missing required field rejected.
2. **Slice B: query_cost_cells with single filter** — TDD: filter by `gpu_slug="h100"` returns only H100 rows.
3. **Slice C: only_fits=True** — TDD: with one fitting + one non-fitting row, returns only the fitting.
4. **Slice D: cost_per_m_output_usd_self_hosted math** — TDD: hourly $5, tps 100 → cost = $5/3600 × 1e6/100 = $13.89.
5. **Slice E: hosted_api_token mode** — TDD: LLM API rows produce `deployment_mode="hosted_api_token"` with `hourly_usd=None`, `price_per_m_*` populated.
6. **Slice F: pricing_type=spot surfaces** — TDD: a spot-priced row produces `CostCell(pricing_type="spot")` with availability_modeled=False.
7. **Slice G: render_cost_cells_resource produces Parquet** — TDD: returns bytes, readable as `pq.read_table()`, has all expected columns.
8. **Slice H: Grep test** — `tests/test_no_sql_in_business_logic.py` greps `src/whatcanirun/plan/cost_cells.py::query_cost_cells` and asserts no `con.sql` or `con.execute` patterns appear.

---

## Acceptance criteria

- [ ] `query_cost_cells(model_slug="qwen-3-coder-30b", gpu_slug="h100")` returns rows for every (quant, provider) where the model fits.
- [ ] Core query is pure Python — grep test confirms no SQL in tool call paths.
- [ ] DuckDB invoked ONLY by `render_cost_cells_resource()`.
- [ ] Every returned CostCell has `availability_modeled=False` and populated `availability_caveat`.
- [ ] Spot-priced rows have `pricing_type="spot"`; on-demand rows have `pricing_type="on_demand"`.
- [ ] hosted_api_token rows have `fit_result=None`, `hourly_usd=None`, `pricing_type=None`.
- [ ] All cells have populated `trust_envelope` with `confidence_breakdown` covering all 6 domains (built in M09).
- [ ] `cost_per_m_output_usd_self_hosted` math verified against worked example in tests.

---

## Common pitfalls

- **The temptation to "just use SQL".** When the join gets gnarly, you'll want to write SQL. Resist. SQL belongs in `render_cost_cells_resource()` only. The grep test exists to catch this drift.
- **TPS confidence vs CostCell confidence_breakdown.** TpsEstimate.confidence is the input to the `throughput` domain in confidence_breakdown — M09 wires it. Don't try to bake it in here.
- **hosted_api_token doesn't need fit_check.** A hosted API call doesn't load weights into your VRAM. Set fit_result=None and skip the math.

---

## When done

Commit:
> `M08: cost cells join layer (Python tool path, DuckDB resource path)`

Mark M08 ✓ in `INDEX.md`. Continue with M09.
