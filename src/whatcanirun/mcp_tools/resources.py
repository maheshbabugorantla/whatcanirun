"""M09 Slice G + H: cost-cells MCP resources.

Two resources land here:

- `cost-cells://current` — Parquet materialization of all current
  cost cells, rendered via M08's `render_cost_cells_resource`.
  Re-rendered when any contributing cache invalidates.

- `cost-cells://provenance` — JSON document declaring every data
  source, the locked ADRs, what we explicitly do NOT model, and
  license attributions. The single audit-trail document.

Both are FastMCP resources (not tools). They're addressable by
URI and may be cached on the client side. spec/M09 § Common
pitfalls #2 calls out the "resources are not tools" mistake —
this module is where that distinction is enforced.

When the contributing caches haven't been warmed yet (first call
after install, no network), the current resource degrades to an
empty parquet table with the documented schema rather than
failing the read — the empty table is still a valid resource
the client can render as "no rows yet".
"""

from __future__ import annotations

import json

from whatcanirun.plan.cost_cells import render_cost_cells_resource

# Per-source attribution + the audit-trail metadata served by the
# provenance resource. Keeping this as a module-level dict means
# every read returns the same bytes; if a future ADR-introduction
# updates the locked list, this dict gets edited and the change
# round-trips through the resource on next read.
_PROVENANCE = {
    "server": "whatcanirun",
    "spec": "https://github.com/maheshbabugorantla/whatcanirun/tree/main/spec",
    "sources": [
        {
            "name": "computeprices",
            "url": "https://www.computeprices.com",
            "attribution": (
                "Pricing data sourced from ComputePrices "
                "(https://www.computeprices.com). ComputePrices aggregates "
                "public GPU and LLM pricing pages; figures may lag actual "
                "provider pricing by up to the cached freshness window."
            ),
            "role": "GPU $/hr, LLM API $/M-token, GPU base catalog",
            "license": "ComputePrices Terms of Service",
        },
        {
            "name": "huggingface",
            "url": "https://huggingface.co",
            "attribution": (
                "Model architecture data sourced from Hugging Face Hub "
                "config.json + safetensors metadata at the published "
                "revision SHA."
            ),
            "role": "Model architecture (n_layers, n_kv_heads, head_dim, etc.)",
            "license": "Per-repo on Hugging Face; this server reads public configs only",
        },
        {
            "name": "artificial_analysis",
            "url": "https://artificialanalysis.ai",
            "attribution": (
                "When AA enrichment is enabled, provider-anchored throughput "
                "data sourced from Artificial Analysis "
                "(https://artificialanalysis.ai). AA is an optional "
                "enhancement; the server operates without it."
            ),
            "role": "Provider-anchored throughput (optional enrichment)",
            "license": "Artificial Analysis Free Tier terms",
        },
        {
            "name": "public_benchmark_anchor",
            "url": "https://github.com/maheshbabugorantla/whatcanirun/blob/main/spec/M10-benchmark-seeds.md",
            "attribution": (
                "Public benchmark anchors curated from blog posts, vendor "
                "release notes, and academic papers. Each cell carries its "
                "source URL in the trust envelope's `verify_links`."
            ),
            "role": "Throughput benchmarks (public seeds in v1; measured in v2)",
            "license": "Citation in trust envelope verify_links",
        },
        {
            "name": "datasheet_yaml",
            "url": "https://github.com/maheshbabugorantla/whatcanirun/tree/main/seeds",
            "attribution": (
                "GPU supplement facts curated from manufacturer datasheets "
                "and Inference Engineering §5.1.1."
            ),
            "role": "GPU fp8/fp4 tflops, form factor, kernel support flags",
            "license": "MIT (project-controlled)",
        },
    ],
    "adrs": [
        {"id": "ADR-001", "decision": "ComputePrices canonical for GPU $/hr and LLM API pricing"},
        {"id": "ADR-002", "decision": "Hugging Face canonical for model architecture"},
        {"id": "ADR-003", "decision": "Artificial Analysis is optional enrichment"},
        {"id": "ADR-004", "decision": "Trust envelope on every numerical response"},
        {
            "id": "ADR-005",
            "decision": "GPU supplement fields in YAML (not in ComputePrices schema)",
        },
        {"id": "ADR-007", "decision": "v1 transport: stdio only"},
        {"id": "ADR-008", "decision": "v1 stack: FastMCP + Pydantic + httpx + DuckDB-on-files"},
        {
            "id": "ADR-010",
            "decision": "TPS heuristic restricted to single-stream (batch=1)",
        },
        {
            "id": "ADR-013",
            "decision": "Snapshot fallback on CP unreachable; never fail tool calls outright",
        },
        {
            "id": "ADR-014",
            "decision": "Python list joins for tool calls; DuckDB only for resource materialization",
        },
        {"id": "ADR-015", "decision": "Raw + Projection storage pattern"},
    ],
    "what_we_do_not_model": [
        "Provider rentability or stock availability (only pricing).",
        "Real-time latency / time-to-first-token figures.",
        "Kernel-level acceptance of the chosen quantization on the chosen GPU.",
        "Tensor-parallel communication efficiency across heterogeneous links.",
        "Provider runtime compatibility (CUDA / driver / framework versions).",
        "Batch>1 throughput in v1 (linear scaling is ~6x wrong; v1 returns requires_measurement).",
        "On-prem TCO (v2 work; gated on usage signal).",
        "Reserved-instance or spot preemption probability.",
        "Hosted-API rate limits or per-key quota caps.",
    ],
    "fall_back_behavior": {
        "computeprices_unreachable": (
            "Server serves last-good local snapshot (30-day rolling window) "
            "with `freshness.computeprices` reflecting staleness."
        ),
        "huggingface_unreachable": (
            "Cached projection by revision SHA; if not cached, the model "
            "routes to UnknownModelResponse asking the user to supply repo_id."
        ),
        "artificial_analysis_unreachable_or_disabled": (
            "Server returns throughput estimates from `bandwidth_heuristic` "
            "(batch=1 only) or `public_benchmark_anchor` (M10 seeds); never "
            "synthesizes a number that requires AA when AA is absent."
        ),
    },
}


async def render_current_cost_cells() -> bytes:
    """Resource handler for `cost-cells://current`.

    Materializes all current cost cells as Parquet via M08's
    `render_cost_cells_resource`. Loads every contributing cache
    (CP prices/catalog, HF model catalog, seed quantizations +
    bench cells) through `load_runtime_deps` so warm caches
    actually populate the parquet payload — the resource matches
    its spec name (`current`, not `empty`).

    Per ADR-013, the resource MUST NOT fail the read on cache
    failure. `load_runtime_deps` already catches
    `ComputePricesUnavailable` and returns empty lists for the
    affected upstream; this handler additionally catches any
    other exception (e.g. HF disk corruption, an unexpected
    httpx error class that escaped M02's wrapper) and degrades
    to the same empty-but-well-formed parquet table the cold-
    cache path emits. The client always gets a valid parquet
    response.
    """
    from whatcanirun.mcp_tools.deps import RuntimeDeps, load_runtime_deps

    try:
        deps = await load_runtime_deps()
    except Exception:
        # Any escape from load_runtime_deps (HF cache corruption,
        # an httpx error class M02 didn't wrap, etc.) collapses
        # to an empty render rather than a failed resource read.
        deps = RuntimeDeps()
    return render_cost_cells_resource(
        gpu_prices=deps.gpu_prices,
        llm_prices=deps.llm_prices,
        gpu_catalog=deps.gpu_catalog,
        model_catalog=deps.model_catalog,
        quantizations=deps.quantizations,
        bench_cells=deps.bench_cells,
        aa_observations=None,
    )


def render_provenance_document() -> str:
    """Resource handler for `cost-cells://provenance`.

    Returns the audit-trail JSON document declaring every upstream
    source, attribution string, locked ADR, and "what we do NOT
    model" entry. Indented for human auditability — the document
    IS the read by a human deciding whether to trust the server.
    """
    return json.dumps(_PROVENANCE, indent=2)
