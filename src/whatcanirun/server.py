"""MCP server entry point: FastMCP instance + stdio launcher.

`whatcanirun-mcp` (defined in pyproject.toml's `[project.scripts]`)
is the public CLI. It starts the FastMCP stdio loop, which
advertises whatcanirun's tools / resources / prompts to whichever
MCP client (Claude Desktop, Cursor, etc.) launched it.

Decorator-based registration (tools/resources/prompts) attaches
to the module-level `mcp` instance. Slices B-M of M09 register
against it; Slice A just stands it up with the trust-contract
`instructions` string the LLM client reads on `initialize`.

The instructions string is the single most important piece of
prose in the project — it's what makes the LLM client speak
trust-contract-respecting language without further training.
Per spec/M09 § "The FastMCP.instructions string" and the relay
rules block in spec/SHARED.md § "Trust contract".
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from fastmcp import FastMCP

from whatcanirun import __version__

# The four-Kibibyte ceiling some MCP clients impose on
# `instructions` is the practical reason this prose is so dense.
# Long-form rationale and worked examples belong in `docs/TRUST.md`
# (M11); the string below is the contract the LLM client reads
# on every `initialize` handshake.
INSTRUCTIONS = """\
This server returns inference cost/fit/throughput plans for LLM workloads.

Every numerical tool output carries a `trust_envelope`. For tools that return a single result Pydantic (e.g. `fit_check`, `compare_deployment_modes`), the envelope is a top-level `trust_envelope` field on the response. For tools that return lists (e.g. `find_cheapest_deployment` returns `list[CostCell]`; `budget_to_plan` returns `list[BudgetPlanRow]`), the envelope is nested PER ITEM — each list element has its own `trust_envelope` field; there is no top-level envelope for the list as a whole. When relaying a list result, walk every row's envelope, not just the first.

Envelope contents:
- sources (each upstream that contributed a number)
- confidence_breakdown (per-domain: pricing, fit_check, throughput, model_architecture, gpu_specs, workload_assumption, freshness — `workload_assumption` appears only on responses that synthesize derived counts from a workload profile, e.g. `BudgetPlanRow.est_total_prompts`; it is omitted entirely when no workload was assumed)
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
"""


mcp: FastMCP = FastMCP(
    name="whatcanirun",
    instructions=INSTRUCTIONS,
    version=__version__,
)


# --------------------------------------------------------------- Tool registry
# Decorator-based registration on the module-level `mcp` instance.
# Each slice adds its tool here; the implementations live in
# `whatcanirun/mcp_tools/` so this file stays a thin transport-layer
# shell that wires the public surface.

from whatcanirun.mcp_tools.budget_to_plan import budget_to_plan as _budget_to_plan  # noqa: E402
from whatcanirun.mcp_tools.catalog import list_catalog as _list_catalog  # noqa: E402
from whatcanirun.mcp_tools.compare_deployment import (  # noqa: E402
    compare_deployment_modes as _compare_deployment_modes,
)
from whatcanirun.mcp_tools.find_cheapest import (  # noqa: E402
    find_cheapest_deployment as _find_cheapest_deployment,
)
from whatcanirun.mcp_tools.fit_check import fit_check as _fit_check  # noqa: E402

mcp.tool(_list_catalog, name="list_catalog")
mcp.tool(_fit_check, name="fit_check")
mcp.tool(_find_cheapest_deployment, name="find_cheapest_deployment")
mcp.tool(_compare_deployment_modes, name="compare_deployment_modes")
mcp.tool(_budget_to_plan, name="budget_to_plan")

from whatcanirun.mcp_tools.dispatch import resolve_model as _resolve_model  # noqa: E402

mcp.tool(_resolve_model, name="resolve_model")


# --------------------------------------------------------------- Resources
# `cost-cells://current` + `cost-cells://provenance` — see
# `whatcanirun/mcp_tools/resources.py` for the handlers.

from whatcanirun.mcp_tools.resources import (  # noqa: E402
    render_current_cost_cells as _render_current_cost_cells,
)
from whatcanirun.mcp_tools.resources import (  # noqa: E402
    render_provenance_document as _render_provenance_document,
)

mcp.resource(
    "cost-cells://current",
    name="cost-cells-current",
    mime_type="application/vnd.apache.parquet",
)(_render_current_cost_cells)
mcp.resource(
    "cost-cells://provenance",
    name="cost-cells-provenance",
    mime_type="application/json",
)(_render_provenance_document)


# --------------------------------------------------------------- Prompts
# `/benchmark-on-budget` — see `whatcanirun/mcp_tools/prompt.py`.

from whatcanirun.mcp_tools.prompt import benchmark_on_budget as _benchmark_on_budget  # noqa: E402

mcp.prompt(_benchmark_on_budget, name="benchmark-on-budget")


async def _prefetch_impl() -> int:
    """Warm every on-disk cache the numerical tools read at call
    time, so the first MCP `tools/call` doesn't pay the cold-cache
    1-3s upstream-fetch latency.

    Two halves: HF first (catalog sync via `HfModelSync.sync_all_tracked`,
    which fetches per-tracked-model config.json + model info under
    `<cache_dir>/huggingface/`), then `load_runtime_deps()` which
    warms ComputePrices (`<cache_dir>/computeprices/`) and reads
    the HF cache the sync just populated.

    Per-source progress goes to stderr (stdout is reserved for the
    eventual stdio protocol path; keeping stderr the only chatter
    channel matches the rest of the server). Returns 0 on success
    even if individual rows or endpoints degraded — partial warmup
    is still useful and matches `load_runtime_deps`'s tolerate-and-
    continue contract. Returns non-zero only on a setup failure
    that prevents either step from running at all."""
    from whatcanirun.catalog.hf_sync import HfModelSync
    from whatcanirun.mcp_tools.deps import load_runtime_deps
    from whatcanirun.paths import SEEDS_DIR, USER_CACHE_DIR, USER_CONFIG_DIR

    seeds_dir: Path = SEEDS_DIR
    cache_dir: Path = USER_CACHE_DIR
    config_dir: Path = USER_CONFIG_DIR

    print(f"prefetch: HF sync_all_tracked → {cache_dir / 'huggingface'}", file=sys.stderr)
    user_yaml = config_dir / "user_models.yaml"
    hf = HfModelSync(cache_dir=cache_dir)
    synced = await hf.sync_all_tracked(
        tracked_yaml_path=seeds_dir / "tracked_models.yaml",
        user_yaml_path=user_yaml if user_yaml.exists() else None,
    )
    print(f"prefetch: HF synced {len(synced)} tracked model(s)", file=sys.stderr)

    print("prefetch: warming ComputePrices + AA caches", file=sys.stderr)
    deps = await load_runtime_deps(seeds_dir=seeds_dir, cache_dir=cache_dir, config_dir=config_dir)
    print(
        "prefetch: cache warm — "
        f"{len(deps.gpu_catalog)} gpus, "
        f"{len(deps.gpu_prices)} gpu-prices, "
        f"{len(deps.llm_catalog)} llm-catalog, "
        f"{len(deps.llm_prices)} llm-prices, "
        f"{len(deps.model_catalog)} hf-models",
        file=sys.stderr,
    )
    return 0


def _run_prefetch() -> int:
    """Synchronous wrapper around `_prefetch_impl` for the argparse
    handler. Separated so tests can monkeypatch this single
    callable without having to mock the async runner. Also creates
    the cache directory upfront (blocking I/O outside the async
    path keeps `_prefetch_impl` ruff-ASYNC240 clean and matches
    the load_runtime_deps convention of pre-creating per-source
    subdirs)."""
    from whatcanirun.paths import USER_CACHE_DIR

    try:
        USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"prefetch: cannot create cache dir {USER_CACHE_DIR}: {exc}",
            file=sys.stderr,
        )
        return 1
    return asyncio.run(_prefetch_impl())


def main(argv: list[str] | None = None) -> None:
    """`whatcanirun-mcp` entry point. Three modes:

    - No args → run the FastMCP stdio transport. This is the mode
      every MCP client uses (Claude Desktop, Cursor, etc. launch
      the binary as a subprocess and talk JSON-RPC over its
      stdin/stdout). `show_banner=False` keeps stderr clean so a
      client doesn't see startup noise interleaved with frames.
    - `--version` → print package version, exit 0. Clean-machine
      smoke test gate.
    - `prefetch` → run `_run_prefetch()` and exit with its return
      code. Warms CP + HF caches synchronously so the first
      tools/call doesn't pay the cold-cache latency."""
    parser = argparse.ArgumentParser(
        prog="whatcanirun-mcp",
        description="Self-hosted stdio MCP server for inference cost/fit/throughput plans.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "prefetch",
        help="Warm CP + HF caches synchronously (avoids cold-cache delay on first tool call).",
    )
    args = parser.parse_args(argv)

    if args.command == "prefetch":
        sys.exit(_run_prefetch())
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
