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

from fastmcp import FastMCP

from whatcanirun import __version__

# The four-Kibibyte ceiling some MCP clients impose on
# `instructions` is the practical reason this prose is so dense.
# Long-form rationale and worked examples belong in `docs/TRUST.md`
# (M11); the string below is the contract the LLM client reads
# on every `initialize` handshake.
INSTRUCTIONS = """\
This server returns inference cost/fit/throughput plans for LLM workloads.

Every numerical tool output includes a `trust_envelope` carrying:
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


def main() -> None:
    """`uvx whatcanirun-mcp` entry point. Runs the FastMCP stdio
    transport by default — the calling client (Claude Desktop,
    Cursor, etc.) drives the loop via the spawned process's
    stdin/stdout. `show_banner=False` keeps stderr clean so an
    MCP client doesn't see startup noise interleaved with JSON-RPC
    frames (some clients are strict about non-protocol output)."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
