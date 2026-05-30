"""M09 Slice I: `/benchmark-on-budget` MCP prompt.

The guided workflow for first-time users who don't know which
catalog item maps to their idea. The prompt chains:

  list_catalog (if model missing) → fit_check (candidate GPUs) →
  budget_to_plan

with the supplied budget + optional model_slug woven into the
prose so the LLM client can render a concrete recipe rather
than asking the user to compose tool calls from scratch.

Keeping the prompt template here (not embedded inside the
FastMCP decorator) means the wording can be edited without
touching the server transport-shell, and unit-tested for the
tool-chain ordering + argument inclusion that spec/M09 § Prompts
requires.
"""

from __future__ import annotations


def benchmark_on_budget(budget_usd: float, model_slug: str | None = None) -> str:
    """Render the `/benchmark-on-budget` prompt body.

    The output is the message the LLM client uses to drive the
    workflow. The tool-chain order — `list_catalog` →
    `fit_check` → `budget_to_plan` — is preserved verbatim in
    the prose so a regression that re-orders the steps fails the
    Slice I tests."""
    if model_slug is None:
        return (
            f"The user has ${budget_usd} to spend on LLM inference for a "
            f"workload they haven't pinned down yet. Walk them through:\n\n"
            f"1. Call `list_catalog` to show the supported models, GPUs, "
            f"quantizations, and workload profiles. Ask the user which "
            f"model they want to deploy (give them 3-4 options across the "
            f"size/capability spectrum).\n\n"
            f"2. Once the user picks a `model_slug`, run `fit_check` for "
            f"each candidate GPU (start with `h100sxm`, `a100sxm`, and "
            f"`l40s`) at the chosen quant. Show which GPUs actually fit "
            f"the model and which ones don't (with the blocking_reasons).\n\n"
            f"3. With a fit candidate identified, call `budget_to_plan` "
            f"with `budget_usd={budget_usd}`, the model_slug the user "
            f"picked, and `workload_profile_slug='chat_assistant'` (or "
            f"whichever profile matches their use case — the catalog "
            f"step exposes them). Surface the top 3 rows ranked by "
            f"`cost_per_m_output_usd`.\n\n"
            f"At every step, relay the trust_envelope's `sources`, the "
            f"WORST `confidence_breakdown` domain, and any `caveats` "
            f"verbatim. Don't hide the workload assumption — the user "
            f"should know what shape of traffic the est_total_prompts is "
            f"conditioned on."
        )
    return (
        f"The user has ${budget_usd} to spend running `{model_slug}` for "
        f"inference. Walk them through:\n\n"
        f"1. (Optional) Call `list_catalog` if you need to confirm the "
        f"available quantizations or workload profiles for {model_slug}.\n\n"
        f"2. Run `fit_check` for the {model_slug} model on a few "
        f"candidate GPUs (e.g. `h100sxm`, `a100sxm`, `l40s`) at the "
        f"user's preferred quant (default to `fp8` if unstated). Surface "
        f"which GPUs fit and which don't (with blocking_reasons + "
        f"sufficiency_caveat for the ones that do).\n\n"
        f"3. Call `budget_to_plan(budget_usd={budget_usd}, "
        f"model_slug='{model_slug}', workload_profile_slug='chat_assistant')` "
        f"to translate the budget into hours, total prompts, and "
        f"wallclock minutes. Show the top 3 rows ranked by "
        f"`cost_per_m_output_usd`.\n\n"
        f"At every step, relay the trust_envelope's `sources`, the WORST "
        f"`confidence_breakdown` domain, and any `caveats` verbatim. The "
        f"workload profile the budget math is conditioned on must be "
        f"named explicitly — don't let the user assume it matches their "
        f"actual traffic shape."
    )
