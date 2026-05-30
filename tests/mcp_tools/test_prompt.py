"""M09 Slice I: `/benchmark-on-budget` MCP prompt.

The prompt is the guided workflow for first-time users who don't
know which catalog item maps to their idea. It chains:
  list_catalog → fit_check x candidate GPUs → budget_to_plan
with example arguments so the LLM client can render a concrete
recipe rather than asking the user to compose tool calls from
scratch.

Spec/M09 § Prompts §1 and § Acceptance criteria require:
- registered as a prompt named `benchmark-on-budget`
- takes `budget_usd` (required) + `model_slug` (optional)
- references the three tools in order in its message template
"""

from __future__ import annotations

import asyncio

from whatcanirun.server import mcp


def _get_prompt(name: str):  # type: ignore[no-untyped-def]
    """Helper: drive the FastMCP `get_prompt` lookup from sync test."""
    return asyncio.run(mcp.get_prompt(name))


def test_benchmark_on_budget_registered_as_prompt() -> None:
    """Spec/M09 acceptance: `/benchmark-on-budget` chains the
    right tools when invoked from Claude Desktop. The registration
    is the precondition — without it the client can't reach the
    chain at all."""
    prompts = asyncio.run(mcp.get_prompts())
    assert "benchmark-on-budget" in prompts, (
        f"`benchmark-on-budget` prompt not registered; registered prompts: {sorted(prompts)}"
    )


def test_prompt_template_chains_three_tools_in_order() -> None:
    """Spec/M09 § Prompts §1: 'Chains `list_catalog` (if model
    missing) → `fit_check` x candidate GPUs → `budget_to_plan`'.
    All three tool names must appear in the rendered prompt
    message, in that order — the LLM client follows the prose
    rather than computing the chain itself."""
    prompt = _get_prompt("benchmark-on-budget")
    rendered = asyncio.run(prompt.render(arguments={"budget_usd": 20.0}))
    # FastMCP returns a list of PromptMessages; the body is in
    # `content.text` for text-typed messages.
    full_text = "\n".join(getattr(m.content, "text", str(m.content)) for m in rendered)
    assert "list_catalog" in full_text
    assert "fit_check" in full_text
    assert "budget_to_plan" in full_text

    # Order check: list_catalog → fit_check → budget_to_plan.
    pos_list = full_text.index("list_catalog")
    pos_fit = full_text.index("fit_check")
    pos_budget = full_text.index("budget_to_plan")
    assert pos_list < pos_fit < pos_budget, (
        "tools should appear in the order list_catalog → fit_check → "
        f"budget_to_plan; got positions {pos_list}, {pos_fit}, {pos_budget}"
    )


def test_prompt_message_quotes_the_budget_argument() -> None:
    """The rendered prompt must include the supplied `budget_usd`
    so the LLM client can pass it into `budget_to_plan` verbatim.
    A regression where the budget is dropped from the template
    forces the LLM to guess — and the guess would silently default
    to a budget the user didn't specify."""
    prompt = _get_prompt("benchmark-on-budget")
    rendered = asyncio.run(prompt.render(arguments={"budget_usd": 37.5}))
    full_text = "\n".join(getattr(m.content, "text", str(m.content)) for m in rendered)
    assert "37.5" in full_text


def test_prompt_handles_optional_model_slug_unset() -> None:
    """When `model_slug` is omitted, the prompt template steers
    the LLM client to call `list_catalog` first so the user can
    pick. The rendered text should reference the catalog lookup
    step rather than failing with a missing-argument error."""
    prompt = _get_prompt("benchmark-on-budget")
    rendered = asyncio.run(prompt.render(arguments={"budget_usd": 20.0}))
    full_text = "\n".join(getattr(m.content, "text", str(m.content)) for m in rendered)
    assert "list_catalog" in full_text  # the catalog-first guidance


def test_prompt_handles_supplied_model_slug() -> None:
    """When `model_slug` IS supplied, the prompt should incorporate
    it so the LLM client can skip the catalog step and go straight
    to `fit_check`. The slug must appear verbatim in the rendered
    text."""
    prompt = _get_prompt("benchmark-on-budget")
    rendered = asyncio.run(
        prompt.render(arguments={"budget_usd": 20.0, "model_slug": "qwen-3-coder-30b"})
    )
    full_text = "\n".join(getattr(m.content, "text", str(m.content)) for m in rendered)
    assert "qwen-3-coder-30b" in full_text
