"""M09 Slice L+M wiring: end-to-end tool dispatch through the
unknown-model dispatcher + workload elicitation.

These tests verify the async tool wrappers (`fit_check`,
`find_cheapest_deployment`, `compare_deployment_modes`,
`budget_to_plan`) correctly route through:

- Slice L Case 3: model not in tracked-models / HF cache →
  UnknownModelResponse
- Slice M: budget_to_plan without workload_profile_slug →
  WorkloadElicitationResponse (regardless of whether the model
  is known)

The tests stub `load_runtime_deps` so they don't depend on a
warmed CP cache or HF sync state. The happy-path end-to-end
(Case 1, model in catalog) is exercised by the pure builder
tests; this file covers the routing branches that only the
tool wrappers see.
"""

from __future__ import annotations

from typing import Any

import pytest

from whatcanirun.mcp_tools.deps import RuntimeDeps
from whatcanirun.mcp_tools.dispatch import (
    UnknownModelResponse,
    WorkloadElicitationResponse,
)


@pytest.fixture
def empty_deps(monkeypatch: Any) -> RuntimeDeps:
    """Stub `load_runtime_deps` to return an empty RuntimeDeps.
    Every model_slug lookup falls through to Case 3 in this state
    — exactly the scenario we want for unknown-model routing tests."""
    deps = RuntimeDeps()

    async def _fake_load(**kwargs: Any) -> RuntimeDeps:
        return deps

    monkeypatch.setattr(
        "whatcanirun.mcp_tools.deps.load_runtime_deps",
        _fake_load,
    )
    # Also patch the function as imported into each tool module's
    # local namespace. The tool modules `from ... import
    # load_runtime_deps` lazily inside the tool body, so the
    # patch above (on the original module) is what the lazy import
    # picks up — but if a future refactor binds the import at
    # module top, this fixture still works because we patch the
    # source-of-truth name.
    return deps


# ---------------------------------------------------------------- Slice L


@pytest.mark.asyncio
async def test_fit_check_returns_unknown_for_uncached_model(
    empty_deps: RuntimeDeps,
) -> None:
    """Per spec/M09 Case 2 + Tool-by-tool Case 2 behavior:
    fit_check collapses Case 2 (CP-only) to Case 3
    (UnknownModelResponse) — fit-checking fundamentally requires
    architecture data, so an unknown model returns the elicitation
    rather than a hollow FitResult."""
    from whatcanirun.mcp_tools.fit_check import fit_check

    result = await fit_check(
        model_slug="mystery-model",
        gpu_slug="h100sxm",
        quant_slug="fp8",
    )
    assert isinstance(result, UnknownModelResponse)
    assert result.requested_model_slug == "mystery-model"


@pytest.mark.asyncio
async def test_find_cheapest_returns_unknown_for_uncached_model(
    empty_deps: RuntimeDeps,
) -> None:
    """find_cheapest_deployment routes uncached models to Case 3
    too (Case 2 partial-CostCell support is M11 follow-up)."""
    from whatcanirun.mcp_tools.find_cheapest import find_cheapest_deployment

    result = await find_cheapest_deployment(model_slug="mystery-model", top_n=3)
    assert isinstance(result, UnknownModelResponse)
    assert result.requested_model_slug == "mystery-model"


@pytest.mark.asyncio
async def test_compare_deployment_modes_returns_unknown_for_uncached_model(
    empty_deps: RuntimeDeps,
) -> None:
    """compare_deployment_modes hard-collapses Case 2 to Case 3
    per spec — its whole purpose is comparing both modes, and
    without architecture data the cloud side is impossible."""
    from whatcanirun.mcp_tools.compare_deployment import compare_deployment_modes

    result = await compare_deployment_modes(
        model_slug="mystery-model",
        gpu_slug="h100sxm",
        quant_slug="fp8",
        batch_size=1,
        context_length=4096,
        workload_profile_slug="chat_assistant",
    )
    assert isinstance(result, UnknownModelResponse)
    assert result.requested_model_slug == "mystery-model"


@pytest.mark.asyncio
async def test_budget_to_plan_returns_unknown_for_uncached_model(
    empty_deps: RuntimeDeps,
) -> None:
    """budget_to_plan with an unknown model + supplied workload
    returns UnknownModelResponse. The workload elicitation
    short-circuits AHEAD of the model lookup — if the user
    omitted both, they see WorkloadElicitationResponse first."""
    from whatcanirun.mcp_tools.budget_to_plan import budget_to_plan

    result = await budget_to_plan(
        budget_usd=20.0,
        model_slug="mystery-model",
        workload_profile_slug="chat_assistant",
    )
    assert isinstance(result, UnknownModelResponse)


# ---------------------------------------------------------------- Slice M


@pytest.mark.asyncio
async def test_budget_to_plan_elicits_workload_when_omitted() -> None:
    """Per spec/M09 § Workload assumption handling: a missing
    workload_profile_slug must produce WorkloadElicitationResponse
    rather than silently defaulting. The check runs BEFORE
    load_runtime_deps so it works without any cache state."""
    from whatcanirun.mcp_tools.budget_to_plan import budget_to_plan

    result = await budget_to_plan(
        budget_usd=20.0,
        model_slug="any-model",
        workload_profile_slug=None,
    )
    assert isinstance(result, WorkloadElicitationResponse)
    assert result.requested_model_slug == "any-model"
    assert result.status == "workload_required"
    assert result.elicit_field == "workload_profile_slug"
    # Spec/M09 § Workload elicit_prompt: lists the three v1 profiles.
    assert "code_completion" in result.elicit_prompt
    assert "chat_assistant" in result.elicit_prompt
    assert "batch_eval" in result.elicit_prompt


@pytest.mark.asyncio
async def test_workload_elicitation_includes_followup_suggestions() -> None:
    """The elicit_prompt is the headline ask; suggested_followups
    is the "what to do instead" list including the
    find_cheapest_deployment escape hatch for users who can't
    map their workload to a profile."""
    from whatcanirun.mcp_tools.budget_to_plan import budget_to_plan

    result = await budget_to_plan(budget_usd=10.0, model_slug="x")
    assert isinstance(result, WorkloadElicitationResponse)
    assert len(result.suggested_followups) >= 1
    assert any("find_cheapest_deployment" in s for s in result.suggested_followups)


# ---------------------------------------------------------------- deps loader


@pytest.fixture
def offline_cp(monkeypatch: Any) -> None:
    """Stub the four CP client methods to raise
    ComputePricesUnavailable so the load_runtime_deps degraded-
    cache path is exercised without touching the network."""
    from whatcanirun.pricing.computeprices import ComputePricesUnavailable

    async def _unavailable(*args: Any, **kwargs: Any) -> list[Any]:
        raise ComputePricesUnavailable("test: offline")

    for method in ("get_gpu_prices", "get_llm_prices", "get_gpu_catalog", "get_llm_catalog"):
        monkeypatch.setattr(
            f"whatcanirun.pricing.computeprices.ComputePricesClient.{method}",
            _unavailable,
        )


@pytest.mark.asyncio
async def test_load_runtime_deps_returns_runtime_deps_when_caches_empty(
    tmp_path: Any,
    offline_cp: None,
) -> None:
    """`load_runtime_deps` must degrade gracefully when CP caches
    are empty and there's no HF sync state — every list is empty,
    no exception. The tools then route everything through Case 3
    rather than crashing."""
    from whatcanirun.mcp_tools.deps import load_runtime_deps

    deps = await load_runtime_deps(
        seeds_dir=None,  # use real seeds (which we have)
        cache_dir=tmp_path / "cache",  # empty fresh cache
        config_dir=tmp_path / "config",  # empty fresh config
    )
    assert isinstance(deps, RuntimeDeps)
    # seeds-backed lists should be non-empty (we have shipped seeds).
    assert deps.quantizations
    assert deps.workload_profiles
    assert deps.tracked_models
    # CP / HF caches are cold so those degrade to empty.
    assert deps.gpu_prices == []
    assert deps.llm_prices == []
    assert deps.gpu_catalog == []
    assert deps.model_catalog == []


@pytest.mark.asyncio
async def test_load_runtime_deps_reads_user_models_yaml(tmp_path: Any, offline_cp: None) -> None:
    """The merged-tracked-models contract: user_models.yaml rows
    union with seeds/tracked_models.yaml rows. A user-supplied
    model must appear in `deps.tracked_models` after a successful
    resolve_model_to_user_yaml call."""
    import yaml

    from whatcanirun.mcp_tools.deps import load_runtime_deps

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "user_models.yaml").write_text(
        yaml.safe_dump([{"slug": "user-added-model", "hf_repo_id": "vendor/user-added"}])
    )

    deps = await load_runtime_deps(
        cache_dir=tmp_path / "cache",
        config_dir=config_dir,
    )
    slugs = {r.slug for r in deps.tracked_models}
    assert "user-added-model" in slugs
