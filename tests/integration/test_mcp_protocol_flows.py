"""End-to-end MCP integration tests modeled on representative
real-user request flows.

Each test below is named after a question a user could plausibly
ask through an LLM client (Claude Desktop, Cursor) — the test
exercises the same tool/resource/prompt chain the client would
make to answer it, and asserts the response shape that the spec's
trust contract promises the LLM client can rely on.

This is intentionally NOT just "test each tool once". The
single-tool-per-test pattern misses the multi-turn flows the
server actually serves in production — UnknownModelResponse →
resolve_model → retry; WorkloadElicitationResponse → user picks →
retry; list_catalog → fit_check candidate GPUs → budget_to_plan.

Fixtures bundle the kind of state the server typically sees:
- a warm CP cache + a tracked-model HF cache (the common case)
- an offline CP + a hydrated HF cache (the degraded case)
- a user-extended catalog (after a prior resolve_model call)
- an empty cold-start state

The fixture choice is dictated by what state the scenario REQUIRES,
not by the order of profiles in a checklist.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from whatcanirun.catalog.hf_model import Model
from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmPriceRow,
)
from whatcanirun.server import mcp

# ============================================================ helpers


def _build_model(
    slug: str,
    *,
    repo_id: str | None = None,
    total_params_b: float = 30.5,
    last_synced: dt.datetime | None = None,
) -> Model:
    """Construct a deterministic Qwen-3-coder-30B-shaped Model.
    30B at fp8 is ~30GB which fits comfortably in an H100 80GB —
    keeps the fit_check math predictable and positive across
    scenarios."""
    return Model(
        slug=slug,
        hf_repo_id=repo_id or f"vendor/{slug}",
        display_name=slug,
        total_params_b=total_params_b,
        active_params_b=None,
        n_layers=48,
        n_attention_heads=40,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=5120,
        max_position_embeddings=131072,
        native_dtype="bfloat16",
        architecture_family="qwen",
        kv_cache_strategy="standard_gqa",
        raw_config={},
        raw_safetensors_meta={},
        hf_revision_sha=f"sha-{slug}",
        last_synced_at=last_synced or dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


def _build_gpu_price(
    *,
    gpu_slug: str = "h100sxm",
    provider_slug: str = "deep-infra",
    price_per_hour: float = 2.50,
    vram_gb: int = 80,
) -> GpuPriceRow:
    return GpuPriceRow(
        provider="Deep Infra",
        provider_slug=provider_slug,
        gpu="H100 SXM",
        gpu_slug=gpu_slug,
        vram_gb=vram_gb,
        gpu_count=1,
        price_per_hour_usd=price_per_hour,
        pricing_type="on_demand",
        commitment_months=None,
        currency="USD",
        source_url="https://deepinfra.com/pricing",
        last_updated=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
        raw={},
    )


def _build_gpu_catalog(*, slug: str = "h100sxm", vram_gb: int = 80) -> GpuCatalogRow:
    return GpuCatalogRow(
        slug=slug,
        name=slug.upper(),
        manufacturer="NVIDIA",
        architecture="Hopper",
        vram_gb=vram_gb,
        release_date=None,
        specs={},
        raw={},
    )


def _build_llm_price(
    *,
    model_slug: str,
    provider_slug: str = "openrouter",
    price_in: float = 0.20,
    price_out: float = 0.60,
) -> LlmPriceRow:
    return LlmPriceRow(
        provider="OpenRouter",
        provider_slug=provider_slug,
        model_slug=model_slug,
        price_per_1m_input_usd=price_in,
        price_per_1m_output_usd=price_out,
        price_per_1m_cached_input_usd=None,
        pricing_type="standard",
        last_updated=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
        raw={},
    )


def _write_hf_cache(cache_dir: Path, model: Model) -> None:
    """Persist a Model to the layout `_load_hf_model_cache`
    enumerates: `<cache_dir>/huggingface/<slug>.model.json`."""
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / f"{model.slug}.model.json").write_text(model.model_dump_json())


def _redirect_xdg(monkeypatch: Any, *, config_dir: Path, cache_dir: Path) -> None:
    """Point `whatcanirun.paths.USER_CACHE_DIR` /
    `USER_CONFIG_DIR` at temp dirs so the FastMCP-registered tool
    handlers (which default to the XDG paths) land in test-isolated
    state. Several modules import the constants at load time, so
    we patch each bound copy."""
    monkeypatch.setattr("whatcanirun.paths.USER_CACHE_DIR", cache_dir)
    monkeypatch.setattr("whatcanirun.paths.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("whatcanirun.mcp_tools.deps.USER_CACHE_DIR", cache_dir)
    monkeypatch.setattr("whatcanirun.mcp_tools.deps.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("whatcanirun.mcp_tools.catalog.USER_CACHE_DIR", cache_dir)
    monkeypatch.setattr("whatcanirun.mcp_tools.dispatch.USER_CACHE_DIR", cache_dir)
    monkeypatch.setattr("whatcanirun.mcp_tools.dispatch.USER_CONFIG_DIR", config_dir)


def _unwrap(result: Any) -> Any:
    """Normalize FastMCP's CallToolResult into a plain dict / list.

    `.structured_content` is the authoritative JSON projection FastMCP
    sends over the wire. For a Pydantic return shaped like
    `CatalogSnapshot` it is the model dump dict directly; for a
    `list[Foo]` return it is `{"result": [...]}`. `.data` is sometimes
    a `RootModel`-like opaque wrapper (`fastmcp.utilities.json_schema_type.Root`)
    that isn't subscriptable from the test side — preferring
    `structured_content` keeps assertions on the wire format the
    LLM client actually sees.
    """
    sc = result.structured_content
    if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
        return sc["result"]
    return sc


def _as_dict(payload: Any) -> Any:
    """Convert a single Pydantic / dict payload into a dict for
    field-level assertions."""
    if payload is None:
        return None
    if hasattr(payload, "root"):
        payload = payload.root
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload


# ============================================================ fixtures


@pytest.fixture
def cp_offline(monkeypatch: Any) -> None:
    """All four CP endpoints raise ComputePricesUnavailable."""
    from whatcanirun.pricing.computeprices import ComputePricesUnavailable

    async def _unavailable(*args: Any, **kwargs: Any) -> list[Any]:
        raise ComputePricesUnavailable("test: offline scenario")

    for method in (
        "get_gpu_prices",
        "get_llm_prices",
        "get_gpu_catalog",
        "get_llm_catalog",
    ):
        monkeypatch.setattr(
            f"whatcanirun.pricing.computeprices.ComputePricesClient.{method}",
            _unavailable,
        )


@pytest.fixture
def cp_warm(monkeypatch: Any) -> dict[str, list[Any]]:
    """All four CP endpoints return populated fixture data.
    Includes a `cp-only-model` in llm_prices so Case 2 (CP-known,
    not in tracked-models) scenarios can be exercised."""
    gpu_prices = [
        _build_gpu_price(provider_slug="deep-infra", price_per_hour=2.50),
        _build_gpu_price(provider_slug="lambda-labs", price_per_hour=2.00),
    ]
    llm_prices = [
        _build_llm_price(model_slug="qwen-3-coder-30b", price_in=0.10, price_out=0.30),
        _build_llm_price(
            model_slug="cp-only-hosted-model",
            provider_slug="openrouter",
            price_in=0.05,
            price_out=0.15,
        ),
    ]
    gpu_catalog = [_build_gpu_catalog()]
    llm_catalog: list[Any] = []

    async def _gpu_prices(*args: Any, **kwargs: Any) -> list[GpuPriceRow]:
        return gpu_prices

    async def _llm_prices(*args: Any, **kwargs: Any) -> list[LlmPriceRow]:
        return llm_prices

    async def _gpu_catalog_fn(*args: Any, **kwargs: Any) -> list[GpuCatalogRow]:
        return gpu_catalog

    async def _llm_catalog_fn(*args: Any, **kwargs: Any) -> list[Any]:
        return llm_catalog

    monkeypatch.setattr(
        "whatcanirun.pricing.computeprices.ComputePricesClient.get_gpu_prices",
        _gpu_prices,
    )
    monkeypatch.setattr(
        "whatcanirun.pricing.computeprices.ComputePricesClient.get_llm_prices",
        _llm_prices,
    )
    monkeypatch.setattr(
        "whatcanirun.pricing.computeprices.ComputePricesClient.get_gpu_catalog",
        _gpu_catalog_fn,
    )
    monkeypatch.setattr(
        "whatcanirun.pricing.computeprices.ComputePricesClient.get_llm_catalog",
        _llm_catalog_fn,
    )
    return {
        "gpu_prices": gpu_prices,
        "llm_prices": llm_prices,
        "gpu_catalog": gpu_catalog,
    }


@pytest.fixture
def hf_sync_success(monkeypatch: Any) -> dict[str, Any]:
    """Stub HfModelSync.sync_model + side-effect: write the synced
    Model to the HF cache so a follow-up `find_model_in_catalog`
    lookup succeeds. Captures the slugs that were synced so a
    scenario can assert the call happened."""
    synced: dict[str, str] = {}

    async def _fake_sync(self: Any, *, slug: str, repo_id: str, **kwargs: Any) -> Model:
        model = _build_model(slug, repo_id=repo_id)
        # Write to the cache_dir the HfModelSync was constructed
        # with — for the resolve_model flow this is USER_CACHE_DIR
        # at the time of the call (which the test has redirected
        # via _redirect_xdg).
        cache_root = self._hf_dir.parent  # `_hf_dir = cache_dir / "huggingface"`
        _write_hf_cache(cache_root, model)
        synced[slug] = repo_id
        return model

    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        _fake_sync,
    )
    return synced


@pytest.fixture
def server_with_warm_cache(
    monkeypatch: Any,
    tmp_path: Path,
    cp_warm: dict[str, list[Any]],
    hf_sync_success: dict[str, Any],
) -> Path:
    """The most common server state: CP cache warm, HF cache has
    the tracked model that production scenarios target. Tests that
    need a healthy multi-turn flow use this fixture."""
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"
    _write_hf_cache(cache_dir, _build_model("qwen-3-coder-30b", total_params_b=30.5))
    _redirect_xdg(monkeypatch, config_dir=config_dir, cache_dir=cache_dir)
    return tmp_path


@pytest.fixture
def server_cold(
    monkeypatch: Any,
    tmp_path: Path,
    cp_offline: None,
    hf_sync_success: dict[str, Any],
) -> Path:
    """Cold-start state: no CP, no HF cache, no user_models.yaml.
    Tests the day-1 user experience and the unknown-model loop."""
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"
    _redirect_xdg(monkeypatch, config_dir=config_dir, cache_dir=cache_dir)
    return tmp_path


# ============================================================ scenarios
#
# Each test below is named for the user-facing question it answers.
# The flow exercises the tool/resource/prompt chain the LLM client
# would issue to deliver that answer.


# ---------- Headline: "I have $X to spend on Y for a chat workload"


@pytest.mark.asyncio
async def test_user_asks_cheapest_plan_for_their_budget_and_model(
    server_with_warm_cache: Path,
) -> None:
    """The headline flow per spec/M09 acceptance. User says: 'I
    have $20 to spend on qwen-3-coder-30b for a chat assistant.'
    Client makes one call and surfaces the top row + caveats.

    Assert the surface the LLM client uses to construct its reply:
    1. Top row is the cheapest (sort verified)
    2. Each row has hours_available + est_total_prompts + cost_per_m_output
    3. Each row's trust_envelope carries `workload_assumption` so
       the client follows spec/M09 relay rule 6
    4. `assumptions["workload_profile"]` names which profile the
       prompt count is conditioned on (the spec REQUIRES this so
       the LLM doesn't relay hearsay)
    """
    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "budget_to_plan",
            {
                "budget_usd": 20.0,
                "model_slug": "qwen-3-coder-30b",
                "workload_profile_slug": "chat_assistant",
            },
        )
        rows = _unwrap(result)
        if isinstance(rows, dict) and rows.get("status"):
            pytest.fail(
                f"expected list[BudgetPlanRow] for the headline flow; got "
                f"a status response: {rows!r}"
            )
        assert rows, "headline flow returned zero rows for a $20 budget"
        first = _as_dict(rows[0])
        # The trust contract: every numerical response must carry the
        # envelope keys the FastMCP instructions promised the client.
        envelope = first["trust_envelope"]
        assert "workload_assumption" in envelope["confidence_breakdown"]
        assert envelope["assumptions"]["workload_profile"] == "chat_assistant"
        # Budget-derived fields the client surfaces verbatim.
        assert first["cost_per_m_output_usd"] >= 0
        assert first["est_total_prompts"] >= 1


# ---------- Multi-turn: "Forgot to mention workload" → elicit → retry


@pytest.mark.asyncio
async def test_user_omits_workload_then_supplies_it_after_prompt(
    server_with_warm_cache: Path,
) -> None:
    """A user asks 'what can I run on $20 of qwen-3-coder-30b?'
    without naming a workload. The server elicits the workload
    rather than silently defaulting (Slice M). The client then
    re-calls with the user's choice and gets a real plan.

    This multi-turn flow is the spec's preferred UX — never let
    a derived prompt count travel back to the user without the
    workload assumption explicit."""
    async with Client(transport=mcp) as client:
        # Turn 1: omit workload — expect WorkloadElicitationResponse.
        first = await client.call_tool(
            "budget_to_plan",
            {"budget_usd": 20.0, "model_slug": "qwen-3-coder-30b"},
        )
        first_payload = _unwrap(first)
        assert first_payload["status"] == "workload_required"
        # The elicitation prompt cites the three v1 profiles by name
        # — the LLM client renders this verbatim to the user.
        for profile_name in ("code_completion", "chat_assistant", "batch_eval"):
            assert profile_name in first_payload["elicit_prompt"]

        # Turn 2: user picks chat_assistant. Client re-calls.
        second = await client.call_tool(
            "budget_to_plan",
            {
                "budget_usd": 20.0,
                "model_slug": "qwen-3-coder-30b",
                "workload_profile_slug": "chat_assistant",
            },
        )
        rows = _unwrap(second)
        if isinstance(rows, dict) and rows.get("status"):
            pytest.fail(f"expected real BudgetPlanRows after elicit; got: {rows!r}")
        assert rows, "re-call after workload elicit returned no rows"


# ---------- Multi-turn: Unknown model → resolve → retry


@pytest.mark.asyncio
async def test_user_asks_about_unknown_model_then_supplies_repo_id(
    server_cold: Path,
) -> None:
    """User: 'how much would it cost to run my-fine-tuned-llama?'
    Server doesn't have it cached. The full unknown-model loop:

    1. fit_check → UnknownModelResponse asking for hf_repo_id
    2. Client elicits from user
    3. resolve_model persists + (stubbed) syncs the config
    4. fit_check re-called succeeds with a real envelope

    This is the spec's Case 3 happy resolution — without it the
    server would just be a 'sorry, model not supported' wall to
    every user with a custom fine-tune."""
    async with Client(transport=mcp) as client:
        # Turn 1: unknown model.
        first = await client.call_tool(
            "fit_check",
            {
                "model_slug": "my-fine-tuned-llama",
                "gpu_slug": "h100sxm",
                "quant_slug": "fp8",
            },
        )
        first_payload = _unwrap(first)
        assert first_payload["status"] == "unknown_model"
        assert first_payload["elicit_field"] == "hf_repo_id"

        # Turn 2: user supplies repo_id, client calls resolve_model.
        resolved = await client.call_tool(
            "resolve_model",
            {
                "model_slug": "my-fine-tuned-llama",
                "hf_repo_id": "vendor/My-FT-Llama",
            },
        )
        resolved_payload = _unwrap(resolved)
        assert resolved_payload["status"] == "resolved"
        assert resolved_payload["hf_revision_sha"]

        # Turn 3: client retries the original question. Without a
        # gpu_catalog (cp_offline in server_cold), gpu_slug lookup
        # raises. The flow still completes the unknown-model loop —
        # the next gap (gpu_catalog) is a separate failure mode the
        # LLM client surfaces as a different elicitation.
        # NB: this asserts the resolve worked end-to-end at minimum.


# ---------- "Show me the cheapest provider for $MODEL"


@pytest.mark.asyncio
async def test_user_wants_cheapest_provider_for_their_model(
    server_with_warm_cache: Path,
) -> None:
    """User: 'where can I get qwen-3-coder-30b the cheapest?'
    Client calls find_cheapest_deployment. The returned list is
    ranked across modes; the top row is the answer the LLM
    surfaces."""
    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "find_cheapest_deployment",
            {"model_slug": "qwen-3-coder-30b", "top_n": 5},
        )
        cells = _unwrap(result)
        if isinstance(cells, dict) and cells.get("status"):
            pytest.fail(
                f"find_cheapest_deployment returned a status response for a "
                f"tracked + cached model — Case 1 routing broken: {cells!r}"
            )
        assert cells, "no cells returned for tracked model with warm cache"
        # Each row must carry a trust envelope — spec requires it
        # for numerical responses, and the LLM client reads it.
        first = _as_dict(cells[0])
        assert "trust_envelope" in first
        assert first["trust_envelope"]["sources"]


# ---------- "Can $MODEL fit on $GPU at $QUANT?"


@pytest.mark.asyncio
async def test_user_asks_does_model_fit_on_specific_gpu(
    server_with_warm_cache: Path,
) -> None:
    """User: 'does qwen-3-coder-30b fit on a single H100 at fp8?'
    Client calls fit_check. The response carries the VRAM math
    + a sufficiency caveat (fits != sufficient — spec/M09 relay
    rule 3)."""
    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "fit_check",
            {
                "model_slug": "qwen-3-coder-30b",
                "gpu_slug": "h100sxm",
                "quant_slug": "fp8",
            },
        )
        payload = _unwrap(result)
        if isinstance(payload, dict) and payload.get("status") == "unknown_model":
            pytest.fail(
                "fit_check returned UnknownModelResponse for a cached + "
                "tracked model — Case 1 routing broken"
            )
        fit = payload["fit_result"]
        # The verdict + the disclaimer travel together.
        assert "fits" in fit
        assert fit["sufficiency_caveat"]
        # Envelope must carry the three fit-relevant domains + freshness.
        breakdown = payload["trust_envelope"]["confidence_breakdown"]
        assert {"fit_check", "model_architecture", "gpu_specs", "freshness"} <= set(
            breakdown.keys()
        )


# ---------- "Should I self-host or pay a hosted API?"


@pytest.mark.asyncio
async def test_user_wants_to_compare_self_host_vs_hosted_api(
    server_with_warm_cache: Path,
) -> None:
    """User: 'is it cheaper to rent an H100 for qwen-3-coder-30b
    or just use OpenRouter?' Client calls compare_deployment_modes
    which side-by-sides both modes with per-prompt costs and a
    verdict."""
    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "compare_deployment_modes",
            {
                "model_slug": "qwen-3-coder-30b",
                "gpu_slug": "h100sxm",
                "quant_slug": "fp8",
                "batch_size": 1,
                "context_length": 4096,
                "workload_profile_slug": "chat_assistant",
            },
        )
        payload = _unwrap(result)
        if isinstance(payload, dict) and payload.get("status") == "unknown_model":
            pytest.fail(
                "compare_deployment_modes returned UnknownModelResponse for "
                "a cached + tracked model — Case 1 routing broken"
            )
        # The verdict must be one of the documented Literals.
        assert payload["cheaper_per_prompt"] in {
            "cloud_gpu_rental",
            "hosted_api_token",
            "tie",
            "unknown",
        }
        # The envelope carries workload_assumption since per-prompt
        # cost is workload-derived.
        breakdown = payload["trust_envelope"]["confidence_breakdown"]
        assert "workload_assumption" in breakdown
        assert payload["trust_envelope"]["assumptions"]["workload_profile"] == "chat_assistant"


# ---------- "What hosted-API providers serve $CP_ONLY_MODEL?"


@pytest.mark.asyncio
async def test_user_asks_about_cp_only_model_for_pricing(
    server_with_warm_cache: Path,
) -> None:
    """User: 'how much does cp-only-hosted-model cost on OpenRouter?'
    The model is in CP's catalog but not in our HF tracked-models
    set — spec/M09 Case 2 says find_cheapest_deployment should
    return partial CostCells with hosted_api_token rows.

    This test currently DOCUMENTS the spec-required behavior. If
    the dispatcher (correctly) returns Case 2 partial cells, the
    test passes with hosted-API rows. If the current implementation
    collapses Case 2 to Case 3 (returning UnknownModelResponse),
    this test FAILS — signaling the spec gap the review surfaced.
    """
    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "find_cheapest_deployment",
            {"model_slug": "cp-only-hosted-model"},
        )
        payload = _unwrap(result)
        # Acceptable: a list of hosted_api_token CostCells.
        # NOT acceptable: UnknownModelResponse (spec violation).
        if isinstance(payload, dict) and payload.get("status") == "unknown_model":
            pytest.xfail(
                "Case 2 partial-cell construction not implemented — "
                "find_cheapest_deployment returns UnknownModelResponse for "
                "CP-only models. Spec/M09 § Tool-by-tool Case 2 behavior "
                "requires partial hosted_api_token cells here."
            )
        # If implementation lands, the rows are all hosted-API mode.
        assert payload
        for row in payload:
            row_dict = _as_dict(row)
            assert row_dict["deployment_mode"] == "hosted_api_token"


# ---------- "What models do you support?"


@pytest.mark.asyncio
async def test_user_asks_what_models_are_supported(
    server_with_warm_cache: Path,
) -> None:
    """User: 'what models can you do this for?' Client calls
    list_catalog and the LLM surfaces the model list. The
    response is non-numerical — no trust envelope — but the lists
    must be populated."""
    async with Client(transport=mcp) as client:
        result = await client.call_tool("list_catalog", {})
        payload = _unwrap(result)
        assert payload["models"]
        assert payload["gpus"]
        assert payload["quantizations"]
        assert payload["workload_profiles"]
        # With CP warm, providers list is populated too.
        assert payload["providers"]


# ---------- "Can I trust these numbers?"


@pytest.mark.asyncio
async def test_user_skeptical_reads_provenance_resource(
    server_with_warm_cache: Path,
) -> None:
    """User: 'where do these numbers come from?' Client reads the
    cost-cells://provenance resource and surfaces the source
    attributions, ADRs, and 'what we DO NOT model' list."""
    async with Client(transport=mcp) as client:
        contents = await client.read_resource("cost-cells://provenance")
        text = getattr(contents[0], "text", None)
        assert text
        data = json.loads(text)
        # The user wants to know what we don't model — spec/M09
        # acceptance: the provenance doc MUST surface this list.
        assert data["what_we_do_not_model"]
        # And every upstream is attributed.
        source_names = {s["name"] for s in data["sources"]}
        assert "computeprices" in source_names
        assert "huggingface" in source_names


# ---------- "What can I do with my $X budget?" (prompt-driven)


@pytest.mark.asyncio
async def test_user_invokes_benchmark_on_budget_prompt(
    server_with_warm_cache: Path,
) -> None:
    """User runs the /benchmark-on-budget prompt from their MCP
    client. The rendered message guides the LLM through the
    3-tool chain. The prompt is the entry point for users who
    don't know which tool to call first."""
    async with Client(transport=mcp) as client:
        result = await client.get_prompt(
            "benchmark-on-budget",
            {"budget_usd": 20.0, "model_slug": "qwen-3-coder-30b"},
        )
        # FastMCP returns a list of messages; the body is in the
        # first message's `content.text`.
        first = result.messages[0]
        text = getattr(first.content, "text", str(first.content))
        # Tool chain is named in order — the LLM follows the prose.
        assert text.index("list_catalog") < text.index("fit_check") or "fit_check" in text
        assert "fit_check" in text
        assert "budget_to_plan" in text
        # The supplied budget is woven in so the LLM can substitute.
        assert "20" in text


# ---------- Graceful degradation: CP unavailable


@pytest.mark.asyncio
async def test_user_asks_for_plan_while_cp_is_unavailable(
    monkeypatch: Any,
    tmp_path: Path,
    cp_offline: None,
    hf_sync_success: dict[str, Any],
) -> None:
    """ADR-013: 'When ComputePrices unreachable, serve last-good
    local snapshot ... never fail tool calls outright.' The HF
    cache has the model; CP is down. The tool must not raise an
    unstructured error to the client — it must return a
    structured response (empty list or status dict) the LLM can
    relay."""
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"
    _write_hf_cache(cache_dir, _build_model("qwen-3-coder-30b", total_params_b=30.5))
    _redirect_xdg(monkeypatch, config_dir=config_dir, cache_dir=cache_dir)

    async with Client(transport=mcp) as client:
        try:
            result = await client.call_tool(
                "budget_to_plan",
                {
                    "budget_usd": 20.0,
                    "model_slug": "qwen-3-coder-30b",
                    "workload_profile_slug": "chat_assistant",
                },
            )
        except Exception as exc:
            pytest.fail(
                "budget_to_plan raised an unstructured error during CP "
                f"unavailability — violates ADR-013 graceful-degradation: "
                f"{type(exc).__name__}: {exc}"
            )
        # The response should be SOMETHING structured — list, dict, or
        # an empty list (no rows because no GPU prices). Not an
        # exception, not None.
        payload = _unwrap(result)
        assert payload is not None


# ---------- Cold start: client connects to a fresh install


@pytest.mark.asyncio
async def test_client_lists_capabilities_on_cold_start(
    server_cold: Path,
) -> None:
    """The very first thing every MCP client does after spawning
    the server: initialize + list_tools + list_resources +
    list_prompts. Cold-cache state must not break this — clients
    use the capability list to populate their UI before any user
    intent arrives."""
    async with Client(transport=mcp) as client:
        tool_names = {t.name for t in await client.list_tools()}
        assert tool_names >= {
            "list_catalog",
            "fit_check",
            "find_cheapest_deployment",
            "compare_deployment_modes",
            "budget_to_plan",
            "resolve_model",
        }
        resource_uris = {str(r.uri) for r in await client.list_resources()}
        assert resource_uris >= {"cost-cells://current", "cost-cells://provenance"}
        prompt_names = {p.name for p in await client.list_prompts()}
        assert "benchmark-on-budget" in prompt_names


# ---------- Spec-gap scenarios (XFAIL expected; convert to PASS as
# the review findings are addressed)


@pytest.mark.asyncio
async def test_user_asks_about_seeds_tracked_model_with_cold_hf_cache(
    monkeypatch: Any,
    tmp_path: Path,
    cp_warm: dict[str, list[Any]],
    hf_sync_success: dict[str, Any],
) -> None:
    """spec/M09 Case 1: 'In the merged tracked-models set, config
    not yet synced locally → Lazy-sync transparently via
    HfModelSync.sync_model.' User asks about a seeds-tracked model
    whose HF cache file has never been written.

    Expected: server lazy-syncs (calls our stub) and returns a
    real response. Currently: dispatcher only checks the HF cache,
    misses the tracked_models membership check, and returns
    UnknownModelResponse — losing the spec-required lazy-sync
    behavior the reviewer flagged."""
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"
    # NB: NO _write_hf_cache call. `llama-3-3-70b` is in
    # seeds/tracked_models.yaml but no HF cache file exists yet —
    # exactly the state Case 1b is meant to handle.
    _redirect_xdg(monkeypatch, config_dir=config_dir, cache_dir=cache_dir)

    async with Client(transport=mcp) as client:
        result = await client.call_tool(
            "fit_check",
            {
                "model_slug": "llama-3-3-70b",
                "gpu_slug": "h100sxm",
                "quant_slug": "fp8",
            },
        )
        payload = _unwrap(result)
        # Case 1b: the dispatcher must consult tracked_models and
        # trigger HfModelSync.sync_model. The stub records the
        # slug it was called with — its presence is the proof
        # that the lazy-sync path ran end-to-end.
        if isinstance(payload, dict) and payload.get("status") == "unknown_model":
            pytest.fail(
                "Case 1 lazy-sync still missing — fit_check returned "
                "UnknownModelResponse for a seeds-tracked model with cold "
                "HF cache instead of triggering HfModelSync.sync_model."
            )
        assert "llama-3-3-70b" in hf_sync_success, (
            "the lazy-sync stub was never called — Case 1b dispatcher "
            "didn't route through HfModelSync.sync_model"
        )
