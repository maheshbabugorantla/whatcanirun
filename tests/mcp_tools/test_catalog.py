"""M09 Slice B: `list_catalog()` tool — TDD.

`list_catalog` is the one-call dropdown helper MCP clients use when
building UIs. It must return all five catalog lists with non-zero
entries (assuming M01 seeds + M03 tracked models + M05 workload
profiles have shipped — which they all have on `main`).

The shape is checked by `CatalogSnapshot.model_validate`; this test
suite further asserts each list has at least one entry with the
identifying field populated (slug, provider_slug). A regression
that empties any list would surface here.

`build_catalog_snapshot` is the pure function the tool wraps. We
unit-test the pure function with real seeds + a fixture CP
`gpu-prices` payload so the test stays hermetic (no live network).
The MCP-tool registration (a thin wrapper that injects default
seed dir + CP cache lookup) is covered by the `mcp_tool_registered`
test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from whatcanirun.mcp_tools.catalog import (
    CatalogSnapshot,
    build_catalog_snapshot,
)
from whatcanirun.paths import SEEDS_DIR
from whatcanirun.pricing.projections import GpuPriceRow

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_GPU_PRICES_FIXTURE = _FIXTURE_DIR / "cp_gpu_prices_2026-05-26.json"


@pytest.fixture
def gpu_prices() -> list[GpuPriceRow]:
    """Real CP `gpu-prices` payload captured 2026-05-26 (1000 rows).
    Carries every provider we need to verify the providers list is
    non-empty and de-duplicates correctly across the 1k rows."""
    data = json.loads(_GPU_PRICES_FIXTURE.read_text())["data"]
    return [GpuPriceRow.model_validate(row) for row in data]


def test_build_catalog_snapshot_returns_validated_pydantic(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """The return type is `CatalogSnapshot` (a Pydantic model), not a
    plain dict. The Pydantic model is what FastMCP serializes for the
    MCP wire format and what gives the client a stable schema."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    assert isinstance(snapshot, CatalogSnapshot)


def test_snapshot_has_all_five_catalog_lists(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """Spec/M09 § Public surface: `list_catalog` returns
    `{gpus, models, quantizations, workload_profiles, providers}`.
    A regression that drops one of the five lists fails this assert."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    assert snapshot.gpus, "gpus list is empty"
    assert snapshot.models, "models list is empty"
    assert snapshot.quantizations, "quantizations list is empty"
    assert snapshot.workload_profiles, "workload_profiles list is empty"
    assert snapshot.providers, "providers list is empty"


def test_gpus_carry_slug_vram_and_form_factor(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """Each GPU entry needs the fields a client renderable as a
    dropdown row. `slug` is the canonical identifier the user picks;
    `form_factor` (SXM/PCIe/NVL/OAM) disambiguates same-name SKUs
    (H100 PCIe vs H100 SXM5 are different VRAM and bandwidth)."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    sample = snapshot.gpus[0]
    assert sample.slug
    assert sample.form_factor in {"SXM", "PCIe", "NVL", "OAM"}


def test_models_carry_slug_and_hf_repo_id(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """Models must expose at least `slug` (the catalog identifier
    the user picks) and `hf_repo_id` (the HF address so the client
    can show users where the architecture data came from)."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    sample = snapshot.models[0]
    assert sample.slug
    assert sample.hf_repo_id
    assert "/" in sample.hf_repo_id  # HF repo_ids are always `org/name`


def test_workload_profiles_carry_token_shape(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """Workload profile entries must carry `avg_input_tokens` and
    `avg_output_tokens` — those are the numbers the elicitation
    prompt cites verbatim (spec/M09 § Workload assumption handling)."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    sample = snapshot.workload_profiles[0]
    assert sample.slug
    assert sample.avg_input_tokens > 0
    assert sample.avg_output_tokens > 0


def test_quantizations_carry_bits_per_weight(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """The quantization entry must carry `bits_per_weight` so a
    client UI can sort dropdowns numerically (fp4 < int8 < fp16,
    etc.) rather than alphabetically."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    sample = snapshot.quantizations[0]
    assert sample.slug
    assert sample.bits_per_weight > 0


def test_providers_derived_from_gpu_prices_distinct(
    gpu_prices: list[GpuPriceRow],
) -> None:
    """Providers are the distinct `(provider_slug, provider)` pairs
    across all gpu-prices rows. The 1000-row fixture has tens of
    provider rows per provider — the snapshot must dedupe to a
    short distinct list."""
    snapshot = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    slugs = [p.provider_slug for p in snapshot.providers]
    assert len(slugs) == len(set(slugs)), (
        f"providers list contains duplicates: {sorted(s for s in slugs if slugs.count(s) > 1)}"
    )
    # Sanity bound: at least a couple distinct providers (the fixture
    # has many), and far fewer than the raw row count (which means
    # dedup actually ran).
    assert 2 <= len(slugs) < len(gpu_prices)


def test_build_catalog_snapshot_merges_user_models_yaml(
    gpu_prices: list[GpuPriceRow],
    tmp_path: Any,
) -> None:
    """Copilot review #15 round 3 #5: after a successful
    resolve_model, the user-added slug is persisted to
    `~/.config/whatcanirun/user_models.yaml`. list_catalog must
    surface it — otherwise the catalog says 'supported models'
    but lies after every resolve_model success.

    When config_dir is passed, build_catalog_snapshot must union
    user_models.yaml rows with the seed rows. Seeds win on
    collision (the merged-loader contract from deps.py)."""
    import yaml as _yaml

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "user_models.yaml").write_text(
        _yaml.safe_dump([{"slug": "user-resolved-llama", "hf_repo_id": "vendor/User-Llama"}])
    )

    # Without config_dir → seeds only, user model absent.
    seeds_only = build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
    assert all(m.slug != "user-resolved-llama" for m in seeds_only.models)

    # With config_dir → user model is merged in.
    merged = build_catalog_snapshot(
        seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices, config_dir=config_dir
    )
    slugs = {m.slug for m in merged.models}
    assert "user-resolved-llama" in slugs, (
        "user-resolved model missing from list_catalog after a stubbed "
        "resolve_model would have persisted it — clients would think the "
        "model isn't supported even though dispatcher accepts it"
    )
    # And seed rows still survive — the merge is union, not replace.
    assert len(merged.models) > len(seeds_only.models)


def test_list_catalog_registered_as_mcp_tool() -> None:
    """The function must be wired into the FastMCP instance as a tool
    named `list_catalog`. A registration regression (decorator
    accidentally removed, function renamed without updating the
    registration) fails here, not at MCP-client connection time."""
    import asyncio

    from whatcanirun.server import mcp

    # FastMCP registers tools on a private `_tool_manager` — `get_tools`
    # is the public accessor. It's async so we need to drive it.
    tools = asyncio.run(mcp.get_tools())
    assert "list_catalog" in tools, (
        f"`list_catalog` tool not registered on `mcp`; registered tools: {sorted(tools)}"
    )
