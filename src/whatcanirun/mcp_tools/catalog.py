"""M09 Slice B: `list_catalog` tool — the one-call dropdown helper.

MCP clients building UIs (provider pickers, model autocomplete,
quantization dropdowns) hit this once to populate every list at
the same time, rather than chasing five separate endpoints. The
response is non-numerical — no trust envelope — because there are
no derived numbers to wrap, only catalog facts.

`build_catalog_snapshot` is the pure function exposed for unit
tests: takes seed paths + CP gpu-prices rows, returns a
`CatalogSnapshot`. `list_catalog` is the FastMCP-registered tool
that wraps it with default paths and a `ComputePricesClient`
pointed at the user's XDG cache. CP unavailability degrades to
an empty `providers` list rather than failing the call — the
other four lists remain useful, and the missing slot is a visible
signal that the cache hasn't been warmed yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.loaders import (
    load_gpu_supplements,
    load_quantizations,
    load_tracked_models,
    load_workload_profiles,
)
from whatcanirun.paths import SEEDS_DIR, USER_CACHE_DIR
from whatcanirun.pricing.computeprices import (
    ComputePricesClient,
    ComputePricesUnavailable,
)
from whatcanirun.pricing.projections import GpuPriceRow

FormFactor = Literal["SXM", "PCIe", "NVL", "OAM"]


class GpuSummary(BaseModel):
    """Per-GPU dropdown row. `form_factor` disambiguates same-name
    SKUs (H100 PCIe vs H100 SXM5 are different VRAM and bandwidth)."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    form_factor: FormFactor


class ModelSummary(BaseModel):
    """Per-model dropdown row. `hf_repo_id` lets the client show the
    user where the architecture data was sourced. `display_name`
    is optional — when omitted, the client should fall back to
    the last path segment of `hf_repo_id`."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    hf_repo_id: str
    display_name: str | None = None


class QuantizationSummary(BaseModel):
    """Per-quantization dropdown row. `bits_per_weight` lets a
    client UI sort numerically (fp4 < int8 < fp16) rather than
    alphabetically. `experimental=True` marks formats whose
    accept-criteria semantics aren't yet pinned down — clients
    may want to surface a warning chip."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    bits_per_weight: int
    experimental: bool


class WorkloadSummary(BaseModel):
    """Per-workload-profile dropdown row. The `avg_input_tokens`
    and `avg_output_tokens` are the numbers cited verbatim in
    Slice M's `WorkloadElicitationResponse.elicit_prompt`."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    avg_input_tokens: int
    avg_output_tokens: int
    is_default: bool


class ProviderSummary(BaseModel):
    """Per-provider dropdown row. The `provider_slug` is what
    appears in CostCell rows; `display_name` is the rendering name
    (e.g. 'Lambda Labs' for `lambda-labs`)."""

    model_config = ConfigDict(extra="forbid")

    provider_slug: str
    display_name: str


class CatalogSnapshot(BaseModel):
    """The dict-of-five-lists the MCP client gets from
    `list_catalog`. Pydantic-validated so the wire schema is
    stable; `extra="forbid"` so a typo or rogue field surfaces at
    construction, not on the client side."""

    model_config = ConfigDict(extra="forbid")

    gpus: list[GpuSummary]
    models: list[ModelSummary]
    quantizations: list[QuantizationSummary]
    workload_profiles: list[WorkloadSummary]
    providers: list[ProviderSummary]


def build_catalog_snapshot(
    *,
    seeds_dir: Path,
    gpu_prices: list[GpuPriceRow],
) -> CatalogSnapshot:
    """Pure constructor: seed YAML reads + dedup of providers from
    CP gpu_prices. The caller is responsible for fetching the
    gpu_prices rows (via `ComputePricesClient.get_gpu_prices()` in
    the tool wrapper, via a fixture in tests).

    The provider dedup keeps insertion order so a client UI sees
    a stable list across calls — `dict[str, str]` is insertion-
    ordered since 3.7."""
    gpu_supplements = load_gpu_supplements(seeds_dir / "gpus_supplement.yaml")
    quantizations = load_quantizations(seeds_dir / "quantizations.yaml")
    tracked_models = load_tracked_models(seeds_dir / "tracked_models.yaml")
    workload_profiles = load_workload_profiles(seeds_dir / "workload_profiles.yaml")

    # Distinct providers by slug, first occurrence wins for the
    # display name (provider name strings are stable across rows
    # in the CP catalog; the dedup just preserves whichever the
    # client sees first).
    provider_display: dict[str, str] = {}
    for row in gpu_prices:
        provider_display.setdefault(row.provider_slug, row.provider)

    return CatalogSnapshot(
        gpus=[GpuSummary(slug=g.slug, form_factor=g.form_factor) for g in gpu_supplements],
        models=[
            ModelSummary(
                slug=m.slug,
                hf_repo_id=m.hf_repo_id,
                display_name=m.display_name,
            )
            for m in tracked_models
        ],
        quantizations=[
            QuantizationSummary(
                slug=q.slug,
                bits_per_weight=q.bits_per_weight,
                experimental=q.experimental,
            )
            for q in quantizations
        ],
        workload_profiles=[
            WorkloadSummary(
                slug=w.slug,
                avg_input_tokens=w.avg_input_tokens,
                avg_output_tokens=w.avg_output_tokens,
                is_default=w.is_default,
            )
            for w in workload_profiles
        ],
        providers=[
            ProviderSummary(provider_slug=slug, display_name=display)
            for slug, display in provider_display.items()
        ],
    )


async def list_catalog() -> CatalogSnapshot:
    """`list_catalog` MCP tool entry point.

    Pulls seed YAMLs from `SEEDS_DIR` and gpu_prices from the
    user's CP cache directory. `ComputePricesUnavailable` (no
    network + no warm cache + no fallback snapshot) degrades to
    an empty providers list — the other four lists remain useful
    and the empty slot is a visible signal the cache hasn't been
    warmed yet.
    """
    cp_cache = USER_CACHE_DIR / "computeprices"
    cp_cache.mkdir(parents=True, exist_ok=True)
    client = ComputePricesClient(cache_dir=cp_cache)
    try:
        gpu_prices = await client.get_gpu_prices()
    except ComputePricesUnavailable:
        gpu_prices = []
    return build_catalog_snapshot(seeds_dir=SEEDS_DIR, gpu_prices=gpu_prices)
