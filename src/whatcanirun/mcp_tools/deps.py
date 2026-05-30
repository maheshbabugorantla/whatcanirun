"""M09 Slice L (step 2): runtime dependency loader for the MCP tools.

The cost-cells query layer wants 7 collections (gpu_prices,
llm_prices, gpu_catalog, model_catalog, quantizations, bench_cells,
aa_observations). Loading them on every tool call is acceptable
at v1 scale (~ms latency to read disk + parse YAML/Parquet/JSON);
M11 may add a module-level memoization layer if benchmarks
demonstrate the need.

This module centralizes the loading so the 4 numerical tool
wrappers (`fit_check`, `find_cheapest_deployment`,
`compare_deployment_modes`, `budget_to_plan`) share one
`load_runtime_deps()` entry point — divergence between them
would be a real bug (e.g. only one tool seeing a stale model
catalog).

`RuntimeDeps` is the typed bundle. Each field defaults to an
empty collection so the tools can still answer (Case 3 /
UnknownModelResponse) when a cache hasn't been warmed yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.benchmark_cells_loader import load_benchmark_cells
from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.loaders import (
    load_quantizations,
    load_tracked_models,
    load_workload_profiles,
)
from whatcanirun.catalog.seed_schemas import Quantization, TrackedModelRow
from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.paths import SEEDS_DIR, USER_CACHE_DIR, USER_CONFIG_DIR
from whatcanirun.pricing.computeprices import (
    ComputePricesClient,
    ComputePricesUnavailable,
)
from whatcanirun.pricing.projections import (
    GpuCatalogRow,
    GpuPriceRow,
    LlmCatalogRow,
    LlmPriceRow,
)


@dataclass(frozen=True)
class RuntimeDeps:
    """Frozen snapshot of every cache the numerical MCP tools
    depend on at one tool-call instant. Constructed by
    `load_runtime_deps()`; tests construct it directly with the
    relevant subset populated."""

    gpu_prices: list[GpuPriceRow] = field(default_factory=list)
    llm_prices: list[LlmPriceRow] = field(default_factory=list)
    gpu_catalog: list[GpuCatalogRow] = field(default_factory=list)
    llm_catalog: list[LlmCatalogRow] = field(default_factory=list)
    model_catalog: list[Model] = field(default_factory=list)
    quantizations: list[Quantization] = field(default_factory=list)
    workload_profiles: list[WorkloadProfile] = field(default_factory=list)
    bench_cells: list[BenchmarkCell] = field(default_factory=list)
    tracked_models: list[TrackedModelRow] = field(default_factory=list)


def _load_hf_model_cache(cache_dir: Path) -> list[Model]:
    """Read every `*.model.json` file under `<cache_dir>/huggingface/`
    and project each to a `Model`. Missing dir or empty dir returns
    an empty list — the dispatcher treats that as "Case 3 for any
    model the user asks about" until a sync warms the cache."""
    hf_dir = cache_dir / "huggingface"
    if not hf_dir.is_dir():
        return []
    models: list[Model] = []
    for path in sorted(hf_dir.glob("*.model.json")):
        try:
            data = json.loads(path.read_text())
            models.append(Model.model_validate(data))
        except (json.JSONDecodeError, ValueError):
            # A corrupted cache file shouldn't take out the whole
            # tool — skip and move on. M11 can add a stderr warning
            # if this becomes common.
            continue
    return models


def _load_merged_tracked_models(
    *,
    seeds_dir: Path,
    config_dir: Path,
) -> list[TrackedModelRow]:
    """Per spec/M09 § 'The merged tracked-models set', the tracked
    set is the UNION of `seeds/tracked_models.yaml` (project-
    controlled) + `<config_dir>/user_models.yaml` (per-user, written
    by `resolve_model`). The user-extension file may not exist
    (no resolve_model calls yet) — that's expected.

    Project-controlled rows win in case of slug collisions: a user
    can't accidentally shadow a curated tracked model. M03's
    `sync_all_tracked` will gain the same merged-loader contract;
    until then this is M09's local merge.
    """
    project_rows = load_tracked_models(seeds_dir / "tracked_models.yaml")
    user_yaml = config_dir / "user_models.yaml"
    if not user_yaml.exists():
        return project_rows
    raw = yaml.safe_load(user_yaml.read_text()) or []
    if not isinstance(raw, list):
        return project_rows
    user_rows: list[TrackedModelRow] = []
    project_slugs = {r.slug for r in project_rows}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if not isinstance(slug, str) or slug in project_slugs:
            # Skip duplicates (project rows win) and malformed entries.
            continue
        try:
            user_rows.append(TrackedModelRow.model_validate(entry))
        except ValueError:
            # Reject malformed user rows silently — the alternative
            # is failing every tool call until the user fixes their
            # YAML. Better to drop the bad row and keep the rest
            # of the system functional. M11 may surface this via
            # stderr.
            continue
    return [*project_rows, *user_rows]


async def load_runtime_deps(
    *,
    seeds_dir: Path | None = None,
    cache_dir: Path | None = None,
    config_dir: Path | None = None,
) -> RuntimeDeps:
    """Load every cache the numerical MCP tools need into a
    `RuntimeDeps` snapshot. Missing caches degrade to empty
    collections — the tools still answer (Case 3 paths kick in)
    rather than failing the call.

    `seeds_dir` / `cache_dir` / `config_dir` default to the XDG
    paths from `whatcanirun.paths`. Tests inject temp dirs so they
    can construct a deterministic small-cache scenario."""
    seeds_dir = seeds_dir or SEEDS_DIR
    cache_dir = cache_dir or USER_CACHE_DIR
    config_dir = config_dir or USER_CONFIG_DIR

    # Seed-backed loads (always present in a normal install).
    quantizations = load_quantizations(seeds_dir / "quantizations.yaml")
    workload_profiles = load_workload_profiles(seeds_dir / "workload_profiles.yaml")
    bench_cells_path = seeds_dir / "benchmark_cells.parquet"
    bench_cells = load_benchmark_cells(bench_cells_path) if bench_cells_path.exists() else []

    # CP cache reads — graceful degradation on unreachable cache.
    cp_cache = cache_dir / "computeprices"
    cp_cache.mkdir(parents=True, exist_ok=True)
    cp_client = ComputePricesClient(cache_dir=cp_cache)

    async def _fetch_or_empty[T](fn: Any) -> list[T]:
        """Call the CP endpoint; degrade to empty list on
        ComputePricesUnavailable so a cold cache + no-network
        environment doesn't take down every tool call."""
        try:
            return await fn()  # type: ignore[no-any-return]
        except ComputePricesUnavailable:
            return []

    gpu_prices: list[GpuPriceRow] = await _fetch_or_empty(cp_client.get_gpu_prices)
    llm_prices: list[LlmPriceRow] = await _fetch_or_empty(cp_client.get_llm_prices)
    gpu_catalog: list[GpuCatalogRow] = await _fetch_or_empty(cp_client.get_gpu_catalog)
    llm_catalog: list[LlmCatalogRow] = await _fetch_or_empty(cp_client.get_llm_catalog)

    model_catalog = _load_hf_model_cache(cache_dir)
    tracked_models = _load_merged_tracked_models(seeds_dir=seeds_dir, config_dir=config_dir)

    return RuntimeDeps(
        gpu_prices=gpu_prices,
        llm_prices=llm_prices,
        gpu_catalog=gpu_catalog,
        llm_catalog=llm_catalog,
        model_catalog=model_catalog,
        quantizations=quantizations,
        workload_profiles=workload_profiles,
        bench_cells=bench_cells,
        tracked_models=tracked_models,
    )
