"""M09 Slice L: Unknown-model dispatcher + `resolve_model` tool.

Per spec/M09 § Unknown model handling, the dispatcher routes
model_slug requests through three cases:

- Case 1 — in merged tracked-models set, not cached locally →
  lazy-sync transparently via M03's HfModelSync.sync_model
- Case 2 — known to CP (catalog + prices), NOT in our tracked-
  models set → partial CostCell with `hosted_api_token` and
  `model_architecture=0.0` confidence
- Case 3 — in neither → return UnknownModelResponse so the LLM
  client can elicit `hf_repo_id` from the user, then call
  `resolve_model` to persist + sync

`resolve_model_to_user_yaml(model_slug, hf_repo_id, config_dir,
cache_dir)` is the persistence + sync primitive backing the
`resolve_model` MCP tool. It:

1. Validates slug + repo_id explicitly with the same regex
   patterns HfModelSync.sync_model enforces, BEFORE the sync call.
   Earlier revisions deferred to sync_model's own regex check,
   but the persistence-order fix (sync first, then write yaml)
   needs the validation result locally so a malformed input never
   reaches the network round-trip — and so we never persist a
   bad `(slug, repo_id)` row to `user_models.yaml`.
2. Calls HfModelSync.sync_model to fetch + cache the config.
3. ONLY on sync success, atomically merges `(slug, hf_repo_id)`
   into `<config_dir>/user_models.yaml`. A sync failure (network,
   HF 404, malformed config) leaves the yaml untouched so we
   never persist a row we can't actually serve.
4. Returns ResolveModelResult with status + diagnostic.

`ResolveModelResult` deliberately carries no trust_envelope per
spec/M09 § Public surface §6 — it's a status + diagnostic
response, not a numerical one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from whatcanirun.paths import USER_CACHE_DIR, USER_CONFIG_DIR

# The elicit_prompt text per spec/M09 § Case 3 — exact wording so
# the LLM client surfaces the same ask the spec audited.
_ELICIT_PROMPT = (
    "I don't have this model in my catalog yet. If you can share the "
    "Hugging Face repo_id (e.g. `meta-llama/Llama-3.3-70B-Instruct`), "
    "I'll fetch its config and add it for this and future requests. "
    "If the model isn't on a public Hugging Face repo, I won't be able "
    "to estimate fit or throughput for it."
)


class UnknownModelResponse(BaseModel):
    """Returned by tools when the model_slug is in NEITHER the
    tracked-models set nor the CP catalog. Carries the elicitation
    prompt the MCP client surfaces to the user; no trust envelope
    because there are no numbers to wrap (spec/M09 § Case 3)."""

    model_config = ConfigDict(extra="forbid")

    requested_model_slug: str
    status: Literal["unknown_model"] = "unknown_model"
    elicit_field: Literal["hf_repo_id"] = "hf_repo_id"
    elicit_prompt: str = _ELICIT_PROMPT
    suggested_followups: list[str] = Field(
        default_factory=lambda: [
            "list_catalog (to see what models are already supported)",
            "budget_to_plan with a publicly tracked model_slug",
        ]
    )


class ResolveModelResult(BaseModel):
    """Returned by `resolve_model`. Carries the status of the
    persistence + sync attempt; no trust envelope (the response
    is a status + diagnostic, not a numerical one — spec/M09 §
    Public surface §6)."""

    model_config = ConfigDict(extra="forbid")

    model_slug: str
    hf_repo_id: str
    status: Literal["resolved", "sync_failed", "not_found_on_hf"]
    hf_revision_sha: str | None = None
    error_detail: str | None = None


def _merge_user_yaml_row(yaml_path: Path, slug: str, hf_repo_id: str) -> None:
    """Atomic-ish update of user_models.yaml: load existing rows
    (or start fresh), replace any existing row for `slug` with the
    new repo_id, write the result via tmp + rename so a crash
    mid-write can't corrupt the file.

    The "atomic-ish" hedge: tmp+rename is atomic at the filesystem
    level for the rename step itself; if a different process is
    also writing user_models.yaml concurrently, the last writer
    wins (no file locking in v1, single-user MCP scope). That's
    acceptable for the stdio MCP scope; a v2 multi-user remote
    setup would need a lockfile or a real config store."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if yaml_path.exists():
        loaded = yaml.safe_load(yaml_path.read_text()) or []
        if isinstance(loaded, list):
            rows = [r for r in loaded if isinstance(r, dict)]
    # Drop any existing row with this slug so the new repo_id wins.
    rows = [r for r in rows if r.get("slug") != slug]
    rows.append({"slug": slug, "hf_repo_id": hf_repo_id})

    tmp = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(rows, sort_keys=False))
    tmp.replace(yaml_path)


async def resolve_model_to_user_yaml(
    model_slug: str,
    hf_repo_id: str,
    *,
    config_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> ResolveModelResult:
    """Persist (model_slug, hf_repo_id) to user_models.yaml +
    trigger HfModelSync.sync_model for the new pair.

    `config_dir` / `cache_dir` default to the XDG locations from
    `whatcanirun.paths`. Tests inject temp paths so the suite
    doesn't touch the user's actual `~/.config` or `~/.cache`.
    """
    # Import locally so test-time `monkeypatch.setattr` on the
    # `HfModelSync.sync_model` class method takes effect — a
    # module-top-level import binds at module load time, before
    # the test's monkeypatch fixture runs.
    from whatcanirun.catalog.hf_sync import (
        _SAFE_REPO_ID_RE,
        _SAFE_SLUG_RE,
        HfModelSync,
    )

    config_dir = config_dir or USER_CONFIG_DIR
    cache_dir = cache_dir or USER_CACHE_DIR

    # Validate identifiers BEFORE writing user_models.yaml. The
    # sync_model call would catch these via the same regex check,
    # but at that point a malformed row has already been
    # persisted — and once persisted it travels through every
    # subsequent dispatch_model_request as a permanently-failing
    # Case 1b candidate. Validate at the boundary so the config
    # file can't accumulate un-syncable entries.
    if not _SAFE_SLUG_RE.match(model_slug):
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="not_found_on_hf",
            hf_revision_sha=None,
            error_detail=(
                f"invalid slug {model_slug!r}: must match "
                f"{_SAFE_SLUG_RE.pattern} "
                "(lowercase alphanumerics + `._-`, no path separators)"
            ),
        )
    if not _SAFE_REPO_ID_RE.match(hf_repo_id):
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="not_found_on_hf",
            hf_revision_sha=None,
            error_detail=(
                f"invalid repo_id {hf_repo_id!r}: must match HF's "
                f"<namespace>/<name> format ({_SAFE_REPO_ID_RE.pattern})"
            ),
        )

    # Sync FIRST, persist AFTER. The previous order persisted the
    # (slug, hf_repo_id) row to user_models.yaml BEFORE confirming
    # the sync — so a 404 / 5xx / network failure would leave a
    # permanent un-syncable row in the user's config. That row
    # would then re-trigger Case 1b on every subsequent tool call
    # AND advertise the model as "supported" in list_catalog,
    # neither of which is true. Defer the write to the success
    # branch so a sync failure leaves the file untouched.
    sync = HfModelSync(cache_dir=cache_dir)
    import httpx

    sync_status: Literal["sync_failed", "not_found_on_hf"]
    try:
        model = await sync.sync_model(slug=model_slug, repo_id=hf_repo_id)
    except asyncio.CancelledError:
        # Propagate cancellation per spec/M09 § ADR-013 — the user
        # disconnect signal must reach the asyncio runtime, not get
        # swallowed by the broad `except Exception` below. No yaml
        # write because we're abandoning the resolve attempt.
        raise
    except httpx.HTTPStatusError as exc:
        # 404 means the repo_id is wrong or private; the user's
        # recourse is to fix the repo_id. 5xx means HF is having a
        # bad day and a retry would likely succeed — a different
        # message for a different recourse.
        sync_status = "not_found_on_hf" if exc.response.status_code == 404 else "sync_failed"
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status=sync_status,
            hf_revision_sha=None,
            error_detail=f"HTTP {exc.response.status_code}: {exc}",
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        # Network-layer failures are transient — a retry might
        # succeed. Surface them as `sync_failed` so the LLM client
        # can offer "try again later" rather than the misleading
        # "check the repo_id" the `not_found_on_hf` framing implies.
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="sync_failed",
            hf_revision_sha=None,
            error_detail=f"{type(exc).__name__}: {exc}",
        )
    except ValueError as exc:
        # `HfModelSync.sync_model` raises ValueError when the regex
        # rejects a malformed slug or repo_id. The user's recourse
        # is the same as 404 — the supplied identifier was wrong.
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="not_found_on_hf",
            hf_revision_sha=None,
            error_detail=f"ValueError: {exc}",
        )
    except Exception as exc:
        # Unrecognized failure mode — surface as `sync_failed` (the
        # generic transient bucket) rather than `not_found_on_hf`
        # which would falsely accuse the user's repo_id.
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="sync_failed",
            hf_revision_sha=None,
            error_detail=f"{type(exc).__name__}: {exc}",
        )

    # Sync succeeded — NOW persist the (slug, hf_repo_id) row so
    # future Case 1b lazy-syncs can find it. If the write itself
    # fails (disk full, permission denied), we surface that rather
    # than returning "resolved" with a phantom persistence claim.
    yaml_path = config_dir / "user_models.yaml"
    try:
        _merge_user_yaml_row(yaml_path, slug=model_slug, hf_repo_id=hf_repo_id)
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        # Persistence failed AFTER a successful HF sync. The full
        # failure surface here:
        #   - OSError — disk full, permission denied, atomic-rename
        #     hit a locked file
        #   - yaml.YAMLError — `_merge_user_yaml_row` parses any
        #     existing file before merging; a malformed
        #     user_models.yaml on disk would raise here even
        #     though sync just succeeded
        #   - UnicodeDecodeError — `.read_text()` on a file with
        #     binary garbage in it
        # All three collapse to the same recourse: surface as
        # sync_failed with descriptive error_detail. Keep
        # status="sync_failed" to stay within the documented
        # 3-status enum, but null the hf_revision_sha so the
        # {status, sha} pair stays internally consistent — a
        # non-null SHA paired with status="sync_failed" would have
        # an LLM client trying to retry sync when the real recourse
        # is "check / repair user_models.yaml". The error_detail
        # captures the actual cause so the client can still relay
        # it accurately. A v2 schema bump could introduce a
        # dedicated `persistence_failed` status.
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="sync_failed",
            hf_revision_sha=None,
            error_detail=(
                f"sync succeeded (sha={model.hf_revision_sha}) but persisting "
                f"to {yaml_path} failed: {type(exc).__name__}: {exc}"
            ),
        )

    return ResolveModelResult(
        model_slug=model_slug,
        hf_repo_id=hf_repo_id,
        status="resolved",
        hf_revision_sha=model.hf_revision_sha,
        error_detail=None,
    )


async def resolve_model(
    model_slug: str,
    hf_repo_id: str,
) -> ResolveModelResult:
    """`resolve_model` MCP tool entry point.

    Thin wrapper over `resolve_model_to_user_yaml` using the
    default XDG paths. The split lets tests inject temp dirs
    without monkeypatching module-level state."""
    return await resolve_model_to_user_yaml(model_slug, hf_repo_id)


# ============================================================ Case dispatcher
# Routes model_slug through Case 1 / Case 2 / Case 3 per spec/M09
# § Unknown model handling.


# Local-scope imports (kept below the response types) to avoid
# circular init order with `mcp_tools.deps`, which itself imports
# Pydantic shapes from `pricing.projections`.
from dataclasses import dataclass  # noqa: E402

from whatcanirun.catalog.hf_model import Model  # noqa: E402
from whatcanirun.catalog.seed_schemas import TrackedModelRow  # noqa: E402
from whatcanirun.mcp_tools.deps import RuntimeDeps  # noqa: E402
from whatcanirun.pricing.projections import LlmCatalogRow, LlmPriceRow  # noqa: E402


def find_model_in_catalog(model_slug: str, deps: RuntimeDeps) -> Model | None:
    """Case 1a lookup: is the model in the cached HF catalog?
    Returns the Model on match, None otherwise. No I/O — just a
    list scan."""
    for model in deps.model_catalog:
        if model.slug == model_slug:
            return model
    return None


def find_tracked_row(model_slug: str, deps: RuntimeDeps) -> TrackedModelRow | None:
    """Case 1b precondition: is the slug in the merged tracked-
    models set (seeds + user_models.yaml)? If yes, the dispatcher
    knows the slug→repo_id mapping and can lazy-sync via HF."""
    for row in deps.tracked_models:
        if row.slug == model_slug:
            return row
    return None


def find_in_cp_llm_catalog(model_slug: str, deps: RuntimeDeps) -> LlmCatalogRow | None:
    """Case 2 lookup: is the model in CP's hosted-API catalog
    (`/api/v1/llm-models`) even though we don't have HF
    architecture data for it?"""
    for row in deps.llm_catalog:
        if row.slug == model_slug:
            return row
    return None


def find_cp_llm_prices(model_slug: str, deps: RuntimeDeps) -> list[LlmPriceRow]:
    """Case 2 supplement: the per-provider LlmPriceRows for the
    slug. The Case 2 partial-cell constructor builds one CostCell
    per row."""
    return [row for row in deps.llm_prices if row.model_slug == model_slug]


@dataclass(frozen=True)
class Case1Resolved:
    """The model is locally available (either was cached or just
    got lazy-synced). The tool proceeds with `model` exactly as
    if Case 1a had been a cache hit."""

    model: Model


def model_catalog_with_resolved(deps: RuntimeDeps, resolved_model: Model) -> list[Model]:
    """Return a `model_catalog` list guaranteed to contain the
    dispatcher-resolved model — and ONLY that model under its
    slug (no duplicate row).

    After Case 1b lazy-sync, `deps.model_catalog` is stale: the
    snapshot was loaded BEFORE `dispatch_model_request` ran, so
    the freshly-synced Model isn't there. Passing the stale
    `deps.model_catalog` into `query_cost_cells` would make the
    just-synced model invisible — the filter would drop it and
    the tool would return zero cells even though sync succeeded.

    The fix is to splice the resolved model in (and dedupe by
    slug so Case 1a cache-hits don't end up with two rows for the
    same slug). All four numerical tool wrappers that call
    `query_cost_cells` after dispatch should use this helper."""
    other = [m for m in deps.model_catalog if m.slug != resolved_model.slug]
    return [resolved_model, *other]


@dataclass(frozen=True)
class Case2HostedOnly:
    """The model is in CP's hosted-API catalog but not in our
    tracked-models set. Tools that support partial cells
    (`find_cheapest_deployment`, `budget_to_plan`) build hosted-
    only CostCells from `prices`; tools that don't (`fit_check`,
    `compare_deployment_modes`) collapse this to Case 3 per
    spec/M09 § Tool-by-tool Case 2 behavior."""

    catalog_row: LlmCatalogRow
    prices: list[LlmPriceRow]


# Discriminated union — the dispatcher returns exactly one of the
# three cases. Tool wrappers branch on `isinstance` to pick the
# right path; the unknown case is the same `UnknownModelResponse`
# that travels back to the LLM client unchanged.
DispatchResult = Case1Resolved | Case2HostedOnly | UnknownModelResponse


async def dispatch_model_request(
    model_slug: str,
    deps: RuntimeDeps,
    *,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> DispatchResult:
    """Spec/M09 § Unknown model handling — the three-case router
    every numerical tool runs first.

    1. Case 1a — model in `deps.model_catalog` (HF cache hit) →
       `Case1Resolved` immediately
    2. Case 1b — slug in `deps.tracked_models` but not yet cached
       → lazy-sync via `HfModelSync.sync_model`, then
       `Case1Resolved` with the freshly-synced Model
    3. Case 2 — slug in CP's `llm_catalog` (and at least one
       matching `llm_prices` row) → `Case2HostedOnly`
    4. Case 3 — none of the above → `UnknownModelResponse`

    If lazy-sync (case 1b) raises (network, 404, HF unreachable),
    we DON'T crash — we fall through to the Case 2 / Case 3
    checks. A CP-only fallback may still answer the user's
    question; a genuine unknown still gets a clean elicitation
    at the next turn. The user always sees a structured response,
    never a raw exception."""
    # Case 1a — cached HF model, fastest path.
    model = find_model_in_catalog(model_slug, deps)
    if model is not None:
        return Case1Resolved(model=model)

    # Case 1b — tracked but not cached, lazy-sync.
    tracked = find_tracked_row(model_slug, deps)
    if tracked is not None:
        # Local import: keeps test-time `monkeypatch.setattr` on
        # the `HfModelSync.sync_model` class method effective. The
        # existing `resolve_model_to_user_yaml` uses the same
        # pattern for the same reason.
        from whatcanirun.catalog.hf_sync import HfModelSync

        cache_dir = cache_dir or USER_CACHE_DIR
        sync = HfModelSync(cache_dir=cache_dir, hf_token=hf_token)
        try:
            model = await sync.sync_model(
                slug=tracked.slug,
                repo_id=tracked.hf_repo_id,
                display_name=tracked.display_name,
                total_params_b=tracked.total_params_b,
                active_params_b=tracked.active_params_b,
                kv_cache_strategy_override=tracked.kv_cache_strategy_override,
            )
            return Case1Resolved(model=model)
        except asyncio.CancelledError:
            # NEVER swallow cancellation — the FastMCP runtime
            # cancels in-flight handlers on client disconnect;
            # silently falling through would have the dispatcher
            # keep doing work the client no longer wants. Re-raise
            # so the cancellation propagates the way asyncio
            # expects.
            raise
        except Exception:
            # Other lazy-sync failures (network, 404, HF
            # unreachable). Fall through to Case 2 / Case 3 checks
            # rather than raising — the user might still get a
            # useful Case 2 answer, or at worst a clean elicitation.
            pass

    # Case 2 — in CP llm catalog with at least one price row.
    cp_row = find_in_cp_llm_catalog(model_slug, deps)
    if cp_row is not None:
        prices = find_cp_llm_prices(model_slug, deps)
        if prices:
            return Case2HostedOnly(catalog_row=cp_row, prices=prices)

    # Case 3 — genuinely unknown.
    return UnknownModelResponse(requested_model_slug=model_slug)


# ============================================================ Workload elicit
# Slice M: when `budget_to_plan` is called without
# `workload_profile_slug`, the spec rejects silent defaults and
# instead elicits the profile via WorkloadElicitationResponse.


_WORKLOAD_ELICIT_PROMPT = (
    "To estimate prompt counts for your budget, I need to know what kind "
    "of workload these prompts represent. Pick one:\n"
    "- code_completion: short prompts (~100 in, ~50 out)\n"
    "- chat_assistant: medium prompts (~500 in, ~200 out)\n"
    "- batch_eval:     long prompts (~2000 in, ~100 out)\n"
    "If none of those fit, ask me for `find_cheapest_deployment` instead - "
    "it returns $/M figures so you can do the math against your own "
    "token distribution."
)


class WorkloadElicitationResponse(BaseModel):
    """Returned by `budget_to_plan` when `workload_profile_slug`
    is omitted. Per spec/M09 § Workload assumption handling, the
    server elicits the profile rather than silently defaulting -
    a default would set `workload_assumption=0.2` and drag the
    top-level confidence to 0.2 anyway, so eliciting up-front
    is the same answer expressed in the API surface.

    No trust_envelope (elicitation, no numbers to wrap)."""

    model_config = ConfigDict(extra="forbid")

    requested_model_slug: str
    status: Literal["workload_required"] = "workload_required"
    elicit_field: Literal["workload_profile_slug"] = "workload_profile_slug"
    elicit_prompt: str = _WORKLOAD_ELICIT_PROMPT
    available_profiles: list[str] = Field(
        default_factory=lambda: ["code_completion", "chat_assistant", "batch_eval"]
    )
    suggested_followups: list[str] = Field(
        default_factory=lambda: [
            "budget_to_plan with workload_profile_slug='chat_assistant' for a starting estimate",
            "find_cheapest_deployment (returns $/M figures, no prompt-count synthesis)",
        ]
    )
