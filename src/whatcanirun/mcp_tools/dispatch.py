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

1. Validates slug + repo_id (delegated to HfModelSync.sync_model's
   regex check — same vector surface)
2. Atomically merges `(slug, hf_repo_id)` into
   `<config_dir>/user_models.yaml`
3. Triggers HfModelSync.sync_model to fetch + cache the config
4. Returns ResolveModelResult with status + diagnostic

`ResolveModelResult` deliberately carries no trust_envelope per
spec/M09 § Public surface §6 — it's a status + diagnostic
response, not a numerical one.
"""

from __future__ import annotations

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
    from whatcanirun.catalog.hf_sync import HfModelSync

    config_dir = config_dir or USER_CONFIG_DIR
    cache_dir = cache_dir or USER_CACHE_DIR

    yaml_path = config_dir / "user_models.yaml"
    _merge_user_yaml_row(yaml_path, slug=model_slug, hf_repo_id=hf_repo_id)

    sync = HfModelSync(cache_dir=cache_dir)
    try:
        model = await sync.sync_model(slug=model_slug, repo_id=hf_repo_id)
    except Exception as exc:  # broad: network errors, 404s, timeout, etc.
        # The HF sync failure modes are diverse (httpx.HTTPStatusError
        # for 404 / 5xx, httpx.ConnectError, ValueError for
        # malformed slugs). We collapse them to a single
        # `not_found_on_hf` status for the user-facing response
        # because the user's recourse is the same in every case:
        # check the repo_id is correct + public, retry.
        return ResolveModelResult(
            model_slug=model_slug,
            hf_repo_id=hf_repo_id,
            status="not_found_on_hf",
            hf_revision_sha=None,
            error_detail=f"{type(exc).__name__}: {exc}",
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


def find_model_in_catalog(model_slug: str, deps: RuntimeDeps) -> Model | None:
    """Case 1 lookup: is the model in the cached HF catalog?
    Returns the Model on match, None otherwise."""
    for model in deps.model_catalog:
        if model.slug == model_slug:
            return model
    return None


def find_in_cp_llm_catalog(model_slug: str, deps: RuntimeDeps) -> LlmCatalogRow | None:
    """Case 2 lookup: is the model in CP's hosted-API catalog
    (`/api/v1/llm-models`) even though we don't have HF
    architecture data for it? Returns the row on match, None
    otherwise."""
    for row in deps.llm_catalog:
        if row.slug == model_slug:
            return row
    return None


# Imports moved here to avoid circular imports; the type-only
# references above are forward-strings until this point.
from whatcanirun.catalog.hf_model import Model  # noqa: E402
from whatcanirun.mcp_tools.deps import RuntimeDeps  # noqa: E402
from whatcanirun.pricing.projections import LlmCatalogRow  # noqa: E402

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
