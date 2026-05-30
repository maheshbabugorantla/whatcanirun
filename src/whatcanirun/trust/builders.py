"""Per-tool `TrustEnvelope` builders.

One function per tool — keeps the envelope-construction logic
out of the tool body so the tool stays a thin coordination layer
between catalog lookups and pure-math kernels. Centralizing the
construction also enforces uniformity: a domain a tool's response
depends on always appears in `confidence_breakdown`, with a value
derived consistently from the calibration table.

Per spec/M09 § Common pitfalls #1: "Every tool builds its own
envelope. If one forgets a domain, the rollup is wrong." This
module is where that's enforced.
"""

from __future__ import annotations

import datetime as dt

from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.workload import WorkloadProfile
from whatcanirun.inference.fit_check import FitResult
from whatcanirun.inference.tps_estimator import TpsEstimate
from whatcanirun.plan.cost_cells import CostCell
from whatcanirun.pricing.projections import GpuCatalogRow, LlmCatalogRow, LlmPriceRow
from whatcanirun.trust.calibration import (
    combine_freshness,
    fit_check_methodology_confidence,
    freshness_confidence,
)
from whatcanirun.trust.envelope import (
    ConfidenceDomain,
    Source,
    SourceName,
    TrustEnvelope,
)

# Per spec/SHARED.md § Calibration: a workload profile supplied via
# tool argument (the typical case after Slice M elicitation) scores
# 0.95. User-supplied custom token counts would score 1.0; a silent
# default would score 0.2 — which M09 avoids by eliciting.
_WORKLOAD_TOOL_ARG_CONFIDENCE = 0.95

# Verbatim caveat the LLM client should relay whenever a response
# carries a workload-derived count (est_total_prompts,
# cost_per_prompt). Pulled into a module constant so identical text
# travels with every workload-conditioned response.
_WORKLOAD_CAVEAT = (
    "est_total_prompts and cost_per_prompt are conditioned on the named "
    "workload_profile. Real traffic with a different (avg_input_tokens, "
    "avg_output_tokens) shape will yield different counts."
)

# Spec/M09 § Case 2: the exact caveat text that travels with every
# Case 2 partial CostCell. The verbatim wording is part of the
# trust contract — softening or paraphrasing would let a CP-only
# answer look more complete than it is.
_CASE_2_CAVEAT = (
    "Architecture data not available for this model — only hosted-API "
    "pricing is shown. Fit-check and self-hosted throughput are not "
    "estimated. To enable full analysis, add an entry to your local "
    "`~/.config/whatcanirun/user_models.yaml` with the model's "
    "Hugging Face `repo_id`."
)

# Refusal reason populated on every Case 2 cell's TpsEstimate so the
# LLM client (per spec/M09 relay rule 2) can explain why no throughput
# number came back.
_CASE_2_REFUSAL_REASON = (
    "Architecture data unavailable for CP-only model; throughput "
    "requires HF config.json to compute single-stream bandwidth bound."
)


def build_fit_check_envelope(
    *,
    fit_result: FitResult,
    model: Model,
    gpu: GpuCatalogRow,
    tp_size: int,
    batch_size: int,
    context_length: int,
    now: dt.datetime,
    gpu_specs_last_updated: dt.datetime,
) -> TrustEnvelope:
    """Build the trust envelope for the `fit_check` tool.

    Confidence domains:
      - `fit_check`         — methodology confidence (constant per
                              `fit_check_methodology_confidence()`)
      - `model_architecture`— HF freshness for the config.json
                              this Model projection was built from
      - `gpu_specs`         — CP freshness for the GpuCatalogRow
                              the VRAM number came from
      - `freshness`         — min over (HF, CP) ages — the
                              weakest-link rollup that the LLM
                              client surfaces as "data this old"

    `workload_assumption` is OMITTED — fit_check doesn't synthesize
    a derived prompt count, so per spec/SHARED.md the key must not
    appear in the breakdown.

    `assumptions` echoes the op-point parameters so a client UI
    surfacing the response can let the user re-issue with different
    values; the FitResult.assumptions echoes the FORMULA constants
    (overhead_pct etc.) and the envelope.assumptions echoes the
    USER-supplied constants (tp_size, batch, ctx). Two different
    kinds of "what was held fixed" both worth surfacing.
    """
    hf_age = now - model.last_synced_at
    cp_age = now - gpu_specs_last_updated

    sources = [
        Source(
            name="huggingface",
            detail=f"model.config.json for {model.hf_repo_id} @ {model.hf_revision_sha}",
            last_updated=model.last_synced_at,
        ),
        Source(
            name="computeprices",
            detail=f"gpu catalog entry for {gpu.slug} (vram_gb={gpu.vram_gb})",
            last_updated=gpu_specs_last_updated,
        ),
    ]

    breakdown: dict[ConfidenceDomain, float] = {
        "fit_check": fit_check_methodology_confidence(),
        "model_architecture": freshness_confidence("huggingface", hf_age),
        "gpu_specs": freshness_confidence("computeprices", cp_age),
        "freshness": combine_freshness(
            [("huggingface", hf_age), ("computeprices", cp_age)],
        ),
    }

    # Carry the FitResult.sufficiency_caveat verbatim in the
    # envelope so a client that relays caveats without
    # dereferencing the embedded FitResult still surfaces the
    # disclaimer. spec/M09 relay rule #3.
    caveats = [fit_result.sufficiency_caveat]

    return TrustEnvelope(
        sources=sources,
        confidence_breakdown=breakdown,
        assumptions={
            "tp_size": tp_size,
            "batch_size": batch_size,
            "context_length": context_length,
        },
        caveats=caveats,
        freshness={
            "huggingface": model.last_synced_at,
            "computeprices": gpu_specs_last_updated,
        },
        verify_links=[
            f"https://huggingface.co/{model.hf_repo_id}",
            "https://www.computeprices.com",
        ],
    )


# ============================================================ shared helpers


def _merge_sources(*cells: CostCell | None) -> list[Source]:
    """Dedup `Source` entries across input CostCells by
    `(name, detail)` so the wrapping envelope doesn't duplicate
    the same upstream contribution. Insertion order preserved
    (first-seen wins)."""
    seen: dict[tuple[SourceName, str], Source] = {}
    for cell in cells:
        if cell is None:
            continue
        for src in cell.trust_envelope.sources:
            seen.setdefault((src.name, src.detail), src)
    return list(seen.values())


def _merge_freshness(*cells: CostCell | None) -> dict[str, dt.datetime]:
    """Min-by-source freshness rollup across input cells. If two
    cells share a source, the OLDER timestamp wins (weakest-link
    semantics propagate to the freshness map)."""
    merged: dict[str, dt.datetime] = {}
    for cell in cells:
        if cell is None:
            continue
        for source_name, ts in cell.trust_envelope.freshness.items():
            if source_name not in merged or ts < merged[source_name]:
                merged[source_name] = ts
    return merged


def _merge_verify_links(*cells: CostCell | None) -> list[str]:
    """Union of `verify_links` across input cells, dedup'd while
    preserving insertion order. Per spec/SHARED.md the wrapping
    envelope MUST surface the audit URLs from every cell that
    contributed; an LLM client reading only the wrapping envelope
    needs them to render a "verify this" hint to the user."""
    seen: dict[str, None] = {}
    for cell in cells:
        if cell is None:
            continue
        for link in cell.trust_envelope.verify_links:
            seen.setdefault(link, None)
    return list(seen.keys())


# ============================================================ budget_to_plan


def build_budget_plan_envelope(
    *,
    cell: CostCell,
    workload: WorkloadProfile,
    budget_usd: float,
) -> TrustEnvelope:
    """Build the trust envelope for one `BudgetPlanRow`. The
    envelope carries forward the underlying CostCell's breakdown,
    sources, freshness, caveats, and verify_links, then adds the
    workload-derived `workload_assumption` domain + a verbatim
    caveat naming the assumed workload shape.

    Per spec/M09 relay rule 6, the LLM client surfaces
    `assumptions["workload_profile"]` whenever
    `workload_assumption` is present — so we MUST populate it
    here. Without it, the rule never triggers and a derived
    prompt count travels to the user without disclosure."""
    breakdown: dict[ConfidenceDomain, float] = dict(cell.trust_envelope.confidence_breakdown)
    breakdown["workload_assumption"] = _WORKLOAD_TOOL_ARG_CONFIDENCE

    return TrustEnvelope(
        sources=list(cell.trust_envelope.sources),
        confidence_breakdown=breakdown,
        assumptions={
            **cell.trust_envelope.assumptions,
            "workload_profile": workload.slug,
            "avg_input_tokens": workload.avg_input_tokens,
            "avg_output_tokens": workload.avg_output_tokens,
            "budget_usd": budget_usd,
        },
        caveats=[*cell.trust_envelope.caveats, _WORKLOAD_CAVEAT],
        freshness=dict(cell.trust_envelope.freshness),
        verify_links=list(cell.trust_envelope.verify_links),
    )


# ============================================================ comparison


# ============================================================ Case 2 partial


def _build_case_2_envelope(
    *,
    price: LlmPriceRow,
    generated_at: dt.datetime,
) -> TrustEnvelope:
    """Build the trust envelope for one Case 2 partial CostCell.

    Per spec/M09 § Case 2, the envelope carries:
    - `model_architecture` = 0.0 (we have NO architecture data)
    - `pricing` derived from CP freshness
    - `freshness` derived from CP `meta.generated_at`
    - the verbatim Case 2 caveat
    - the default availability caveat is on the CostCell itself

    `assumptions` includes the LLM hosted-API pricing tier (CP's
    `pricing_type`) — spec/M09 calls out that the hosted-API tier
    can't ride in the CostCell's `pricing_type` field (which is
    the GPU `on_demand|spot` enum), so it travels here instead."""
    # `generated_at` is always a `datetime` (RuntimeDeps default is
    # `datetime.min` with UTC tzinfo — a deliberate sentinel for the
    # cold-cache / CP-unreachable case). No None-guard needed; the
    # cold-cache subtraction yields a multi-millennium timedelta
    # which `freshness_confidence` correctly maps to the lowest band.
    age = dt.datetime.now(dt.UTC) - generated_at
    pricing_score = freshness_confidence("computeprices", age)

    # `throughput=0.0` matches M08's `_partial_envelope_for_hosted_api`
    # shape — the Case 2 CostCell carries `tps_estimate.confidence=0.0`
    # (requires_measurement), so the envelope must surface that as
    # the `throughput` domain. Without it, spec/M09 relay rule #2
    # ("when confidence_breakdown.throughput == 0.0, the server is
    # refusing to estimate that combination") can't fire uniformly
    # across hosted-API CostCell sources — M08-built ones would
    # carry the signal, M09 Case 2 ones would silently swallow it.
    breakdown: dict[ConfidenceDomain, float] = {
        "model_architecture": 0.0,  # we have NO architecture data
        "throughput": 0.0,  # requires_measurement, matches tps_estimate.confidence
        "pricing": pricing_score,
        "freshness": pricing_score,
    }

    # Anchor BOTH `Source.last_updated` and the freshness map on
    # the same `generated_at` that drove `pricing_score`. Using
    # `price.last_updated` here (the per-row CP timestamp) would
    # let the reported staleness disagree with the confidence
    # score the LLM client surfaces — a per-row timestamp can be
    # weeks old even when the snapshot itself is hours fresh, or
    # vice versa. Anchoring on the snapshot timestamp keeps the
    # two signals consistent.
    return TrustEnvelope(
        sources=[
            Source(
                name="computeprices",
                detail=(
                    f"llm-prices entry for model_slug={price.model_slug}, "
                    f"provider_slug={price.provider_slug}"
                ),
                last_updated=generated_at,
            )
        ],
        confidence_breakdown=breakdown,
        assumptions={"llm_pricing_tier": price.pricing_type},
        caveats=[_CASE_2_CAVEAT],
        freshness={"computeprices": generated_at},
        verify_links=["https://www.computeprices.com/api/v1/llm-prices"],
    )


def build_case_2_partial_cells(
    *,
    model_slug: str,
    catalog_row: LlmCatalogRow,
    prices: list[LlmPriceRow],
    batch_size: int,
    context_length: int,
    llm_prices_generated_at: dt.datetime,
) -> list[CostCell]:
    """Build hosted_api_token-only CostCells from CP llm_prices
    for a model that's CP-known but not in our tracked-models set
    (spec/M09 § Case 2). One CostCell per `LlmPriceRow` (each row
    is one provider's quote for the same model).

    Per spec/M09 § Case 2 field map:
    - `gpu_slug`, `quant_slug`, `tp_size`, `hourly_usd`,
      `pricing_type`, `decode_tps`, `fit_result`,
      `cost_per_m_output_usd_self_hosted` all `None`
    - `price_per_m_input_usd` / `price_per_m_output_usd` from the
      CP price row
    - `tps_estimate.source = "requires_measurement"`,
      `confidence = 0.0`, `refusal_reason` cites the architecture gap
    - `availability_modeled = False`
    - `trust_envelope.confidence_breakdown["model_architecture"] = 0.0`
    - `trust_envelope.caveats` carries the verbatim Case 2 prose
    """
    _ = catalog_row  # echoed back to callers via deps; not in CostCell
    cells: list[CostCell] = []
    for price in prices:
        cells.append(
            CostCell(
                gpu_slug=None,
                provider_slug=price.provider_slug,
                model_slug=model_slug,
                quant_slug=None,
                tp_size=None,
                batch_size=batch_size,
                context_length=context_length,
                deployment_mode="hosted_api_token",
                hourly_usd=None,
                pricing_type=None,
                price_per_m_input_usd=price.price_per_1m_input_usd,
                price_per_m_output_usd=price.price_per_1m_output_usd,
                decode_tps=None,
                tps_estimate=TpsEstimate(
                    value=None,
                    source="requires_measurement",
                    confidence=0.0,
                    refusal_reason=_CASE_2_REFUSAL_REASON,
                ),
                fit_result=None,
                cost_per_m_output_usd_self_hosted=None,
                availability_modeled=False,
                trust_envelope=_build_case_2_envelope(
                    price=price, generated_at=llm_prices_generated_at
                ),
            )
        )
    return cells


def build_deployment_comparison_envelope(
    *,
    cloud_cell: CostCell | None,
    hosted_cell: CostCell | None,
    workload: WorkloadProfile,
) -> TrustEnvelope:
    """Build the trust envelope for a `DeploymentComparison`. The
    envelope wraps the synthesized per-prompt-cost comparison, so
    it adds `workload_assumption` (the per-prompt figure is
    workload-derived) and carries the weakest-link rollup of every
    domain that appeared in either side's envelope.

    `verify_links` are merged across both component cells via
    `_merge_verify_links` — an LLM client reading the wrapping
    envelope needs the audit URLs from each provider's CostCell.
    Dropping them would leave the client without an audit path
    for the per-prompt verdict it surfaces."""
    breakdown: dict[ConfidenceDomain, float] = {
        "workload_assumption": _WORKLOAD_TOOL_ARG_CONFIDENCE,
    }
    # Carry forward the WORST per-side confidence per shared
    # domain so the wrapping envelope inherits the weakest link.
    for cell in (cloud_cell, hosted_cell):
        if cell is None:
            continue
        for domain, score in cell.trust_envelope.confidence_breakdown.items():
            existing = breakdown.get(domain)
            breakdown[domain] = min(existing, score) if existing is not None else score

    return TrustEnvelope(
        sources=_merge_sources(cloud_cell, hosted_cell),
        confidence_breakdown=breakdown,
        assumptions={
            "workload_profile": workload.slug,
            "avg_input_tokens": workload.avg_input_tokens,
            "avg_output_tokens": workload.avg_output_tokens,
        },
        caveats=[_WORKLOAD_CAVEAT],
        freshness=_merge_freshness(cloud_cell, hosted_cell),
        verify_links=_merge_verify_links(cloud_cell, hosted_cell),
    )
