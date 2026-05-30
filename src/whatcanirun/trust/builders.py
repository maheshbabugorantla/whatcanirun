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
from whatcanirun.inference.fit_check import FitResult
from whatcanirun.pricing.projections import GpuCatalogRow
from whatcanirun.trust.calibration import (
    combine_freshness,
    fit_check_methodology_confidence,
    freshness_confidence,
)
from whatcanirun.trust.envelope import ConfidenceDomain, Source, TrustEnvelope


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
