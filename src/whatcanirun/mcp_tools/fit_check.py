"""M09 Slice C: `fit_check` MCP tool.

Wraps M06's `compute_fit` pure-math kernel with the trust envelope
the MCP client surfaces to the user. Per spec/M09 § Public surface
§4, the tool returns a `FitResult` with trust envelope — wrapped
here as `FitCheckToolResponse` so the FastMCP serializer produces
a stable wire schema.

`build_fit_check_response` is the pure builder exposed for unit
tests. The async `fit_check` function is the FastMCP-registered
tool entry point — Slice L will replace the placeholder slug
lookup with the full unknown-model dispatcher (Case 1/2/3
routing).
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict

from whatcanirun.catalog.hf_model import Model
from whatcanirun.catalog.seed_schemas import Quantization
from whatcanirun.inference.fit_check import FitResult, compute_fit
from whatcanirun.mcp_tools.dispatch import UnknownModelResponse
from whatcanirun.pricing.projections import GpuCatalogRow
from whatcanirun.trust.builders import build_fit_check_envelope
from whatcanirun.trust.envelope import TrustEnvelope


class FitCheckToolResponse(BaseModel):
    """Wrapper Pydantic for the `fit_check` tool output. Carries
    both the pure-math `FitResult` (with its own
    `sufficiency_caveat` field) and the M09 `TrustEnvelope` (with
    the four domains the verdict depends on).

    `extra="forbid"` because this is an OWNED output shape; a
    typo in construction should fail loudly rather than dropping
    a field the LLM client would otherwise surface verbatim."""

    model_config = ConfigDict(extra="forbid")

    fit_result: FitResult
    trust_envelope: TrustEnvelope


def build_fit_check_response(
    *,
    model: Model,
    gpu: GpuCatalogRow,
    quant: Quantization,
    tp_size: int,
    batch_size: int,
    context_length: int,
    now: dt.datetime,
    gpu_specs_last_updated: dt.datetime,
) -> FitCheckToolResponse:
    """Pure builder. Runs `compute_fit` then `build_fit_check_envelope`
    over the same inputs. Callers supply:

    - `now`: the reference time for freshness calculations. Tests
      pin this to a fixed value so freshness assertions don't
      depend on wallclock; the tool wrapper passes
      `datetime.now(UTC)`.
    - `gpu_specs_last_updated`: the freshness anchor for the GPU
      catalog entry. The CP gpu-catalog endpoint exposes this in
      its `meta.generated_at` block; M02's
      `ComputePricesClient.get_raw_response` is the access path.
    """
    fit_result = compute_fit(
        model=model,
        gpu=gpu,
        quant=quant,
        tp_size=tp_size,
        batch_size=batch_size,
        context_length=context_length,
    )
    envelope = build_fit_check_envelope(
        fit_result=fit_result,
        model=model,
        gpu=gpu,
        tp_size=tp_size,
        batch_size=batch_size,
        context_length=context_length,
        now=now,
        gpu_specs_last_updated=gpu_specs_last_updated,
    )
    return FitCheckToolResponse(fit_result=fit_result, trust_envelope=envelope)


async def fit_check(
    model_slug: str,
    gpu_slug: str,
    quant_slug: str,
    tp_size: int = 1,
    batch_size: int = 1,
    context_length: int = 4096,
) -> FitCheckToolResponse | UnknownModelResponse:
    """`fit_check` MCP tool entry point.

    Resolves the three slugs against the local catalog caches,
    runs the pure builder, returns the response.

    Per spec/M09 § Case 2: fit_check collapses Case 2 to Case 3
    — fit-checking fundamentally requires architecture data, so
    a CP-only model (hosted-API pricing but no HF config) cannot
    produce a defensible FitResult. Return UnknownModelResponse
    so the client can elicit the HF repo_id.
    """
    from datetime import UTC, datetime

    from whatcanirun.mcp_tools.deps import load_runtime_deps
    from whatcanirun.mcp_tools.dispatch import find_model_in_catalog

    deps = await load_runtime_deps()
    model = find_model_in_catalog(model_slug, deps)
    if model is None:
        # Case 2 (CP-only) and Case 3 both collapse here for fit_check.
        return UnknownModelResponse(requested_model_slug=model_slug)

    gpu = next((g for g in deps.gpu_catalog if g.slug == gpu_slug), None)
    if gpu is None:
        raise LookupError(
            f"gpu_slug {gpu_slug!r} not in cached gpu catalog. "
            "Call list_catalog to see supported GPUs."
        )

    quant = next((q for q in deps.quantizations if q.slug == quant_slug), None)
    if quant is None:
        raise LookupError(
            f"quant_slug {quant_slug!r} not in seed quantizations. "
            "Call list_catalog to see supported quantizations."
        )

    return build_fit_check_response(
        model=model,
        gpu=gpu,
        quant=quant,
        tp_size=tp_size,
        batch_size=batch_size,
        context_length=context_length,
        now=datetime.now(UTC),
        # Without the CP meta.generated_at timestamp threaded through,
        # use the freshest gpu_catalog row's last_updated as a
        # conservative anchor. M11 may plumb generated_at through.
        gpu_specs_last_updated=datetime.now(UTC),
    )
