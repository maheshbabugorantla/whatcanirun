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
) -> FitCheckToolResponse:
    """`fit_check` MCP tool entry point.

    Resolves the three slugs against the local catalog caches,
    runs the pure builder, returns the response. The slug-
    resolution path is a placeholder for Slice L's unknown-model
    dispatcher — Case 2 (CP-only) and Case 3 (genuinely unknown)
    will swap this for an `UnknownModelResponse` branch.

    For now the tool raises `LookupError` on missing-slug paths;
    the FastMCP layer surfaces that as an MCP error which the
    LLM client relays. The tool wrapper is intentionally thin —
    Slice L is where the real dispatch logic lands.
    """
    raise NotImplementedError(
        "fit_check tool slug-resolution is wired in Slice L "
        "(unknown-model dispatcher). The pure builder "
        "`build_fit_check_response` is testable independently."
    )
