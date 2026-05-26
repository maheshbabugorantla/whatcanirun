"""Pydantic projections of ComputePrices `/api/v1/*` responses.

Per ADR-015 (Raw + Projection): every upstream response is stored
verbatim in `raw`; the typed fields below are the subset we currently
consume. `extra="ignore"` lets future CP releases ship new fields
without breaking validation — those fields survive in `raw` and can be
projected later by adding a field to the model.

`pricing_type` and `manufacturer` use `Literal[...]` enums on purpose:
if CP introduces a new value here, we want validation to fail loudly
so we can decide whether and how to model it. Per ADR-015's rationale,
narrow-type only the things we already know are stable.

Convention: instantiate via `.project(payload)`, which validates the
declared fields and stores the full payload in `raw` in one call.
Callers that build rows by hand can still use the ordinary constructor.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

Manufacturer = Literal["NVIDIA", "AMD", "Intel"]
GpuPricingType = Literal["on_demand", "reserved", "spot"]
LlmPricingType = Literal["standard", "batch"]


class _CpRow(BaseModel):
    """Shared projection helper.

    Concrete row types inherit and add fields. Subclasses MUST keep
    `extra="ignore"` and the `raw: dict[str, Any]` field — those are
    the load-bearing pieces of ADR-015's contract.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def project(cls, payload: dict[str, Any]) -> Self:
        """Project a CP row dict into the typed model, storing the full
        payload verbatim in `raw`.

        Pydantic validates declared fields against `payload` (extras
        silently dropped from the projection per ADR-015); `raw` is set
        from a shallow copy so future-added CP fields survive even
        though they weren't typed at projection time.
        """
        return cls.model_validate({**payload, "raw": dict(payload)})


class GpuCatalogRow(_CpRow):
    """One row of `GET /api/v1/gpus` — the GPU catalog (66 rows on
    2026-05-26 capture).
    """

    slug: str
    name: str
    manufacturer: Manufacturer
    architecture: str | None
    vram_gb: int
    release_date: date | None
    # specs is undocumented at field level — CP gains fields here without
    # notice (cuda_cores, fp16_tflops, memory_bandwidth_gb_s, ...). Keep
    # the union loose per ADR-015.
    specs: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class GpuPriceRow(_CpRow):
    """One row of `GET /api/v1/gpu-prices` — per-(provider, GPU,
    pricing_type) hourly rental rates (1000 rows on 2026-05-26 capture).
    """

    provider: str
    provider_slug: str
    gpu: str
    gpu_slug: str
    vram_gb: int
    gpu_count: int
    price_per_hour_usd: float
    pricing_type: GpuPricingType
    commitment_months: int | None
    currency: str
    source_url: str
    last_updated: datetime


class LlmCatalogRow(_CpRow):
    """One row of `GET /api/v1/llm-models` — the LLM model catalog
    (214 rows on 2026-05-26 capture).
    """

    slug: str
    name: str
    creator: str
    family: str | None
    context_window: int | None
    modalities: list[str]
    knowledge_cutoff: date | None


class LlmPriceRow(_CpRow):
    """One row of `GET /api/v1/llm-prices` — per-(provider, model)
    hosted-API token pricing (498 rows on 2026-05-26 capture).

    Note: CP's field names use `price_per_1m_*` (with `per_`), not the
    `price_1m_*` shown in earlier spec drafts. `price_per_1m_cached_input_usd`
    is the prompt-caching tier that some providers populate.
    """

    provider: str
    provider_slug: str
    model_slug: str
    price_per_1m_input_usd: float | None
    price_per_1m_output_usd: float | None
    price_per_1m_cached_input_usd: float | None
    pricing_type: LlmPricingType
    last_updated: datetime
