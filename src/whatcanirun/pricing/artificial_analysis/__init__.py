"""Artificial Analysis (AA) integration — package entry point.

Re-exports the public surface from the three concern-specific
submodules so consumers can `from whatcanirun.pricing.artificial_analysis
import ArtificialAnalysisClient, AaModelRow, AaSlugMappingRow,
resolve_aa_slug, AaDisabled` regardless of which submodule actually
defines each symbol. Treat the submodules as implementation detail;
the package surface IS the contract.

  - `client`        — ArtificialAnalysisClient, AaDisabled,
                      AA_MODELS_URL (HTTP + cache + snapshot +
                      ADR-013 fallback)
  - `projections`   — AaModelRow, ReasoningEffort (Pydantic shape
                      of one AA `data[]` row, per ADR-015)
  - `slug_mapping`  — AaSlugMappingRow, AaSlugVariant,
                      resolve_aa_slug (curated CP slug → AA slug
                      pairing loaded from seeds/aa_slug_mapping.yaml)
"""

from whatcanirun.pricing.artificial_analysis.client import (
    AA_MODELS_URL,
    AaDisabled,
    ArtificialAnalysisClient,
)
from whatcanirun.pricing.artificial_analysis.projections import (
    AaModelRow,
    ReasoningEffort,
)
from whatcanirun.pricing.artificial_analysis.slug_mapping import (
    AaSlugMappingRow,
    AaSlugVariant,
    resolve_aa_slug,
)

__all__ = [
    "AA_MODELS_URL",
    "AaDisabled",
    "AaModelRow",
    "AaSlugMappingRow",
    "AaSlugVariant",
    "ArtificialAnalysisClient",
    "ReasoningEffort",
    "resolve_aa_slug",
]
