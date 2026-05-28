"""M04 acceptance-criterion gap: spec § Attribution requires every
`TrustEnvelope.sources` entry that ships an AA-sourced number to
carry the exact license_attribution string from AA's free-tier
terms. M08 (TrustEnvelope construction) and M09 (cost-cells://
provenance resource) both consume this string; they must import a
single source-of-truth constant rather than retyping it — otherwise
the spec text drifts and we end up shipping an attribution that
doesn't satisfy AA's ToS.

This test pins the exact string from `spec/M04 § Attribution` so a
future drive-by edit can't silently change what we attribute to AA.
"""

from __future__ import annotations

from whatcanirun.pricing.artificial_analysis import AA_ATTRIBUTION_STRING


def test_attribution_constant_matches_spec_exactly() -> None:
    """The string is verbatim from spec/M04-aa-optional-client.md §
    Attribution. Any change to either side requires updating both
    (and re-reading AA's free-tier terms to confirm the current
    required wording)."""
    expected = (
        "Includes data from Artificial Analysis (https://artificialanalysis.ai/), "
        "used under their free-tier API terms with attribution."
    )
    assert expected == AA_ATTRIBUTION_STRING


def test_attribution_constant_exported_from_package() -> None:
    """M08/M09 consumers import directly from the package — verify
    the symbol is in `__all__` so static analysis + IDE
    autocompletion both see it."""
    from whatcanirun.pricing import artificial_analysis as pkg

    assert "AA_ATTRIBUTION_STRING" in pkg.__all__
    assert pkg.AA_ATTRIBUTION_STRING == AA_ATTRIBUTION_STRING
