"""Per-source freshness → confidence calibration.

The canonical breakpoints from spec/SHARED.md § "Staleness policy
— freshness decays confidence". Centralizing them here means
every trust-envelope builder reaches for the same function — a
single breakpoint change updates every tool consistently.

The breakpoints are calibrated to actual upstream refresh
cadences (CP ~hourly, AA ~8x/day with 6h cache, HF config.json
rarely changes, datasheet facts don't decay). A future change
to those cadences should update the function AND the spec
section in lock-step.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

# `freshness_confidence` accepts the public source-name alphabet
# from `trust/envelope.py`'s `SourceName` literal. We import that
# directly rather than re-declaring so the two stay in sync.
from whatcanirun.trust.envelope import SourceName


def freshness_confidence(source: SourceName, age: dt.timedelta) -> float:
    """Return the freshness confidence for a source given how old
    its cached data is. Breakpoints per spec/SHARED.md § Staleness
    policy. Unknown sources get a conservative 0.50 (low but
    non-zero — they contributed *something* worth surfacing)."""
    if age.total_seconds() < 0:
        # Cached "future" timestamp — almost certainly a clock skew
        # bug or test fixture mistake. Treat as fresh rather than
        # asserting and breaking the live call path.
        age = dt.timedelta(0)

    if source == "computeprices":
        # CP refreshes ~hourly per ADR-001.
        if age < dt.timedelta(hours=2):
            return 0.95
        if age < dt.timedelta(hours=24):
            return 0.75
        return 0.40
    if source == "artificial_analysis":
        # AA refreshes ~8x/day; our cache TTL is 6h per M04.
        if age < dt.timedelta(hours=12):
            return 0.95
        if age < dt.timedelta(hours=72):
            return 0.75
        return 0.40
    if source == "huggingface":
        # config.json rarely changes after a model's release.
        if age < dt.timedelta(days=30):
            return 0.95
        return 0.80
    if source == "datasheet_yaml":
        # Manufacturer facts don't decay; controlled by us.
        return 0.99
    if source == "public_benchmark_anchor":
        # Blog posts get stale (the benchmark methodology stays
        # constant but the cited GPU's drivers/frameworks evolve).
        if age < dt.timedelta(days=90):
            return 0.85
        if age < dt.timedelta(days=365):
            return 0.70
        return 0.45
    # `own_measured_benchmark` and `bandwidth_heuristic` don't have
    # a spec-defined decay curve yet (v2 / heuristic respectively).
    # Conservative middle value until those land.
    return 0.50


# Per-domain default confidence for the *underlying methodology*
# (separate from per-source freshness). These are the values to
# combine with `freshness_confidence` via `min(...)` to get the
# domain's final entry in `confidence_breakdown`.
_FIT_CHECK_METHODOLOGY_CONFIDENCE = 0.85
"""Spec/SHARED.md § Calibration implies fit_check methodology
confidence sits below the architecture-data freshness because of
the framework_overhead heuristic (15% of weights, 2GB floor) and
the kv_cache_strategy=='sliding_window' deferral. 0.85 reflects
'sound math, known heuristics' — fits=False is high-confidence
because VRAM exhaustion is unambiguous; fits=True carries the
sufficiency_caveat to compensate."""


def fit_check_methodology_confidence() -> float:
    """Return the methodology-confidence component for the
    `fit_check` domain. Kept as a function (not just exposing the
    constant) so future calibration logic (e.g. per-model-family
    adjustment for MLA vs GQA) can land here without changing the
    builder API."""
    return _FIT_CHECK_METHODOLOGY_CONFIDENCE


# `Lowest = "lowest" / Newest = "newest"` — kept as a sentinel so the
# `_combine_freshness` helper below can disambiguate intent at call
# sites that need it; today only `lowest` is used.
_FreshnessRollup = Literal["lowest"]


def combine_freshness(
    pairs: list[tuple[SourceName, dt.timedelta]],
    *,
    rollup: _FreshnessRollup = "lowest",
) -> float:
    """Reduce a list of (source, age) pairs to a single freshness
    confidence via the weakest-link rule (min of per-source
    freshness scores). The `rollup` parameter exists so future
    builders can opt into a different policy without touching
    every call site."""
    if not pairs:
        # No upstream contributed data — the caller should not be
        # building a numerical-output envelope at all. Return 0.0
        # so any envelope that slips through has obviously broken
        # confidence rather than an arbitrary "looks fresh" score.
        return 0.0
    scores = [freshness_confidence(source, age) for source, age in pairs]
    if rollup == "lowest":
        return min(scores)
    raise ValueError(f"unknown freshness rollup: {rollup!r}")
