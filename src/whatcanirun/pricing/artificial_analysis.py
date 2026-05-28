"""Async Artificial Analysis (AA) client — OPTIONAL enrichment for M07.

When `AA_API_KEY` is set, ingests AA's `/api/v2/data/llms/models`
endpoint (524-row free-tier response on 2026-05-27) and exposes
per-model TPS aggregates as Tier-2 anchors in `tps_estimator`. When
the key is unset, every method either raises `AaDisabled` or returns
empty — the rest of the system works unchanged with no AA mentions
in trust envelopes.

AA optionality is a strict guarantee, not best-effort: M07's Tier 2
must be able to ask `client.enabled` and route to Tier 3/4 without
ever touching the network. The AA free tier carries attribution
requirements (see spec/M04 § Attribution); any consumer that ships
an AA-sourced number into a `TrustEnvelope.sources` entry must
include the AA `license_attribution` string.

Mirrors the M02 ComputePrices client's empty-string-is-anonymous
env-var semantics so a CI safeguard `AA_API_KEY=""` doesn't
accidentally enable an unusable bearer header.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class AaDisabled(Exception):  # noqa: N818  (name is the public contract)
    """Raised when an AA-only operation is requested but no
    `AA_API_KEY` was supplied.

    Callers that route around the AA tier (M07, the unknown-model
    dispatcher in M09) check `client.enabled` first and avoid this
    exception entirely. The exception exists for the "forgot to
    check" case — make the failure loud rather than silently
    returning an empty list that downstream code mistakes for "AA
    returned no match for this model".
    """


class ArtificialAnalysisClient:
    """AA `/api/v2/data/llms/models` client with optional auth.

    All AA-touching methods raise `AaDisabled` when `self.enabled`
    is False. Callers SHOULD branch on `client.enabled` rather than
    catch the exception in the hot path.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        # ctor arg wins over env var (matches M02 CP convention so
        # tests can run with deterministic keys without polluting
        # from a developer's `.env`). Empty / whitespace env var is
        # treated as anonymous so CI safeguards like
        # `AA_API_KEY=""` don't enable a broken bearer path.
        if api_key is None:
            env_key = os.environ.get("AA_API_KEY", "").strip()
            api_key = env_key or None
        self._api_key = api_key

    @property
    def enabled(self) -> bool:
        """True iff an AA API key is available. M07 Tier-2 routing
        check, M09 trust-envelope attribution check, and every
        AA-only method gate off this."""
        return self._api_key is not None

    async def get_models(self) -> list[Any]:
        """Return the projected list of `AaModelRow`. Raises
        `AaDisabled` when no key is configured.

        Slice B will implement the live fetch + cache; Slice C ships
        only the disabled-mode behavior so M07 can already wire its
        routing logic against the stable surface.
        """
        if not self.enabled:
            raise AaDisabled(
                "AA_API_KEY is not configured; AA enrichment is off. "
                "Set the env var or pass `api_key=...` to enable."
            )
        raise NotImplementedError("Slice B will land the HTTP path.")

    async def get_raw_response(self) -> dict[str, Any]:
        """Return the full unparsed AA payload (top-level dict with
        `status`, `prompt_options`, `data`). Used by M09's trust-
        envelope provenance + the schema-evolution audit. Raises
        `AaDisabled` when no key is configured."""
        if not self.enabled:
            raise AaDisabled(
                "AA_API_KEY is not configured; AA enrichment is off. "
                "Set the env var or pass `api_key=...` to enable."
            )
        raise NotImplementedError("Slice B will land the HTTP path.")
