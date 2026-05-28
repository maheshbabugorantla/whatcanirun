"""End-to-end schema-evolution tests for ArtificialAnalysisClient.

ADR-015: upstream-data clients must tolerate new fields without
breaking validation. CI's dedicated `schema-evolution` job collects
tests carrying `@pytest.mark.schema_evolution` and fails the build
if none are collected.

AA's evolving surface area:
  - top-level row fields (anything not in our projection survives in
    `raw`)
  - nested `evaluations` sub-keys (Intelligence Index revisions ship
    new keys every few releases — `aime_25`, `lcr`, `tau2`,
    `ifbench`, `hle` were all added since AA's docs were written)
  - nested `pricing` sub-keys (cache/batch/tiered variants land per
    provider)

Each test injects a synthetic future field at one of those levels
and asserts it round-trips end-to-end (HTTP → projection → cache).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing.artificial_analysis import (
    AA_MODELS_URL,
    ArtificialAnalysisClient,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "aa"


def _payload() -> dict[str, Any]:
    return json.loads((_FIXTURES / "aa_models_2026-05-27.json").read_text())


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_top_level_field_survives_end_to_end(cache_dir: Path) -> None:
    """A future AA release adds a top-level field on a model row
    (e.g. a new `tier` discriminator separating free vs preview
    models). The projection ignores it for typed access (per
    `extra="ignore"`) but the value lives on in `raw` so a later
    code change can project it without re-deploying."""
    payload = _payload()
    payload["data"][0] = {
        **payload["data"][0],
        "future_tier": "preview",  # field we don't currently model
    }
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(200, json=payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows = await client.get_models()
    assert rows[0].raw["future_tier"] == "preview"


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_evaluations_subkey_survives_end_to_end(cache_dir: Path) -> None:
    """AA's Intelligence Index ships new evaluation keys every few
    revisions. Simulating Intelligence Index v5 with a new
    `gdpval_aa` key (spec/M04's example) — the projection's typed
    `evaluations: dict[str, float | None]` accepts any string key,
    so the new value is queryable on the projected row AND lives
    in `raw` for the audit trail.

    If anyone narrow-types `evaluations` to a Literal whitelist in
    a future cleanup pass, THIS test goes red — exactly the trust-
    contract guard rail ADR-015 exists to provide."""
    payload = _payload()
    payload["data"][0] = {
        **payload["data"][0],
        "evaluations": {
            **payload["data"][0]["evaluations"],
            "gdpval_aa": 0.5,  # simulated Intelligence Index v5 key
        },
    }
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(200, json=payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows = await client.get_models()
    assert rows[0].evaluations["gdpval_aa"] == 0.5
    assert rows[0].raw["evaluations"]["gdpval_aa"] == 0.5


@pytest.mark.schema_evolution
@pytest.mark.asyncio
@respx.mock
async def test_unknown_pricing_subkey_survives_end_to_end(cache_dir: Path) -> None:
    """`pricing` already carries `price_1m_blended_3_to_1`,
    `price_1m_input_tokens`, `price_1m_output_tokens`; AA can add
    cache/batch/tiered variants per provider at any time. The dict
    must accept arbitrary sub-keys (with `None` for "provider
    doesn't offer this tier") rather than fail validation."""
    payload = _payload()
    payload["data"][0] = {
        **payload["data"][0],
        "pricing": {
            **payload["data"][0]["pricing"],
            "price_1m_cached_input_tokens": 0.05,
            "price_1m_batch_input_tokens": None,
        },
    }
    respx.get(AA_MODELS_URL).mock(return_value=httpx.Response(200, json=payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows = await client.get_models()
    assert rows[0].pricing["price_1m_cached_input_tokens"] == 0.05
    assert rows[0].pricing["price_1m_batch_input_tokens"] is None
