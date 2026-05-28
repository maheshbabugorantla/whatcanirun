"""Slice B: `ArtificialAnalysisClient.get_models()` HTTP path with
respx-stubbed upstream.

AA's endpoint is `GET /api/v2/data/llms/models`. Auth is
`X-Api-Key: <key>` (NOT `Authorization: Bearer` — verified live with
a real key on 2026-05-27). The free tier is 1k/day; tests never
touch the live network so the budget is irrelevant for CI.

Tenacity retry policy mirrors M02's CP client: 429 / 5xx / connection
errors are transient (retry with exponential backoff); other 4xx is a
client bug that retrying would only mask. Tests pass
`retry_wait_*_s=0` so the suite doesn't sleep ~7s per fallback path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.pricing.artificial_analysis import (
    ArtificialAnalysisClient,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"

_AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


@pytest.fixture(scope="module")
def aa_payload() -> dict[str, Any]:
    return json.loads((_FIXTURES / "aa_models_2026-05-27.json").read_text())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "aa"


@pytest.fixture
def fast_client(cache_dir: Path) -> ArtificialAnalysisClient:
    """Client with retries on but zero backoff so tests don't burn
    real seconds on the fallback path."""
    return ArtificialAnalysisClient(
        cache_dir=cache_dir,
        api_key="aa_test_key_123",
        retry_attempts=4,
        retry_wait_min_s=0.0,
        retry_wait_max_s=0.0,
    )


# ---------------------------------------------------------- happy path


@pytest.mark.asyncio
@respx.mock
async def test_get_models_returns_525_projected_rows(
    cache_dir: Path, aa_payload: dict[str, Any]
) -> None:
    """First call hits AA, parses the full data[] array, and returns
    a list of `AaModelRow`. The live 2026-05-27 capture has 525 rows
    (spec said 524; close enough — varies day to day)."""
    respx.get(_AA_URL).mock(return_value=httpx.Response(200, json=aa_payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="aa_test_key_123")
    rows = await client.get_models()

    assert len(rows) == 525
    # Spot-check a known row from the fixture.
    sample = next(r for r in rows if r.slug == "gpt-oss-120b-low")
    assert sample.median_output_tokens_per_second == 335.934
    assert sample.model_creator["slug"] == "openai"
    # ADR-015: raw payload preserved on every row.
    assert sample.raw["id"] == "c99f3bde-7c08-4de8-bd5c-8ee9123ebffa"


@pytest.mark.asyncio
@respx.mock
async def test_get_raw_response_returns_top_level_envelope(
    cache_dir: Path, aa_payload: dict[str, Any]
) -> None:
    """`get_raw_response()` returns the top-level envelope —
    `status`, `prompt_options`, `data` — not just the `data[]`
    array. M09's trust-envelope provenance consumes the `status`
    field; the schema-evolution audit reads the unprojected
    payload."""
    respx.get(_AA_URL).mock(return_value=httpx.Response(200, json=aa_payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="aa_test_key_123")
    raw = await client.get_raw_response()

    assert set(raw.keys()) >= {"status", "data"}
    assert raw["status"] == 200
    assert len(raw["data"]) == 525


# --------------------------------------------------------------- auth header


@pytest.mark.asyncio
@respx.mock
async def test_x_api_key_header_set_on_request(cache_dir: Path, aa_payload: dict[str, Any]) -> None:
    """AA uses `X-Api-Key`, NOT `Authorization: Bearer` (verified
    live 2026-05-27)."""
    route = respx.get(_AA_URL).mock(return_value=httpx.Response(200, json=aa_payload))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="hunter2")
    await client.get_models()

    request = route.calls.last.request
    assert request.headers.get("x-api-key") == "hunter2"
    # And we MUST NOT also be sending a Bearer header — would leak
    # the same secret over a header AA doesn't expect.
    assert "authorization" not in {k.lower() for k in request.headers}


# -------------------------------------------------------- shape validation


@pytest.mark.asyncio
@respx.mock
async def test_get_models_returns_empty_on_missing_data_key(
    cache_dir: Path,
) -> None:
    """If AA returns a payload missing the `data` key, `get_models`
    routes the shape failure through the same graceful-fallback
    path as HTTP failures (no cache → empty list + logged warning).
    AA is optional and the parent tool call must keep working — the
    operator sees the schema break via the warning log, not via a
    propagated exception. The unwrapped `get_raw_response` accessor
    still surfaces the broken payload directly to its caller (M09's
    trust-envelope provenance code opts in to that)."""
    respx.get(_AA_URL).mock(return_value=httpx.Response(200, json={"status": 200}))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows = await client.get_models()
    assert rows == []


@pytest.mark.asyncio
@respx.mock
async def test_get_models_returns_empty_on_non_list_data(
    cache_dir: Path,
) -> None:
    """Same graceful-fallback contract for non-list `data`."""
    respx.get(_AA_URL).mock(return_value=httpx.Response(200, json={"status": 200, "data": "oops"}))

    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    rows = await client.get_models()
    assert rows == []


# --------------------------------------------------------------- retry policy


@pytest.mark.asyncio
@respx.mock
async def test_transient_500_then_success_recovers(
    fast_client: ArtificialAnalysisClient, aa_payload: dict[str, Any]
) -> None:
    """Tenacity retries on 5xx; second attempt sees the recovered
    upstream. Mirrors M02's retry contract."""
    route = respx.get(_AA_URL).mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(500, text="still boom"),
            httpx.Response(200, json=aa_payload),
        ]
    )

    rows = await fast_client.get_models()
    assert route.call_count == 3
    assert len(rows) == 525


@pytest.mark.asyncio
@respx.mock
async def test_429_is_retried(
    fast_client: ArtificialAnalysisClient, aa_payload: dict[str, Any]
) -> None:
    """AA's 1k/day budget makes 429 plausible under heavy CI use.
    Retry rather than fail the whole sync."""
    route = respx.get(_AA_URL).mock(
        side_effect=[
            httpx.Response(429, text="rate-limited"),
            httpx.Response(200, json=aa_payload),
        ]
    )
    rows = await fast_client.get_models()
    assert route.call_count == 2
    assert len(rows) == 525


@pytest.mark.asyncio
@respx.mock
async def test_401_does_not_retry(
    fast_client: ArtificialAnalysisClient,
) -> None:
    """401 = bad key. Two contracts apply here:

      1. Retrying 401 just wastes upstream's logs and our quota —
         `_is_retryable_http_error` returns False so tenacity makes
         one attempt and stops.
      2. Per spec/M04 § Acceptance criteria, AA upstream failures
         (401/429/500) MUST NOT propagate to the parent tool call.
         Slice H's graceful-fallback wrapping in `get_models` turns
         the propagated 401 into an empty list + logged warning.

    This test pins BOTH: the call count stays at 1 (no retry) AND
    the caller sees `[]` rather than a raised HTTPStatusError. The
    raw-response accessor (`get_raw_response`) does propagate the
    error — covered in `test_client_fallback.py` if/when M07 needs
    that distinction."""
    route = respx.get(_AA_URL).mock(return_value=httpx.Response(401, text="bad key"))

    rows = await fast_client.get_models()
    assert rows == []
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_connection_error_is_retried(
    fast_client: ArtificialAnalysisClient, aa_payload: dict[str, Any]
) -> None:
    """Network blips during sync — retry."""
    route = respx.get(_AA_URL).mock(
        side_effect=[
            httpx.ConnectError("network blip"),
            httpx.Response(200, json=aa_payload),
        ]
    )
    rows = await fast_client.get_models()
    assert route.call_count == 2
    assert len(rows) == 525
