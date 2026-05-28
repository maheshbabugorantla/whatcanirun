"""Slice C: `ArtificialAnalysisClient` is feature-flagged off when no
`AA_API_KEY` is present.

The whole AA integration is optional per spec/M04: when the key is
unset, the rest of the system works unchanged — no calls, no
caveats, no trust envelope entries naming AA. That's a strict
guarantee, not a "best effort": M07's Tier 2 must be able to ask
`client.enabled` and route to Tier 3/4 without ever touching the
network.

Mirrors M02's empty-string-is-anonymous semantics on the env var so
a CI safeguard `AA_API_KEY=""` doesn't accidentally trip the enabled
path with a malformed `Authorization: Bearer ` header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whatcanirun.pricing.artificial_analysis import (
    AaDisabled,
    ArtificialAnalysisClient,
)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "aa"


def test_explicit_none_key_disables_client(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing with `api_key=None` AND no env var → disabled."""
    monkeypatch.delenv("AA_API_KEY", raising=False)
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key=None)
    assert client.enabled is False


def test_empty_env_var_treated_as_anonymous(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CI safeguard like `AA_API_KEY=""` in compose.claude.yml or
    a `.env` template MUST be treated as "no key" — matches M02's
    `COMPUTEPRICES_API_KEY` semantics. Sending an empty bearer
    header is worse than no header (some upstreams 400 on it)."""
    monkeypatch.setenv("AA_API_KEY", "")
    client = ArtificialAnalysisClient(cache_dir=cache_dir)
    assert client.enabled is False


def test_whitespace_only_env_var_treated_as_anonymous(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rationale — a stray `AA_API_KEY=" "` in a shell wrapper
    shouldn't enable an unusable bearer header."""
    monkeypatch.setenv("AA_API_KEY", "   \t  ")
    client = ArtificialAnalysisClient(cache_dir=cache_dir)
    assert client.enabled is False


def test_real_env_var_enables_client(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_API_KEY", "aa_test_key_123")
    client = ArtificialAnalysisClient(cache_dir=cache_dir)
    assert client.enabled is True


def test_explicit_ctor_arg_overrides_env(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ctor arg wins over env var — same convention as the CP
    client. Lets tests run with deterministic keys without
    cross-contaminating from the developer's `.env`."""
    monkeypatch.setenv("AA_API_KEY", "env-key")
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="ctor-key")
    assert client.enabled is True
    # Internal access is intentional here — the explicit-override
    # behavior is part of the public contract and we want a
    # regression test that pins which one wins.
    assert client._api_key == "ctor-key"


@pytest.mark.asyncio
async def test_get_models_raises_aa_disabled_when_no_key(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract for callers who haven't checked `.enabled`:
    `get_models()` raises `AaDisabled` cleanly rather than making a
    network call with an empty Authorization header. Catches bugs in
    the M07 routing code where someone forgets to check
    `client.enabled` before calling."""
    monkeypatch.delenv("AA_API_KEY", raising=False)
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key=None)
    with pytest.raises(AaDisabled):
        await client.get_models()


@pytest.mark.asyncio
async def test_get_raw_response_raises_aa_disabled_when_no_key(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same contract for the raw-payload accessor — must not let a
    caller bypass the disabled check by reaching for raw bytes."""
    monkeypatch.delenv("AA_API_KEY", raising=False)
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key=None)
    with pytest.raises(AaDisabled):
        await client.get_raw_response()
