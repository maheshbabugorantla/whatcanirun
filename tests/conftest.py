"""Shared pytest fixtures.

Per ADR-013 and ADR-015: tests use captured fixtures from /tests/fixtures/,
NEVER live network calls. CI runs with COMPUTEPRICES_API_KEY="" and AA_API_KEY="".
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the captured-response fixtures used by all tests."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def disable_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session-scoped guard: any accidental live httpx call raises immediately.

    Add this fixture to tests that should NEVER touch the network. Tests that
    use respx will install their own transport and don't need this.
    """
    import httpx

    async def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError(
            "Live network call attempted in test. Use respx fixtures, "
            "or update tests/fixtures/ with a captured response."
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", _raise)
