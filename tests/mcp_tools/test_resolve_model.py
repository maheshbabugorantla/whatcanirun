"""M09 Slice L (step 1): `resolve_model` tool — TDD.

`resolve_model(model_slug, hf_repo_id)` is the tool the MCP client
calls after receiving an `UnknownModelResponse` and eliciting the
HF repo_id from the user. It:

1. Validates the slug + repo_id format (no path traversal vectors)
2. Appends `(slug, hf_repo_id)` to `~/.config/whatcanirun/user_models.yaml`
3. Triggers `HfModelSync.sync_model(slug=..., repo_id=...)` to
   fetch + cache the model's HF config.json at the current
   revision SHA
4. Returns `ResolveModelResult` with status + the resolved revision SHA

Per spec/M09 § Public surface §6, `ResolveModelResult` deliberately
does NOT carry a trust_envelope — the response is a status +
diagnostic, no numerical fields. The follow-up tool call is where
the envelope appears.

These tests use a temp config directory + a stubbed
`HfModelSync.sync_model` so no live network is needed in CI.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from whatcanirun.catalog.hf_model import Model
from whatcanirun.mcp_tools.dispatch import (
    ResolveModelResult,
    UnknownModelResponse,
    resolve_model_to_user_yaml,
)


def _build_model(slug: str, repo_id: str) -> Model:
    return Model(
        slug=slug,
        hf_repo_id=repo_id,
        display_name=repo_id.split("/")[-1],
        total_params_b=None,
        active_params_b=None,
        n_layers=80,
        n_attention_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        max_position_embeddings=131072,
        native_dtype="bfloat16",
        architecture_family="llama",
        kv_cache_strategy="standard_gqa",
        raw_config={},
        raw_safetensors_meta={},
        hf_revision_sha="abc123def456",
        last_synced_at=dt.datetime(2026, 5, 28, tzinfo=dt.UTC),
    )


@pytest.fixture
def user_config_dir(tmp_path: Path) -> Path:
    """A fresh per-test XDG-style config directory so tests don't
    contaminate each other's user_models.yaml state."""
    return tmp_path / "config" / "whatcanirun"


@pytest.fixture
def stub_sync_success(monkeypatch: Any) -> AsyncMock:
    """Stub `HfModelSync.sync_model` to return a successful Model
    without touching the network or filesystem. The mock returns a
    minimal Model carrying the requested slug + repo_id and a fixed
    fake hf_revision_sha."""
    mock = AsyncMock()

    async def _fake_sync(*, slug: str, repo_id: str, **kwargs: Any) -> Model:
        return _build_model(slug, repo_id)

    mock.side_effect = _fake_sync
    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        mock,
    )
    return mock


@pytest.fixture
def stub_sync_404(monkeypatch: Any) -> AsyncMock:
    """Stub `HfModelSync.sync_model` to raise the equivalent of a
    'repo not found' error — what happens when the user supplies a
    typo or a private repo their token can't see."""
    import httpx

    mock = AsyncMock()
    mock.side_effect = httpx.HTTPStatusError(
        message="404 Not Found",
        request=httpx.Request("GET", "https://huggingface.co/api/models/x"),
        response=httpx.Response(404),
    )
    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        mock,
    )
    return mock


# ---------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_resolve_returns_resolved_status_on_success(
    user_config_dir: Path,
    stub_sync_success: AsyncMock,
    tmp_path: Path,
) -> None:
    """Happy path: valid slug + valid repo_id, sync succeeds —
    response is `status='resolved'` with the hf_revision_sha from
    the just-completed sync."""
    result = await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/My-Llama",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(result, ResolveModelResult)
    assert result.status == "resolved"
    assert result.hf_revision_sha == "abc123def456"
    assert result.error_detail is None


@pytest.mark.asyncio
async def test_resolve_persists_pair_to_user_yaml(
    user_config_dir: Path,
    stub_sync_success: AsyncMock,
    tmp_path: Path,
) -> None:
    """Spec/M09 § Public surface §6: 'Persists the (model_slug,
    hf_repo_id) mapping to ~/.config/whatcanirun/user_models.yaml'.
    The file must exist after the call and contain the new row in
    YAML's documented `TrackedModelRow` shape."""
    import yaml

    await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/My-Llama",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )

    user_yaml = user_config_dir / "user_models.yaml"
    assert user_yaml.exists()
    rows = yaml.safe_load(user_yaml.read_text())
    assert isinstance(rows, list)
    assert any(r["slug"] == "my-llama" and r["hf_repo_id"] == "vendor/My-Llama" for r in rows)


@pytest.mark.asyncio
async def test_resolve_appends_to_existing_yaml(
    user_config_dir: Path,
    stub_sync_success: AsyncMock,
    tmp_path: Path,
) -> None:
    """Two resolve_model calls in sequence must produce a 2-row
    user_models.yaml — appending, not overwriting. A future bug
    that re-writes the file with just the latest row would lose
    prior user-supplied mappings."""
    import yaml

    await resolve_model_to_user_yaml(
        model_slug="first-model",
        hf_repo_id="vendor/first",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    await resolve_model_to_user_yaml(
        model_slug="second-model",
        hf_repo_id="vendor/second",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    rows = yaml.safe_load((user_config_dir / "user_models.yaml").read_text())
    slugs = {r["slug"] for r in rows}
    assert slugs == {"first-model", "second-model"}


@pytest.mark.asyncio
async def test_resolve_updates_existing_pair_for_same_slug(
    user_config_dir: Path,
    stub_sync_success: AsyncMock,
    tmp_path: Path,
) -> None:
    """If the same slug is resolved twice with a different repo_id,
    the second call must overwrite the first row's repo_id rather
    than ending up with two rows for the same slug — that would
    later confuse the M03 loader."""
    import yaml

    await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/old-llama",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/new-llama",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    rows = yaml.safe_load((user_config_dir / "user_models.yaml").read_text())
    # one row for `my-llama`, with the updated repo_id
    my_llama_rows = [r for r in rows if r["slug"] == "my-llama"]
    assert len(my_llama_rows) == 1
    assert my_llama_rows[0]["hf_repo_id"] == "vendor/new-llama"


@pytest.mark.asyncio
async def test_resolve_returns_sync_failed_on_sync_error(
    user_config_dir: Path,
    stub_sync_404: AsyncMock,
    tmp_path: Path,
) -> None:
    """When `HfModelSync.sync_model` raises (404, private repo,
    network error), the tool returns `status='not_found_on_hf'`
    rather than crashing. The error_detail field carries the
    diagnostic so the LLM client can relay it to the user."""
    result = await resolve_model_to_user_yaml(
        model_slug="bogus",
        hf_repo_id="vendor/does-not-exist",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "not_found_on_hf"
    assert result.error_detail is not None
    assert result.hf_revision_sha is None


def test_unknown_model_response_carries_elicit_prompt() -> None:
    """Spec/M09 § Case 3: UnknownModelResponse must include the
    `elicit_prompt` text the MCP client surfaces to the user. The
    prose is fixed per spec — a regression that empties it would
    leave the client guessing what to ask."""
    response = UnknownModelResponse(requested_model_slug="mystery-model")
    assert response.requested_model_slug == "mystery-model"
    assert response.status == "unknown_model"
    assert response.elicit_field == "hf_repo_id"
    assert "Hugging Face" in response.elicit_prompt
    assert "repo_id" in response.elicit_prompt


def test_unknown_model_response_includes_suggested_followups() -> None:
    """The elicit_prompt is the headline ask; suggested_followups
    is the LLM client's "what to do if the user can't supply"
    fallback list — drops to list_catalog or budget_to_plan with
    a publicly tracked model."""
    response = UnknownModelResponse(requested_model_slug="mystery")
    assert len(response.suggested_followups) >= 1
    assert any("list_catalog" in s for s in response.suggested_followups)


@pytest.mark.asyncio
async def test_resolve_rejects_malformed_slug_before_yaml_write(
    user_config_dir: Path,
    tmp_path: Path,
) -> None:
    """Copilot review #15 #3: a malformed slug (path separator,
    dot segments) must NOT land in `user_models.yaml`. If it
    persists pre-validation, every subsequent dispatch_model_request
    would re-attempt the doomed sync and pollute the user's
    config with un-syncable junk."""
    result = await resolve_model_to_user_yaml(
        model_slug="../etc/passwd",  # path separator + traversal
        hf_repo_id="vendor/legit",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "not_found_on_hf"
    assert "invalid slug" in (result.error_detail or "")
    # CRUCIAL: the file must NOT have been written. Even an empty
    # file would expose the next loader to a malformed-row code path.
    assert not (user_config_dir / "user_models.yaml").exists()


@pytest.mark.asyncio
async def test_resolve_rejects_malformed_repo_id_before_yaml_write(
    user_config_dir: Path,
    tmp_path: Path,
) -> None:
    """Symmetric guard: an HF repo_id that fails the
    `<namespace>/<name>` shape (extra slash, query string, etc.)
    must be rejected before persisting."""
    result = await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/name?token=stolen",  # query-string smuggling
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "not_found_on_hf"
    assert "invalid repo_id" in (result.error_detail or "")
    assert not (user_config_dir / "user_models.yaml").exists()


@pytest.mark.asyncio
async def test_resolve_returns_sync_failed_on_http_5xx(
    user_config_dir: Path,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A 5xx response from HF means the repo_id is fine but HF is
    having a bad day — the recourse is to retry, not to fix the
    repo_id. Status must be `sync_failed`, not `not_found_on_hf`,
    so the LLM client surfaces the correct retry guidance."""
    import httpx

    mock = AsyncMock()
    mock.side_effect = httpx.HTTPStatusError(
        message="503 Service Unavailable",
        request=httpx.Request("GET", "https://huggingface.co/api/models/x"),
        response=httpx.Response(503),
    )
    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        mock,
    )
    result = await resolve_model_to_user_yaml(
        model_slug="any-slug",
        hf_repo_id="vendor/any",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "sync_failed"
    assert "503" in (result.error_detail or "")
    # Sync-first ordering: the yaml MUST NOT exist after a 5xx —
    # the previous order persisted before confirming sync, leaving
    # an un-syncable row that would re-trigger Case 1b on every
    # subsequent dispatch_model_request.
    assert not (user_config_dir / "user_models.yaml").exists()


@pytest.mark.asyncio
async def test_resolve_returns_sync_failed_on_network_error(
    user_config_dir: Path,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A network-layer failure (DNS, refused connection, TLS, etc.)
    means HF is unreachable from this machine. Status must be
    `sync_failed` so the LLM client offers a retry rather than
    accusing the user's repo_id."""
    import httpx

    mock = AsyncMock()
    mock.side_effect = httpx.ConnectError("connection refused")
    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        mock,
    )
    result = await resolve_model_to_user_yaml(
        model_slug="any-slug",
        hf_repo_id="vendor/any",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "sync_failed"
    # Same sync-first guarantee: no yaml on network-layer failure.
    assert not (user_config_dir / "user_models.yaml").exists()


@pytest.mark.asyncio
async def test_resolve_does_not_pollute_yaml_on_hf_404(
    user_config_dir: Path,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Copilot review #15 round 5: the original ordering wrote
    user_models.yaml BEFORE calling HfModelSync.sync_model. On a
    404 (wrong/private repo_id) the bad mapping persisted, which
    meant:
      (a) list_catalog would advertise the model as supported
          even though it can't be loaded, and
      (b) every subsequent tool call would re-trigger Case 1b
          lazy-sync against the bad repo_id.

    The fix defers the yaml write to the success branch — a
    404 must leave the file completely untouched."""
    import httpx

    mock = AsyncMock()
    mock.side_effect = httpx.HTTPStatusError(
        message="404 Not Found",
        request=httpx.Request("GET", "https://huggingface.co/api/models/x"),
        response=httpx.Response(404),
    )
    monkeypatch.setattr(
        "whatcanirun.catalog.hf_sync.HfModelSync.sync_model",
        mock,
    )
    result = await resolve_model_to_user_yaml(
        model_slug="non-existent-model",
        hf_repo_id="vendor/does-not-exist",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "not_found_on_hf"
    assert "404" in (result.error_detail or "")
    # The crucial assertion: NO persistence on the failure path.
    assert not (user_config_dir / "user_models.yaml").exists(), (
        "404 left an un-syncable row in user_models.yaml — list_catalog "
        "would advertise it as supported and every tool call would "
        "re-attempt the doomed Case 1b sync"
    )


@pytest.mark.asyncio
async def test_resolve_persists_yaml_only_after_sync_succeeds(
    user_config_dir: Path,
    stub_sync_success: AsyncMock,
    tmp_path: Path,
) -> None:
    """Order-of-operations regression: on the success path the
    yaml DOES get written (and contains the resolved row).
    Together with the 404 / 5xx / network tests above, this pins
    the contract: yaml-on-success, no-yaml-on-failure."""
    import yaml

    result = await resolve_model_to_user_yaml(
        model_slug="my-llama",
        hf_repo_id="vendor/My-Llama",
        config_dir=user_config_dir,
        cache_dir=tmp_path / "cache",
    )
    assert result.status == "resolved"
    user_yaml = user_config_dir / "user_models.yaml"
    assert user_yaml.exists()
    rows = yaml.safe_load(user_yaml.read_text())
    assert any(r["slug"] == "my-llama" and r["hf_repo_id"] == "vendor/My-Llama" for r in rows)


def test_resolve_model_registered_as_mcp_tool() -> None:
    """Registration smoke test — the tool surface advertised on
    `initialize` must include `resolve_model`. Without it the
    MCP client can't act on UnknownModelResponse."""
    import asyncio

    from whatcanirun.server import mcp

    tools = asyncio.run(mcp.get_tools())
    assert "resolve_model" in tools, (
        f"`resolve_model` tool not registered on `mcp`; registered tools: {sorted(tools)}"
    )
