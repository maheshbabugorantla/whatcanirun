"""Security-hardening regression tests for HfModelSync.

These cover two specific attack scenarios from the M03 pre-push
security review:
  1. TOCTOU symlink redirect via predictable `.tmp` path
  2. CRLF injection via HF_TOKEN env var
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from whatcanirun.catalog.hf_sync import HfModelSync


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cp"


@pytest.fixture
def llama_config() -> dict[str, Any]:
    import json

    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    return json.loads((fixtures / "hf_llama-3-3-70b_config.json").read_text())


# -------------------------------------------------- TOCTOU on the .tmp path


@pytest.mark.asyncio
@respx.mock
async def test_tmp_path_symlink_redirect_is_refused(
    cache_dir: Path, llama_config: dict[str, Any], tmp_path: Path
) -> None:
    """The cache `.tmp` filename is predictable
    (`<cache_dir>/huggingface/<slug>.model.json.tmp`). A local
    attacker who can plant a symlink there before the sync runs
    could redirect the JSON write to any file the user can write
    (config files, ssh keys, etc).

    Defense: the atomic write must use O_EXCL | O_NOFOLLOW so that
    a pre-existing file or symlink at the tmp path causes the write
    to fail rather than silently follow the redirect."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    respx.get("https://huggingface.co/api/models/meta-llama/Llama-3.3-70B-Instruct").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"https://huggingface.co/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    # Pre-create the cache dir and plant a symlink at the predictable
    # .tmp path pointing to an attacker-controlled target outside the
    # cache.
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    target = tmp_path / "victim.json"
    target.write_text("original victim contents")
    tmp_link = hf_dir / "llama-3-3-70b.model.json.tmp"
    tmp_link.symlink_to(target)

    sync = HfModelSync(cache_dir=cache_dir)
    with pytest.raises(FileExistsError):
        await sync.sync_model(
            repo_id=repo_id,
            slug="llama-3-3-70b",
            display_name="Llama",
            total_params_b=70.6,
            active_params_b=None,
        )

    # Victim file MUST NOT have been overwritten by the model JSON.
    assert target.read_text() == "original victim contents"


# -------------------------------------------------- HF_TOKEN CRLF injection


def test_constructor_rejects_hf_token_with_carriage_return(cache_dir: Path) -> None:
    """A token containing `\\r` could enable a CRLF header injection
    on httpx implementations that don't validate the value (this is
    bounded — modern h11 rejects — but defense-in-depth catches it
    at the boundary instead of relying on the transport)."""
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HfModelSync(cache_dir=cache_dir, hf_token="good_token\rX-Injected: evil")


def test_constructor_rejects_hf_token_with_newline(cache_dir: Path) -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HfModelSync(cache_dir=cache_dir, hf_token="good_token\nX-Injected: evil")


def test_constructor_rejects_hf_token_with_null_byte(cache_dir: Path) -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HfModelSync(cache_dir=cache_dir, hf_token="good_token\x00null")


def test_env_var_with_crlf_also_rejected(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same validation applies whether the token comes via ctor arg
    or env var — the env source isn't a more-trusted path."""
    monkeypatch.setenv("HF_TOKEN", "good\rX-Injected: x")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HfModelSync(cache_dir=cache_dir)


def test_constructor_accepts_normal_token(cache_dir: Path) -> None:
    """Sanity: normal tokens pass through. Both ASCII letters and
    the dot-underscore-dash chars HF uses in bearer tokens are
    allowed."""
    sync = HfModelSync(cache_dir=cache_dir, hf_token="hf_abcDEF123-_.")
    headers = sync._headers()
    assert headers["Authorization"] == "Bearer hf_abcDEF123-_."
