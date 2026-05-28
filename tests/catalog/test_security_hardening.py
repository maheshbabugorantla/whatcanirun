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


def test_open_excl_nofollow_refuses_symlink(cache_dir: Path, tmp_path: Path) -> None:
    """The atomic-write tmp paths are no longer predictable (per-
    attempt unique `<path>.<pid>.<token>.tmp` per the concurrent-
    safe refactor), but a local attacker with write access to the
    cache directory could still race to plant a symlink at any path
    the writer is about to open. The boundary defense moves down to
    `_open_excl_nofollow`: any pre-existing entry at the target path
    — file, directory, or symlink — must raise FileExistsError
    without following. Without `O_NOFOLLOW` a planted symlink would
    silently redirect the JSON write to whatever the symlink
    targets (config files, ssh keys, anything the process user can
    write).

    This direct unit test on the private helper proves the defense
    holds independently of the higher-level `_write_atomic` name
    scheme — if future maintenance changes the tmp-naming strategy,
    the symlink refusal still has its own regression coverage."""
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    target = tmp_path / "victim.json"
    target.write_text("original victim contents")

    planted_tmp = hf_dir / "some_predictable_or_raced_name.tmp"
    planted_tmp.symlink_to(target)

    with pytest.raises(FileExistsError):
        HfModelSync._open_excl_nofollow(planted_tmp, b"attacker payload")

    # Victim file MUST NOT have been overwritten via the symlink.
    assert target.read_text() == "original victim contents"


def test_open_excl_nofollow_refuses_existing_regular_file(cache_dir: Path) -> None:
    """O_EXCL must also reject a pre-existing regular file at the
    target tmp path. With unique per-attempt tmp names from
    `_write_atomic`, a collision here implies either a UUID
    collision (~0 probability at 64 random bits) or an attacker
    pre-planting the exact name we're about to use — both worth
    escalating loudly rather than overwriting silently."""
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    occupied = hf_dir / "already_there.tmp"
    occupied.write_text("not ours")

    with pytest.raises(FileExistsError):
        HfModelSync._open_excl_nofollow(occupied, b"new payload")

    assert occupied.read_text() == "not ours"


# -------------------------------------- stale .tmp from prior crash (unique-name era)


@pytest.mark.asyncio
@respx.mock
async def test_stale_tmp_from_prior_crash_does_not_block_sync(
    cache_dir: Path, llama_config: dict[str, Any]
) -> None:
    """Under the unique per-attempt tmp naming
    (`<path>.<pid>.<token>.tmp`), a SIGKILLed prior sync's orphaned
    tmp file bears a name no future sync will ever choose. The
    stale file just sits inert on disk; future syncs allocate
    fresh unique names and proceed cleanly. This test pins the
    behavior so a future regression to predictable-naming would
    fail loudly here as well as in the concurrency tests."""
    repo_id = "meta-llama/Llama-3.3-70B-Instruct"
    sha = "abc"
    respx.get("https://huggingface.co/api/models/meta-llama/Llama-3.3-70B-Instruct").mock(
        return_value=httpx.Response(200, json={"sha": sha, "modelId": repo_id})
    )
    respx.get(f"https://huggingface.co/{repo_id}/raw/{sha}/config.json").mock(
        return_value=httpx.Response(200, json=llama_config)
    )

    # Plant an orphan tmp bearing the legacy predictable suffix.
    # Under the old scheme this would perma-block; under unique-
    # names it's simply ignored.
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    stale_tmp = hf_dir / "llama-3-3-70b.model.json.tmp"
    stale_tmp.write_text("leftover from a SIGKILLed prior sync")

    sync = HfModelSync(cache_dir=cache_dir)
    model = await sync.sync_model(
        repo_id=repo_id,
        slug="llama-3-3-70b",
        display_name="Llama",
        total_params_b=70.6,
        active_params_b=None,
    )

    # Sync succeeded despite the stale tmp.
    assert model.slug == "llama-3-3-70b"
    assert (hf_dir / "llama-3-3-70b.model.json").exists()


# -------------------------------------- concurrent writes to the same slug


@pytest.mark.asyncio
async def test_concurrent_writes_to_same_slug_do_not_destroy_each_other(
    cache_dir: Path,
) -> None:
    """Per-attempt unique tmp names mean two writers targeting the
    same final path each use distinct tmp paths — writer B can
    never unlink writer A's active tmp. Whichever `tmp.replace(
    path)` runs last publishes the final payload; the other's tmp
    is dropped by the replace's atomicity. Neither path errors.

    Without unique tmps, the stale-tmp recovery approach would
    race destructively: B sees A's tmp, treats it as stale,
    unlinks it, A's subsequent rename fails with ENOENT.

    This is the concurrent-safety property Copilot's round-4
    review flagged. The test serializes the two writes (no real
    threading needed) but they target the same final cache file
    and back-to-back invocations cover the race condition the
    unique-tmp design eliminates by construction."""
    hf_dir = cache_dir / "huggingface"
    hf_dir.mkdir(parents=True)
    final_path = hf_dir / "shared.model.json"

    HfModelSync._write_atomic(final_path, b"writer A payload")
    # No leftover tmp from writer A pollutes writer B's namespace
    # because the suffix is unique each call.
    HfModelSync._write_atomic(final_path, b"writer B payload")

    assert final_path.read_bytes() == b"writer B payload"
    # Any tmp files left behind would have unique names; the final
    # cache file is the only `*.json` we expect to find.
    assert sorted(p.name for p in hf_dir.iterdir() if p.name.endswith(".json")) == [
        "shared.model.json"
    ]


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
