"""Security-hardening regression tests for ArtificialAnalysisClient.

Two specific scenarios from the M04 pre-push security review:
  1. AA_API_KEY CRLF / NUL injection — defense-in-depth at the
     boundary even though h11 rejects these at serialization (matches
     M03's HF_TOKEN posture).
  2. Snapshot write race — two concurrent refreshes within the same
     wall-clock second must not produce a corrupt `.json.gz` file
     (atomic write via per-attempt-unique tmp + rename).
"""

from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from whatcanirun.pricing.artificial_analysis import ArtificialAnalysisClient


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "aa"


# -------------------------------------------------- AA_API_KEY CRLF injection


def test_constructor_rejects_aa_key_with_carriage_return(cache_dir: Path) -> None:
    """A key containing `\\r` could enable CRLF header injection on
    transports that don't validate (httpx's h11 does, but defense-
    in-depth catches it at the boundary). Matches M03's HF_TOKEN
    rejection."""
    with pytest.raises(ValueError, match="AA_API_KEY"):
        ArtificialAnalysisClient(cache_dir=cache_dir, api_key="good_key\rX-Injected: evil")


def test_constructor_rejects_aa_key_with_newline(cache_dir: Path) -> None:
    with pytest.raises(ValueError, match="AA_API_KEY"):
        ArtificialAnalysisClient(cache_dir=cache_dir, api_key="good_key\nX-Injected: evil")


def test_constructor_rejects_aa_key_with_null_byte(cache_dir: Path) -> None:
    with pytest.raises(ValueError, match="AA_API_KEY"):
        ArtificialAnalysisClient(cache_dir=cache_dir, api_key="good_key\x00null")


def test_env_var_with_crlf_also_rejected(cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same validation regardless of whether the key comes via
    ctor arg or env var. The env path isn't a more-trusted source —
    a stray CRLF in a shell wrapper deserves the same rejection."""
    monkeypatch.setenv("AA_API_KEY", "good\rX-Injected: x")
    with pytest.raises(ValueError, match="AA_API_KEY"):
        ArtificialAnalysisClient(cache_dir=cache_dir)


def test_constructor_accepts_normal_key(cache_dir: Path) -> None:
    """Sanity: normal printable ASCII keys pass through unchanged."""
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="aa_test_KEY-_.0123")
    assert client._api_key == "aa_test_KEY-_.0123"


# ----------------------------------------- concurrent snapshot writes


def test_snapshot_writes_use_unique_tmp_and_dont_corrupt_each_other(
    cache_dir: Path,
) -> None:
    """`_write_snapshot` uses per-attempt-unique tmp + rename. Two
    concurrent refreshes within the same ISO-8601 second would
    otherwise collide on the destination path — one writer
    truncating the other's bytes mid-write produces a corrupt
    `.json.gz` that `_recover_from_disk` silently skips. With
    atomic writes the destination is always either the old
    contents or one of the two complete new payloads — never a
    partial mix.

    Pin the property by issuing many concurrent writes to the
    same client (same _now() second by happenstance) and asserting
    every snapshot file on disk decompresses + parses cleanly.
    """
    client = ArtificialAnalysisClient(cache_dir=cache_dir, api_key="k")
    # Pre-create the snapshots dir.
    snapshots_dir = client._snapshots_dir()
    snapshots_dir.mkdir(parents=True)

    payload = b'{"status": 200, "data": []}'

    def writer(_: int) -> None:
        client._write_snapshot(payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(32)))

    snapshots = list(snapshots_dir.glob("*.json.gz"))
    assert len(snapshots) >= 1, "no snapshots persisted at all"
    for snap in snapshots:
        # Each surviving snapshot must decompress + match the
        # payload byte-for-byte. A partial-write corruption would
        # raise BadGzipFile here, OR produce truncated bytes.
        with gzip.open(snap, "rb") as f:
            assert f.read() == payload, f"corrupt snapshot at {snap}"
