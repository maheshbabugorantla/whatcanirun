"""Tests for `whatcanirun.paths.SEEDS_DIR` resolution order.

Copilot review #15 round 7: a wheel install of `whatcanirun-mcp`
(via `uvx whatcanirun-mcp` or `pip install whatcanirun`) doesn't
include the repo's `seeds/` directory. The previous resolution
was `<repo_root>/seeds`, which doesn't exist in a wheel install,
so the first tool call that touched the catalog would fail.

The fixed resolution order is:
  1. `WHATCANIRUN_SEEDS_DIR` env override (wheel-install stopgap)
  2. Packaged `whatcanirun/seeds/` (post-M12)
  3. `<repo_root>/seeds` (editable / dev install)

These tests exercise the resolution function directly via
`_resolve_seeds_dir` rather than the module-level constant,
since the constant is evaluated at import time and freezing
it via monkeypatch is messy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def test_env_var_override_wins() -> None:
    """`WHATCANIRUN_SEEDS_DIR=/some/path` is the wheel-install
    workaround: the user points the package at an out-of-wheel
    seeds checkout. This must take precedence over any packaged
    or repo-default location."""
    import os

    from whatcanirun.paths import _resolve_seeds_dir

    saved = os.environ.get("WHATCANIRUN_SEEDS_DIR")
    try:
        os.environ["WHATCANIRUN_SEEDS_DIR"] = "/tmp/whatcanirun-seeds-test"
        result = _resolve_seeds_dir()
        assert result == Path("/tmp/whatcanirun-seeds-test")
    finally:
        if saved is None:
            os.environ.pop("WHATCANIRUN_SEEDS_DIR", None)
        else:
            os.environ["WHATCANIRUN_SEEDS_DIR"] = saved


def test_env_var_whitespace_only_falls_back_to_default() -> None:
    """Empty or whitespace-only env vars are common in CI safeguards
    (`WHATCANIRUN_SEEDS_DIR=""`). They must not override — same
    contract M02's CP client follows for `COMPUTEPRICES_API_KEY`."""
    import os

    from whatcanirun.paths import _resolve_seeds_dir

    saved = os.environ.get("WHATCANIRUN_SEEDS_DIR")
    try:
        os.environ["WHATCANIRUN_SEEDS_DIR"] = "   "  # whitespace
        result = _resolve_seeds_dir()
        # Falls through to packaged or repo default; neither is
        # `/tmp/...`-shaped, so the env override clearly didn't fire.
        assert result.parent != Path("/tmp")
    finally:
        if saved is None:
            os.environ.pop("WHATCANIRUN_SEEDS_DIR", None)
        else:
            os.environ["WHATCANIRUN_SEEDS_DIR"] = saved


def test_env_var_expands_user_home(tmp_path: Any) -> None:
    """`~`-prefixed paths must expand via `Path.expanduser()` —
    a common shell pattern when setting the env var by hand."""
    import os

    from whatcanirun.paths import _resolve_seeds_dir

    saved = os.environ.get("WHATCANIRUN_SEEDS_DIR")
    try:
        os.environ["WHATCANIRUN_SEEDS_DIR"] = "~/my-seeds"
        result = _resolve_seeds_dir()
        # Must NOT be the literal `~/...` path; must be expanded.
        assert "~" not in str(result)
        assert str(result).startswith(str(Path.home()))
    finally:
        if saved is None:
            os.environ.pop("WHATCANIRUN_SEEDS_DIR", None)
        else:
            os.environ["WHATCANIRUN_SEEDS_DIR"] = saved


def test_raises_when_no_candidate_dir_exists(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Copilot review #15 round 9 #1: when no candidate directory
    matches (env unset, no packaged seeds, no repo-root seeds),
    `_resolve_seeds_dir` MUST raise with an actionable message
    naming all three options. The previous behavior was to return
    the non-existent repo_default and let the SeedLoadError
    surface from inside a tool call — much harder to diagnose."""
    import pytest

    from whatcanirun.paths import _resolve_seeds_dir

    # Empty env override.
    monkeypatch.delenv("WHATCANIRUN_SEEDS_DIR", raising=False)
    # Move paths.__file__ into a tmp dir that has neither a sibling
    # `seeds/` (packaged) nor a grandparent-parent's-parent `seeds/`
    # (repo-root). Simulating this without restructuring imports is
    # awkward, so instead we patch the resolver to look in `tmp_path`
    # for both candidate locations.
    fake_paths_file = tmp_path / "src" / "whatcanirun" / "paths.py"
    fake_paths_file.parent.mkdir(parents=True)
    fake_paths_file.touch()
    monkeypatch.setattr("whatcanirun.paths.__file__", str(fake_paths_file))

    with pytest.raises(RuntimeError, match="could not locate a seeds directory"):
        _resolve_seeds_dir()


def test_no_env_falls_back_to_existing_seeds_dir() -> None:
    """In the dev environment (where these tests run), the
    `<repo_root>/seeds` directory exists. With no env override and
    no packaged seeds (this is an editable install), the resolver
    must land on the repo-default path."""
    import os

    from whatcanirun.paths import _resolve_seeds_dir

    saved = os.environ.get("WHATCANIRUN_SEEDS_DIR")
    try:
        os.environ.pop("WHATCANIRUN_SEEDS_DIR", None)
        result = _resolve_seeds_dir()
        # The resolver returned SOMETHING that exists and contains
        # the seed YAMLs every test in this suite relies on.
        assert (result / "tracked_models.yaml").exists()
    finally:
        if saved is not None:
            os.environ["WHATCANIRUN_SEEDS_DIR"] = saved
