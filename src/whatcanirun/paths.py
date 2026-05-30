"""Filesystem path conventions for whatcanirun.

`SEEDS_DIR` resolution order:

  1. `WHATCANIRUN_SEEDS_DIR` env var (override) — wins if set and
     non-empty. The stopgap that lets `uvx whatcanirun-mcp` users
     point at a seeds checkout outside the wheel until M12 ships
     packaged seeds.
  2. Packaged `whatcanirun/seeds/` — present in the wheel after
     M12's `[tool.hatch.build.targets.wheel] force-include` lands.
     Not present in editable installs.
  3. `<repo_root>/seeds` — the dev default. Lives two parents up
     from this file since paths.py is at `src/whatcanirun/paths.py`.

`USER_CACHE_DIR` and `USER_CONFIG_DIR` follow XDG with sensible
fallbacks: `~/.cache/whatcanirun` and `~/.config/whatcanirun` on
Linux/macOS. Sub-directories per upstream source live underneath
(`USER_CACHE_DIR/computeprices/` for CP, `USER_CACHE_DIR/huggingface/`
for HF). `USER_CONFIG_DIR/user_models.yaml` is where Slice L's
`resolve_model` tool persists user-elicited HF repo IDs.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_seeds_dir() -> Path:
    """Pick the right seeds directory per the resolution order
    in the module docstring. Raises `RuntimeError` with an
    actionable message if NONE of the three options yields an
    existing directory — the alternative was a non-existent path
    surfacing later as `SeedLoadError` from inside a tool call,
    which made the root cause much harder to diagnose."""
    # 1. Explicit env override — the wheel-install workaround.
    # Trust the user-supplied path even if it doesn't exist yet
    # (they may have set it pointing to a future location). Tools
    # that try to read the path will surface the missing-file
    # error themselves with the right context.
    raw_override = os.environ.get("WHATCANIRUN_SEEDS_DIR", "").strip()
    if raw_override:
        return Path(raw_override).expanduser()

    # 2. Packaged seeds — only present after M12's wheel-build
    # change lands. Until then this directory doesn't exist; the
    # check is forward-compat scaffolding so the env var path
    # above can stay the workaround and this path becomes the
    # default once M12 ships.
    packaged = Path(__file__).resolve().parent / "seeds"
    if packaged.is_dir():
        return packaged

    # 3. Repo-root seeds — the dev / editable-install default.
    # paths.py lives at src/whatcanirun/paths.py — two parents up
    # is the repo root.
    repo_default = Path(__file__).resolve().parent.parent.parent / "seeds"
    if repo_default.is_dir():
        return repo_default

    # No candidate matched — fail fast with a clear message
    # naming all three resolution options the user can fix. The
    # alternative was returning the non-existent repo_default,
    # which would surface later as `SeedLoadError`/`FileNotFoundError`
    # from inside the first tool call that touched the catalog —
    # a confusing tooltip far from the root cause.
    raise RuntimeError(
        "whatcanirun could not locate a seeds directory.\n"
        "Resolution order tried:\n"
        f"  1. $WHATCANIRUN_SEEDS_DIR (env override): unset or empty\n"
        f"  2. packaged whatcanirun/seeds/ (post-M12 wheel): not found at {packaged}\n"
        f"  3. <repo-root>/seeds (editable install): not found at {repo_default}\n"
        "\n"
        "If you're using a wheel install (e.g. `uvx whatcanirun-mcp`), set "
        "WHATCANIRUN_SEEDS_DIR to a checkout of the project's `seeds/` directory "
        "until M12 ships packaged seeds in the wheel."
    )


# `SEEDS_DIR` is evaluated at import time. Any module that does
# `from whatcanirun.paths import SEEDS_DIR` caches the value at
# its own import. A test that sets `WHATCANIRUN_SEEDS_DIR` AFTER
# the consumer has imported `SEEDS_DIR` won't see the override.
# Tests redirecting seeds should either (a) call
# `_resolve_seeds_dir()` directly (the pattern in
# `tests/test_paths.py`), (b) monkeypatch each consumer's bound
# `SEEDS_DIR` (the pattern integration tests use via
# `_redirect_xdg`), or (c) set the env var BEFORE any whatcanirun
# module is imported.
SEEDS_DIR = _resolve_seeds_dir()


def _xdg_dir(env_var: str, fallback_base: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    base = Path(raw).expanduser() if raw else fallback_base
    return base / "whatcanirun"


USER_CACHE_DIR = _xdg_dir("XDG_CACHE_HOME", Path.home() / ".cache")
USER_CONFIG_DIR = _xdg_dir("XDG_CONFIG_HOME", Path.home() / ".config")
