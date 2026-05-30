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
    in the module docstring."""
    # 1. Explicit env override — the wheel-install workaround.
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
    return Path(__file__).resolve().parent.parent.parent / "seeds"


SEEDS_DIR = _resolve_seeds_dir()


def _xdg_dir(env_var: str, fallback_base: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    base = Path(raw).expanduser() if raw else fallback_base
    return base / "whatcanirun"


USER_CACHE_DIR = _xdg_dir("XDG_CACHE_HOME", Path.home() / ".cache")
USER_CONFIG_DIR = _xdg_dir("XDG_CONFIG_HOME", Path.home() / ".config")
