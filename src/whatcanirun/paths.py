"""Filesystem path conventions for whatcanirun.

`SEEDS_DIR` is the canonical location of the seed YAML/Parquet
data that ships with the project. In editable installs (the dev
default + the `uv pip install -e .` pattern) it resolves to
`<repo_root>/seeds`. Shipping wheels will need to bundle the
seeds inside the package as `src/whatcanirun/seeds/` and update
this constant accordingly — M12's job.

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

# paths.py lives at src/whatcanirun/paths.py — two parents up
# is the repo root in editable installs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_DIR = _PROJECT_ROOT / "seeds"


def _xdg_dir(env_var: str, fallback_base: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    base = Path(raw).expanduser() if raw else fallback_base
    return base / "whatcanirun"


USER_CACHE_DIR = _xdg_dir("XDG_CACHE_HOME", Path.home() / ".cache")
USER_CONFIG_DIR = _xdg_dir("XDG_CONFIG_HOME", Path.home() / ".config")
