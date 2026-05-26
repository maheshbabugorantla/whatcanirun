"""Smoke test: entry point runs and prints the version."""

from __future__ import annotations

import pytest

from whatcanirun import __version__
from whatcanirun.server import main


def test_main_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    out = capsys.readouterr().out
    assert __version__ in out
