"""M12 Slice A — argparse routing for `whatcanirun-mcp`.

The entry point gains two subcommands beyond the default stdio
launch:

- `whatcanirun-mcp --version` — prints the package version and
  exits. Used by the clean-machine smoke test and gives users
  a way to confirm the installed checkout matches what their
  client is launching.
- `whatcanirun-mcp prefetch` — runs `load_runtime_deps()` and
  `HfModelSync.sync_all_tracked` synchronously with stderr
  progress, so the cold-cache download/index step is an
  observable operator action instead of a hidden first-call
  surprise on the MCP `tools/call`.

The bare invocation (`whatcanirun-mcp` with no args) keeps the
existing behaviour: spin up the FastMCP stdio loop. That's the
mode every MCP client uses; argparse must not break it.

These tests stub the network-touching handlers (`mcp.run` and
the prefetch function) so the routing layer is exercised in
isolation. End-to-end prefetch coverage lives in the M12 Slice C
release-gate test (`@pytest.mark.release`).
"""

from __future__ import annotations

import contextlib
import io
import sys
from typing import Any

import pytest

import whatcanirun.server as server_mod
from whatcanirun import __version__


def test_main_no_args_starts_stdio_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """`whatcanirun-mcp` with no args drives the FastMCP stdio
    loop with `show_banner=False`. Every MCP client launches the
    binary this way; argparse routing must not break it or change
    the kwargs the existing handshake depends on."""
    calls: list[dict[str, Any]] = []

    def _fake_run(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(server_mod.mcp, "run", _fake_run)
    server_mod.main(argv=[])
    assert calls == [{"show_banner": False}]


def test_main_version_flag_prints_package_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`whatcanirun-mcp --version` prints the package's
    `__version__` and exits 0. The clean-machine smoke test uses
    it to confirm a successful install before launching the
    stdio loop. It must NOT trigger `mcp.run` or `prefetch`."""
    mcp_calls: list[Any] = []
    prefetch_calls: list[Any] = []
    monkeypatch.setattr(server_mod.mcp, "run", lambda **kw: mcp_calls.append(kw))
    monkeypatch.setattr(server_mod, "_run_prefetch", lambda: prefetch_calls.append(None) or 0)

    with pytest.raises(SystemExit) as exc_info:
        server_mod.main(argv=["--version"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert __version__ in captured.out
    assert mcp_calls == []
    assert prefetch_calls == []


def test_main_prefetch_subcommand_invokes_prefetch_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`whatcanirun-mcp prefetch` runs the synchronous warmup
    handler and exits with its return code. The handler does the
    heavy lifting (CP + HF + AA fetches); the routing layer just
    has to wire the subcommand to it and propagate the exit
    code so a clean-machine script can branch on success."""
    mcp_calls: list[Any] = []
    prefetch_called = False

    def _fake_prefetch() -> int:
        nonlocal prefetch_called
        prefetch_called = True
        return 0

    monkeypatch.setattr(server_mod.mcp, "run", lambda **kw: mcp_calls.append(kw))
    monkeypatch.setattr(server_mod, "_run_prefetch", _fake_prefetch)

    with pytest.raises(SystemExit) as exc_info:
        server_mod.main(argv=["prefetch"])
    assert exc_info.value.code == 0
    assert prefetch_called is True
    assert mcp_calls == []


def test_main_prefetch_propagates_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the prefetch handler returns non-zero (e.g. CP is down
    + HF is unreachable + no cached fallback), the CLI exits with
    that code so a wrapper script sees the failure. Without this,
    `scripts/install_host_uv.sh` would silently continue past a
    failed warmup and the release-gate test would then fail with
    a less-actionable error than the upstream the prefetch tried."""
    monkeypatch.setattr(server_mod, "_run_prefetch", lambda: 2)
    monkeypatch.setattr(server_mod.mcp, "run", lambda **kw: None)

    with pytest.raises(SystemExit) as exc_info:
        server_mod.main(argv=["prefetch"])
    assert exc_info.value.code == 2


def test_main_unknown_subcommand_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`whatcanirun-mcp bogus` is a user typo, not silent
    fall-through to stdio launch. Argparse handles this by exiting
    2 with a usage message on stderr — the test pins that
    contract so a future refactor can't accidentally make typos
    launch the server."""
    monkeypatch.setattr(server_mod.mcp, "run", lambda **kw: None)
    monkeypatch.setattr(server_mod, "_run_prefetch", lambda: 0)

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as exc_info:
        server_mod.main(argv=["bogus"])
    assert exc_info.value.code == 2


def test_main_default_argv_uses_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """When called without `argv` (the production path —
    `pyproject.toml` registers `whatcanirun-mcp = whatcanirun.server:main`
    which calls `main()` with no arguments), routing reads from
    `sys.argv[1:]`. This test guards against a refactor that
    drops the default and silently breaks the installed script."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(server_mod.mcp, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["whatcanirun-mcp"])
    server_mod.main()
    assert calls == [{"show_banner": False}]
