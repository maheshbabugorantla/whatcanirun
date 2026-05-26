# M12 — Release (PyPI + Registry Submissions)

**Status:** ⬜ Not started
**Effort:** 3h
**Dependencies:** M00–M11
**Unblocks:** v2 conditional triggers (usage signals)

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

`uvx whatcanirun-mcp` works from a clean machine. The package is published to PyPI. The server is listed in three MCP registries: PulseMCP, mcpservers.org, and the official `anthropic/registry` repo. Glama.ai is a stretch goal.

---

## Scope

### Pre-release checklist

1. **Clean-machine test** — in a fresh Docker container (NOT the sandbox), run:
   ```bash
   docker run --rm -it python:3.12-slim bash
   pip install uv
   uvx whatcanirun-mcp --version
   uvx whatcanirun-mcp  # confirm stdio handshake works
   ```
   If this fails, the release is not ready. Common cause: missing data files in the wheel (check `tool.hatch.build.targets.wheel.packages` and `include`).

2. **Seed data bundling** — `seeds/*.yaml` and `seeds/benchmark_cells.parquet` MUST be in the wheel. Verify with:
   ```bash
   uv pip download whatcanirun==0.1.0 --no-deps -d /tmp/whl
   unzip -l /tmp/whl/whatcanirun-0.1.0-*.whl | grep seeds/
   ```

3. **Version bump** — `pyproject.toml` version from `0.0.1` to `0.1.0` (first public release).

4. **CHANGELOG.md** — first entry. Lists every milestone shipped, all ADRs locked, all upstream sources with attribution.

5. **License finalized** — confirm `MIT` (or chosen alternative) in `pyproject.toml` and `LICENSE` file.

### Publish to PyPI

```bash
uv build                          # produces dist/whatcanirun-0.1.0.tar.gz and .whl
uv publish                        # uses PYPI_TOKEN env var (or interactive)
```

Verify on https://pypi.org/project/whatcanirun/ within 5 minutes.

### Registry submissions

1. **PulseMCP** — https://www.pulsemcp.com/submit
   Fields: name, description, GitHub URL, install command, screenshot (optional but recommended — show a Claude Desktop conversation using `budget_to_plan`)

2. **mcpservers.org** — open a PR against the registry repo (likely `mcp-servers/mcp-servers` or similar; check at submission time)
   Required: server name, description, install command, source URL, license

3. **anthropic/registry** — open a PR against the official Anthropic MCP registry
   Follow their CONTRIBUTING.md; likely requires:
   - Manifest JSON with tool/resource/prompt schemas
   - Demo screenshot
   - Maintainer contact

4. **Glama.ai** — submit at https://glama.ai/mcp/servers (stretch — may require manual review delay)

### Post-release

- Tweet/social announcement (optional)
- GitHub release with CHANGELOG.md excerpt
- Pin the install instructions to the repo top

---

## Vertical slices

1. **Slice A: Clean-machine test infrastructure** — script in `scripts/test_clean_machine.sh` that spins up the Docker container and runs the smoke test
2. **Slice B: Seed bundling** — fix `pyproject.toml` if seeds aren't in the wheel; verify with the unzip command
3. **Slice C: CHANGELOG.md** — write it; commit
4. **Slice D: Version bump + release commit** — tag `v0.1.0`
5. **Slice E: PyPI publish** — `uv publish`; verify; smoke test from PyPI
6. **Slice F: PulseMCP submission** — fill out the form, screenshot in hand
7. **Slice G: mcpservers.org PR** — open the PR
8. **Slice H: anthropic/registry PR** — open the PR

---

## Acceptance criteria

- [ ] `uvx whatcanirun-mcp --version` works from a clean Python 3.12 container, no pre-installed deps
- [ ] Seeds (`seeds/*.yaml`, `benchmark_cells.parquet`) are bundled in the wheel
- [ ] PyPI listing live at https://pypi.org/project/whatcanirun/
- [ ] `CHANGELOG.md` exists with v0.1.0 entry
- [ ] Git tag `v0.1.0` pushed; GitHub Release published
- [ ] PulseMCP submission accepted (verify within 24h)
- [ ] mcpservers.org PR opened (merge may take longer)
- [ ] anthropic/registry PR opened (merge may take longer)

---

## Common pitfalls

- **Wheel missing seed data.** `hatchling` doesn't include non-Python files by default. Add to `pyproject.toml`:
  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["src/whatcanirun"]
  include = ["seeds/*.yaml", "seeds/*.parquet"]
  ```
- **PyPI name squatting.** Reserve `whatcanirun` (or your chosen name) BEFORE M12 — earlier in the project, even before M00. If someone else takes it, you have to rename everything.
- **AA attribution missing.** Their ToS requires attribution on every product surface using their data. Confirm `docs/TRUST.md`, `README.md`, and `cost-cells://provenance` all name Artificial Analysis with the link.
- **CI green ≠ release-ready.** CI uses fixtures. The clean-machine test uses live `uvx` install. Always run both.

---

## After M12

You shipped v1. Take a break. Don't start v2 immediately — wait for usage signals (the `INDEX.md` v2 trigger conditions). v1 should run for at least 30 days before you decide what's worth building next.

---

## When done

Commit:
> `M12: v0.1.0 release — published to PyPI, listed on PulseMCP/mcpservers.org/anthropic-registry`

Mark M12 ✓ in `INDEX.md`. **v1 is shipped.**
