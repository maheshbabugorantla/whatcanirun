# M00 — Project Bootstrap

**Status:** ⬜ Not started
**Effort:** 4h
**Dependencies:** None
**Unblocks:** Everything else

> Read [`SHARED.md`](SHARED.md) before starting. This spec assumes you've internalized the design contract, ADRs, and domain glossary.

---

## Goal

A new Python project that boots cleanly. `uvx whatcanirun-mcp --help` runs. `pytest -q` reports 0 tests, 0 failures. CI is green on the first push. The mattpocock skills workflow is wired and `/to-prd spec/M01-catalog-supplements.md` produces an issue.

Nothing in this milestone touches FastMCP business logic or upstream APIs. It is pure scaffolding — but the scaffolding has to be *right* because every later milestone inherits it.

---

## Scope

- `pyproject.toml` configured with the dependencies in `SHARED.md` § Project layout (already drafted in repo root).
- `uv sync --all-extras --dev` produces a clean lockfile.
- `src/whatcanirun/` package skeleton with empty `__init__.py` in each subdirectory.
- `src/whatcanirun/server.py` with a placeholder `main()` that prints version and exits.
- `tests/conftest.py` with shared fixtures (none needed yet, but the file exists).
- `.pre-commit-config.yaml` wired with ruff, mypy, basic hygiene hooks (already drafted).
- `.github/workflows/ci.yml` runs ruff + mypy + pytest on every push (already drafted).
- Pre-commit installed locally: `uv run pre-commit install`.
- `.claude/skills-lock.json` references mattpocock skills (already drafted).
- `/setup-matt-pocock-skills` run successfully in Claude Code.

---

## Out of scope

- Any business logic (no FastMCP tools, no Pydantic models for domain entities).
- Any upstream API integration (no httpx clients yet).
- Any seed YAML files (those land in M01–M05).
- Any tests beyond a smoke test that `python -m whatcanirun.server --help` works.

---

## Vertical slices (use `/to-issues` to produce these)

1. **Slice A: pyproject.toml + uv lockfile**
   - Run `uv sync --all-extras --dev`. Confirm `uv.lock` is generated.
   - Commit `uv.lock`.
   - Verify `uv run python -c "import fastmcp, pydantic, httpx, duckdb, huggingface_hub, pyarrow, yaml"` succeeds.

2. **Slice B: Package skeleton**
   - Create `src/whatcanirun/__init__.py` with `__version__ = "0.0.1"`.
   - Create empty `__init__.py` files in each subdirectory (`catalog/`, `pricing/`, `inference/`, `plan/`, `trust/`, `mcp_tools/`).
   - Create `src/whatcanirun/server.py` with a stub:
     ```python
     def main() -> None:
         """Entry point for `uvx whatcanirun-mcp`. M00 stub — replaced in M09."""
         from whatcanirun import __version__
         print(f"whatcanirun v{__version__} — MCP server not yet implemented (M09)")
     ```
   - Confirm `uv run whatcanirun-mcp` executes.

3. **Slice C: Tooling**
   - `uv run ruff check .` — exits 0.
   - `uv run ruff format --check .` — exits 0.
   - `uv run mypy src` — exits 0 (empty modules pass strict mode).
   - `uv run pytest -q` — exits 0 with "no tests ran".

4. **Slice D: Pre-commit**
   - `uv run pre-commit install`.
   - Stage a deliberately malformed file (extra whitespace, missing newline at EOF) and confirm pre-commit catches it.
   - Unstage. Move on.

5. **Slice E: CI**
   - Push branch. Confirm GitHub Actions runs the `test` job to green.
   - Push a deliberately failing commit (e.g., add an unused import). Confirm CI fails.
   - Fix. Push. Confirm green again.

6. **Slice F: Skills workflow**
   - Inside the sandbox: `claude` to launch Claude Code.
   - Run `/setup-matt-pocock-skills`. Confirm the skill files appear in `~/.claude/skills/`.
   - Run `/to-prd spec/M00-bootstrap.md` as a smoke test. It should produce a draft issue body referring to this file. Discard the issue (we're already past M00); confirm the workflow works.

---

## Acceptance criteria

- [ ] `docker compose -f compose.claude.yml build` succeeds.
- [ ] `docker compose -f compose.claude.yml run --rm claude-code claude --version` prints a version.
- [ ] `uv sync --all-extras --dev` produces a committed `uv.lock`.
- [ ] `uv run whatcanirun-mcp` prints the version stub.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q` all exit 0.
- [ ] `pre-commit run --all-files` passes on a fresh clone.
- [ ] First push to GitHub: CI green within 5 minutes.
- [ ] `.claude/skills-lock.json` references all 14 mattpocock skills (already present in repo).
- [ ] `/setup-matt-pocock-skills` ran without error.
- [ ] `CLAUDE.md` exists and is referenced from the repo root (already drafted).

---

## Common pitfalls

- **Python version mismatch.** `pyproject.toml` requires 3.12. `uv python install 3.12` if your default is older.
- **CI cache stale.** First CI run can be slow (no uv cache). Subsequent runs hit the cached venv.
- **Pre-commit on uv lockfile.** `uv.lock` should NOT be reformatted by ruff. Verify it's not in `.pre-commit-config.yaml`'s scope.
- **Docker sandbox UID mismatch.** On macOS / WSL the volume driver handles UID translation. On native Linux with non-1000 UID, edit `user: "1000:1000"` in `compose.claude.yml` to match your host.

---

## When this is done

Commit message for the M00 PR:
> `M00: project bootstrap — uv, FastMCP scaffold, CI, mattpocock skills wired`

Mark M00 as ✓ in `spec/INDEX.md`. Move to M01.
