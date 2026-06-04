# M12 — Release (clone-install, power-user audience)

**Status:** ⬜ Not started
**Effort:** 3h
**Dependencies:** M00–M11
**Unblocks:** v2 conditional triggers (usage signals)

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

`git clone && uv run whatcanirun-mcp` (host-uv path) — or
`docker build && docker run --rm -i whatcanirun-mcp` (Docker path)
— produces a working stdio MCP server that power users can wire
into their MCP client config. A scripted FastMCP `Client` over
`StdioTransport` exercises every tool, resource, and prompt and
audits the trust-envelope invariants on the response shapes.

A new `whatcanirun-mcp prefetch` subcommand runs the upstream
download + index step synchronously so the cold-cache delay is
an observable operator action instead of a hidden first-call
surprise.

PyPI publishing and MCP-registry submissions are **deferred to
v2** — see § Deferred to v2 below. v1 ships as a self-hosted clone
target: the GitHub repo, a tagged `v0.1.0` release, and the
README install block are the only discovery surfaces.

---

## Audience

Power users who:

- have `git`, `uv`, and either Python 3.12 or Docker on their host;
- are comfortable pasting a JSON config block that points at a
  cloned directory or a docker image they built locally;
- want to inspect the source before running it.

GUI users who'd benefit from a one-click `uvx`-style install are
explicitly the *v2 audience*, not v1's. v1 stabilizes the API
through real usage; v2 publishes the artifact once the surface
isn't churning.

---

## Scope

### Slice A — `prefetch` subcommand

Add an argparse layer to `src/whatcanirun/server.py:main`:

- `whatcanirun-mcp` (no args) — current behaviour: `mcp.run(show_banner=False)`.
- `whatcanirun-mcp prefetch` — calls `load_runtime_deps()`
  synchronously, emitting per-source progress to stderr
  (ComputePrices endpoints, HuggingFace `sync_all_tracked`,
  optional Artificial Analysis). Exits 0 on success, non-zero
  with a stderr diagnostic on failure.
- `whatcanirun-mcp --version` — prints `whatcanirun.__version__`
  and exits 0. Used by the clean-machine smoke test.

Tests (`tests/test_prefetch_cli.py`) cover argparse routing
without hitting the network; the prefetch path is exercised
end-to-end by Slice C's release-marked test.

### Slice B1 — host-uv install path

The canonical install for v1:

```bash
git clone https://github.com/maheshbabugorantla/whatcanirun
cd whatcanirun
uv sync
uv run whatcanirun-mcp prefetch
```

MCP client config block points at the cloned directory:

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/abs/path/to/whatcanirun", "whatcanirun-mcp"]
}
```

`scripts/install_host_uv.sh` (idempotent) is the smoke harness
used by Slice C's release gate and quoted directly in the README.

### Slice B2 — Docker install path (fallback)

`Dockerfile` based on `python:3.12-slim`:

- `uv` installed; `uv sync --frozen` for reproducible deps;
- seeds copied in;
- entry point `["whatcanirun-mcp"]`;
- `WORKDIR` and `XDG_CACHE_HOME` set so the named cache volume
  lands at the expected on-disk layout.

`scripts/run_mcp_docker.sh` wraps the launch invocation
(`docker run --rm -i -v whatcanirun-cache:/root/.cache/whatcanirun
-e COMPUTEPRICES_API_KEY -e HF_TOKEN -e AA_API_KEY
whatcanirun:latest`) so the MCP client config block stays a
single-line `command` pointing at the script. Cache lives on a
named volume so the next launch is warm.

No image is published anywhere in v1. Users build locally with
`docker build -t whatcanirun .`. Container-registry publishing
sits with PyPI in § Deferred to v2.

### Slice C — FastMCP stdio release gate

`tests/release/test_stdio_install.py` is marked
`@pytest.mark.release` so it does not run in the default
`pytest -q` CI suite. The gate test:

1. spawns `uv run whatcanirun-mcp` as a subprocess;
2. attaches a `fastmcp.Client` over `StdioTransport`;
3. drives the Phase-3 tool battery from the test plan:
   - `list_catalog()` → GPUs + providers + tracked models;
   - `fit_check(llama-3-1-70b-instruct, h100-80gb, fp16, 1, 1, 4096)`
     → `fits=True`, breakdown sums match weight + KV + overhead,
     `sufficiency_caveat` populated;
   - `find_cheapest_deployment(llama-3-1-8b-instruct)`
     → top-10 sorted ascending by `cost_per_m_output_usd`;
   - `compare_deployment_modes(llama-3-1-8b-instruct, h100-80gb,
     fp16, 1, 4096, chat_short)` → both rows present,
     `workload_assumption` in both envelopes;
   - `budget_to_plan(100.0, llama-3-1-70b-instruct, chat_short)`
     → ranked plan with `est_total_prompts` populated per row.
4. reads both resources (`cost-cells://current`,
   `cost-cells://provenance`) and the `/benchmark-on-budget` prompt;
5. for every numerical response, asserts the trust-envelope
   invariants:
   - `trust_envelope` present;
   - `confidence == min(confidence_breakdown.values())`;
   - `workload_assumption` present iff the response synthesized a
     workload-derived count;
   - `verify_links` non-empty;
   - `freshness` per source matches a real datetime.

Invocation: `pytest -m release` (locally), and in
`scripts/install_host_uv.sh` post-prefetch.

### Slice D — `CHANGELOG.md`

First entry — `## [0.1.0] — 2026-06-XX`. Lists:

- every milestone shipped (M00–M12), with the M10 partial-ship
  note (Tier 1b removed from v1) called out explicitly;
- every ADR locked (ADR-001 through ADR-015), grouped by concern;
- every upstream attributed (ComputePrices, Hugging Face,
  Artificial Analysis, Kiely 2026 *Inference Engineering*
  methodology), with license and verify-links;
- known limitations: Tier 1a `own_measured` deferred to v2 M17;
  Tier 1b `public_benchmark_anchor` removed from v1 and not tied
  to a v2 milestone; TPS heuristic is single-stream only (ADR-010);
- license terms (MIT for the project, CC-BY-4.0 for the
  benchmark dataset per ADR-006);
- explicit "not on PyPI in v1" note pointing at § Deferred to v2.

### Slice E — `docs/MCP.md` flip

Replace all four client config blocks (Claude Desktop, Claude
Code, Cursor, Cline) with the host-uv variant plus a Docker
alternative. Add a top "v1 install" note saying PyPI lands in v2.
Update troubleshooting: `uvx: command not found` →
`uv: command not found`; the cold-cache section refers users to
`whatcanirun-mcp prefetch`.

### Slice F — README install block

A canonical install block at the README top:

1. host-uv path (headline) — clone, sync, prefetch, wire into
   client config;
2. Docker path (fallback) — build, run via the launch script,
   wire into client config;
3. link out to [`docs/MCP.md`](../docs/MCP.md) for per-client
   examples;
4. link out to [`docs/TRUST.md`](../docs/TRUST.md) for the
   trust contract.

### Slice G — Release cut

After PR merge to main:

1. `git tag v0.1.0`, `git push origin v0.1.0`;
2. `gh release create v0.1.0` with the CHANGELOG `[0.1.0]`
   section as the body;
3. flip `spec/INDEX.md` M12 row from ⬜ to ✓ on a small commit;
4. update `docs/PRD.md` M12 row label.

---

## Deferred to v2

Original M12 included PyPI publishing and three MCP-registry
submissions (PulseMCP, mcpservers.org, anthropic/registry).
Those are deferred to v2 because:

- v1 has not been used by anyone but the maintainer; the tool
  signatures, the trust-envelope shape, and the cache layout
  may still need a churn round once real users hit them. A
  published PyPI artifact constrains that churn.
- Registry submissions are discovery surfaces — useful when the
  product is stable, expensive (in maintenance and review-cycle
  time) when it isn't.
- Reserving the `whatcanirun` PyPI name *now* (against
  squatting) is cheap and worth doing; *publishing* `0.1.0` to
  it is not.

When usage signals justify it (the `spec/INDEX.md` v2 trigger
table), v2 will:

1. publish to PyPI as `whatcanirun` v0.2.0 (or higher);
2. add a `whatcanirun-mcp` console script alias if the import
   path differs from the install path;
3. submit to PulseMCP via https://www.pulsemcp.com/submit;
4. open a PR against the mcpservers.org registry repo;
5. open a PR against `anthropic/registry`;
6. (stretch) submit to Glama.ai at https://glama.ai/mcp/servers;
7. publish a docker image to GHCR for the Docker install path
   so users don't need to build locally.

---

## Vertical slices

1. **Slice A:** `prefetch` subcommand + argparse — TDD.
2. **Slice B1:** host-uv install path + `install_host_uv.sh`.
3. **Slice B2:** `Dockerfile` + `run_mcp_docker.sh`.
4. **Slice C:** FastMCP stdio release-gate test.
5. **Slice D:** `CHANGELOG.md` v0.1.0 entry.
6. **Slice E:** flip `docs/MCP.md` client config blocks.
7. **Slice F:** README install block.
8. **Slice G:** post-merge release cut (tag + GitHub Release +
   `spec/INDEX.md` flip).

---

## Acceptance criteria

- [ ] `whatcanirun-mcp --version` prints the current
      `whatcanirun.__version__` and exits 0.
- [ ] `whatcanirun-mcp prefetch` runs `load_runtime_deps()`
      synchronously, emitting per-source progress to stderr, and
      exits 0 on success.
- [ ] `scripts/install_host_uv.sh` runs `uv sync`, `prefetch`, and
      the release-gate test against a fresh clone of the working
      tree and exits 0.
- [ ] `docker build -t whatcanirun .` builds; the resulting image
      starts as a stdio MCP server when launched via
      `scripts/run_mcp_docker.sh`.
- [ ] `tests/release/test_stdio_install.py` passes under
      `pytest -m release`; trust-envelope invariants asserted per
      § Slice C.
- [ ] `docs/MCP.md` shows the clone-install config blocks for all
      four supported clients; no `uvx whatcanirun-mcp` references
      remain except in the "v2 will publish to PyPI" note.
- [ ] `README.md` has the canonical install block at the top.
- [ ] `CHANGELOG.md` exists with the `[0.1.0]` entry.
- [ ] Git tag `v0.1.0` exists on `main` and a GitHub Release with
      the CHANGELOG excerpt is published.
- [ ] `spec/INDEX.md` and `docs/PRD.md` M12 rows flipped to ✓.

---

## Common pitfalls

- **`uv run --directory` doesn't pick up `.env` automatically.**
  The clone-install path inherits env from the client config's
  `env:` block (recommended) or the launching shell (Claude Code
  in a terminal). `docs/MCP.md` already documents this; the
  same caveats apply.
- **Docker stdio + cold cache.** Without a named cache volume,
  every `docker run` is cold. `scripts/run_mcp_docker.sh` MUST
  mount a named volume by default — the v1 cache lives at
  `/root/.cache/whatcanirun` inside the container; the volume
  name `whatcanirun-cache` is the convention.
- **PR description must say "v1 is not on PyPI".** First-time
  visitors who skim the README and look for `pip install
  whatcanirun` need to know they won't find it on the index.
- **Release notes must repeat the M10 partial-ship language.**
  The "Tier 1b removed from v1, NOT tied to v2 M17" detail is
  load-bearing for the trust contract and silently changing it
  in release notes would undo the M11 docs reconciliation.

---

## After M12

v1 is shipped. Take a break. Don't start v2 immediately — wait
for usage signals (the `INDEX.md` v2 trigger conditions). v1
should run for at least 30 days before deciding what's worth
building next.

---

## When done

Commit:
> `M12: v0.1.0 release — clone-install (host-uv + Docker), prefetch CLI, release-gate test, CHANGELOG; PyPI deferred to v2`

Mark M12 ✓ in `INDEX.md`. **v1 is shipped.**
