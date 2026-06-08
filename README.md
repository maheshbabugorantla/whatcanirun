# whatcanirun

An MCP server that answers the one question every LLM inference user actually asks:

> **"I have $X to spend on model Y — what can I actually run?"**

Self-hosted, free public APIs only, ships as a stdio MCP server you clone and run on your own host. No accounts, no hosting, no telemetry.

## What it does

Given a dollar budget and an open-weight model, returns a ranked, source-backed plan: which GPU on which provider, whether the model fits in VRAM, how many tokens per second you'll get, how many prompts that buys you, and exactly which assumptions and caveats underlie every number.

Every response carries a structured `trust_envelope` — sources, per-domain confidence, assumptions, caveats, freshness, and audit links. Hobbyist or power user, same honest output.

## Status

✅ **v0.1.0 shipped 2026-06-04** — [GitHub Release](https://github.com/maheshbabugorantla/whatcanirun/releases/tag/v0.1.0). Ships as a clone-install repo for power users (host-`uv` or Docker; no PyPI yet). All 13 v1 milestones landed (M00 bootstrap through M12 release, with M10 partially shipped — Tier 1b public-benchmark cells removed from v1 after the source landscape proved infeasible; see [`spec/INDEX.md`](spec/INDEX.md) footnote). PyPI publication and MCP-registry submissions are deferred to v2 once the tool surface stabilizes through real usage.

See [`spec/INDEX.md`](spec/INDEX.md) for the milestone roadmap and [`spec/M12-release.md`](spec/M12-release.md) § "Deferred to v2" for the PyPI deferral rationale.

## What's wired today

M09 landed the full MCP surface — the stdio server is callable end-to-end against the live CP / HF / AA upstream stack.

**Tools** — the four numerical tools (`fit_check`, `find_cheapest_deployment`, `compare_deployment_modes`, `budget_to_plan`) each carry a `trust_envelope` on every response; `list_catalog` and `resolve_model` return catalog facts or persistence status and are deliberately envelope-exempt because they don't synthesize a number.

- `list_catalog` — enumerate the GPU SKUs, providers, and tracked models the server knows about, suitable as a dropdown source for an LLM client. Catalog facts only; no envelope.
- `fit_check(model, gpu, quant)` — pure-math VRAM verdict; returns a `FitResult` with weight/KV-cache/overhead breakdown, headroom, blocking reasons, and a `sufficiency_caveat` (fits=True is necessary but not sufficient).
- `find_cheapest_deployment(model, workload_profile, ...)` — ranked list of cost cells across providers/GPUs/quants; each row carries its own envelope per the per-row-envelope contract.
- `compare_deployment_modes(model, workload_profile)` — side-by-side comparison of cloud-GPU-rental vs hosted-API-token economics for the same model.
- `budget_to_plan(budget_usd, model, workload_profile)` — converts a dollar budget into a ranked plan with `hours_available`, `est_total_prompts`, and `est_wallclock_minutes` per row.
- `resolve_model(model_slug, hf_repo_id)` — for models the server doesn't already know, persist a `(slug, hf_repo_id)` mapping and sync the architecture from Hugging Face; backs the unknown-model elicitation flow. Status-only; no envelope.

**Resources:**

- `cost-cells://current` — full materialized cost-cell table as Parquet (DuckDB-backed).
- `cost-cells://provenance` — JSON document with upstream source attributions, license terms, and audit links.

**Prompt:**

- `/benchmark-on-budget` — guided template for the most common "I have $X" question.

## Install

v1 ships as a clone-install repo for power users. Two paths,
identical MCP surface — pick the one that matches what's already
on your host.

### Host-uv (recommended)

Requires Python 3.12 + [`uv`](https://docs.astral.sh/uv/) on the host.

```bash
git clone https://github.com/maheshbabugorantla/whatcanirun
cd whatcanirun
./scripts/install_host_uv.sh
```

The install script runs `uv sync`, warms the upstream caches via
`whatcanirun-mcp prefetch`, runs the release-gate stdio test, and
prints the MCP client config block users paste into their client
(Claude Desktop / Claude Code / Cursor / Cline) — with the
absolute repo path already substituted.

### Docker

Requires `docker`. The image entry point is the stdio MCP server.

```bash
git clone https://github.com/maheshbabugorantla/whatcanirun
cd whatcanirun
docker build -t whatcanirun:latest .
# Optional one-shot cache warmup on the named volume:
docker run --rm -i \
  -v whatcanirun-cache:/var/cache/whatcanirun \
  whatcanirun:latest prefetch
```

Point the MCP client at `scripts/run_mcp_docker.sh` — the
wrapper handles `-i`, `--rm`, the named cache volume, and env-var
passthrough so the client config block stays a single-line
`command` field.

### Per-client configuration

Full per-client examples (Claude Desktop, Claude Code, Cursor,
Cline), env-var passthrough caveats, and troubleshooting live in
[`docs/MCP.md`](docs/MCP.md).

### API keys (all optional)

The server runs with **zero API keys**. Set any of these in your
MCP client's `env:` block (or your shell) to lift rate limits or
enable optional enrichment:

| Variable | What it unlocks | Without it |
|---|---|---|
| `COMPUTEPRICES_API_KEY` | Lifts ComputePrices anonymous rate limits. Free key on request via [computeprices.com](https://computeprices.com/). | Anonymous reads with lower quota; [ADR-013](docs/ADRs/ADR-013-snapshot-fallback.md) snapshot fallback covers any rate-limit hits. |
| `HF_TOKEN` | Auth for private/gated Hugging Face configs. Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). | Public-only reads — sufficient for every model in `seeds/tracked_models.yaml`. |
| `AA_API_KEY` | Enables Artificial Analysis enrichment ([ADR-003](docs/ADRs/ADR-003-aa-optional-enrichment.md)) — AA is the `provider_anchor` (Tier 2) throughput source. Free tier at [artificialanalysis.ai/api](https://artificialanalysis.ai/api). | Throughput falls back to the bandwidth heuristic (Tier 3, batch=1 only) or refuses with `requires_measurement` (Tier 4). |

Empty strings count as "unset" — `AA_API_KEY=""` doesn't break the
anonymous path. Per-client `env:` block syntax (Claude Desktop,
Cursor, Cline) is in [`docs/MCP.md`](docs/MCP.md) § Environment
variables.

## Validating a running install

[`docs/SCENARIOS.md`](docs/SCENARIOS.md) carries eight prose-shape
end-to-end scenarios you can walk through in chat to validate that a
running install behaves as expected — not just that it responds, but
that it relays the trust contract honestly (caveats verbatim,
weakest-link confidence, no bluffing on `requires_measurement`). The
scenarios cover the headline budget / fit / cost / compare questions
plus the unknown-model + workload elicitation flows.

The mechanical server-side gate is `pytest -m release`, which the
host install script runs automatically; the scenarios layer on top to
catch *client-side* relay regressions a server-only gate can't see.

## Data sources

- **[ComputePrices](https://computeprices.com/)** — GPU rental + LLM API pricing across 70+ providers
- **[Hugging Face](https://huggingface.co/)** — model architecture from `config.json`
- **[Artificial Analysis](https://artificialanalysis.ai/)** *(optional)* — quality scores + per-model throughput aggregates

Attribution and license respect for upstream sources lives in [`docs/TRUST.md`](docs/TRUST.md) and the `cost-cells://provenance` MCP resource.

## License

MIT — declared in `pyproject.toml`. The benchmark *dataset*
(published as Parquet on Hugging Face Datasets, post-v2 M17)
ships under CC-BY-4.0 per [ADR-006](docs/ADRs/ADR-006-benchmark-cells-parquet.md).
