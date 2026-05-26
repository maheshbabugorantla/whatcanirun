# whatcanirun

An MCP server that answers the one question every LLM inference user actually asks:

> **"I have $X to spend on model Y — what can I actually run?"**

Self-hosted, free public APIs only, ships as a stdio MCP server you install with one command. No accounts, no hosting, no telemetry.

## What it does

Given a dollar budget and an open-weight model, returns a ranked, source-backed plan: which GPU on which provider, whether the model fits in VRAM, how many tokens per second you'll get, how many prompts that buys you, and exactly which assumptions and caveats underlie every number.

Every response carries a structured `trust_envelope` — sources, per-domain confidence, assumptions, caveats, freshness, and audit links. Hobbyist or power user, same honest output.

## Status

⬜ Pre-alpha. v1 in active development.

See [`spec/INDEX.md`](spec/INDEX.md) for the milestone roadmap.

## Quickstart (post-v1)

```bash
uvx whatcanirun-mcp
```

Then add to Claude Desktop / Claude Code config — see [`docs/MCP.md`](docs/MCP.md).

## Data sources

- **[ComputePrices](https://computeprices.com/)** — GPU rental + LLM API pricing across 70+ providers
- **[Hugging Face](https://huggingface.co/)** — model architecture from `config.json`
- **[Artificial Analysis](https://artificialanalysis.ai/)** *(optional)* — quality scores + per-model throughput aggregates

Attribution and license respect for upstream sources lives in [`docs/TRUST.md`](docs/TRUST.md) and the `cost-cells://provenance` MCP resource.

## License

MIT (TBD — see open decisions in spec/SHARED.md)
