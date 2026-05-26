# M11 — Tests + Golden-Path + Docs

**Status:** ⬜ Not started
**Effort:** 6h (10h realistic)
**Dependencies:** M00–M10
**Unblocks:** M12

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

Test suite that runs in <30 seconds with no live network. One golden-path integration test gates release. Three doc files (`README.md`, `docs/MCP.md`, `docs/TRUST.md`) explain the product, the setup, and the trust contract to readers who never read the spec.

---

## Scope

### Tests

**Unit tests** (per-milestone, already written in M02–M08):
- Pydantic schemas — extra fields, missing required fields, type coercion
- Pure functions — fit_check, tps_estimator
- HTTP clients — `respx` mocked; happy path, retry, fallback, malformed response
- Schema-evolution tests — `@pytest.mark.schema_evolution`, exercised in CI

**Integration tests** (`tests/integration/`):
- MCP tool calls end-to-end through the FastMCP test client
- Use fixtured upstream responses; no live network

**Golden path** (gates release):
```python
# tests/integration/test_golden_path.py
@pytest.mark.integration
async def test_budget_to_plan_u3_transcript():
    """The headline use case. CI fails if this regresses."""
    server = whatcanirun.server.build_app(cache_dir=fixture_cache_dir)
    client = FastMCPTestClient(server)

    response = await client.call_tool(
        "budget_to_plan",
        budget_usd=20.0,
        model_slug="qwen-3-coder-30b",
    )

    plan_rows = response.result
    assert len(plan_rows) >= 3

    # Sorted ASC by cost_per_m_output_usd
    costs = [r.cost_per_m_output_usd for r in plan_rows]
    assert costs == sorted(costs)

    # Every row has a populated trust envelope with all 6 domains
    for row in plan_rows:
        env = row.trust_envelope
        assert env.confidence == min(env.confidence_breakdown.values())
        assert set(env.confidence_breakdown.keys()) == {
            "pricing", "fit_check", "throughput",
            "model_architecture", "gpu_specs", "freshness",
        }
        assert row.cost_cell.availability_modeled is False
        assert "Price source does not guarantee" in row.cost_cell.availability_caveat
        if row.cost_cell.fit_result is not None:
            assert row.cost_cell.fit_result.sufficiency_caveat  # non-empty
```

### Docs

**`README.md`** (already drafted at repo root; M11 polishes it):
- One-paragraph pitch ("I have $X, what LLM can I run?")
- Quickstart `uvx whatcanirun-mcp` block
- The U3 transcript as the first code example
- Link to MCP.md for setup, TRUST.md for the trust contract
- Attribution to ComputePrices and AA

**`docs/MCP.md`** — installation per client:
- Claude Desktop config block (the canonical one)
- Claude Code config block
- Cursor config block
- Cline config block
- Troubleshooting: PATH issues with `uvx`, stdio handshake timeouts, COMPUTEPRICES_API_KEY env var passthrough

**`docs/TRUST.md`** — the trust contract in detail:
- Why the trust envelope exists
- Every confidence domain explained with examples
- The 5-tier TPS provenance taxonomy
- The "we model pricing, not rentability" caveat
- The staleness policy with curve diagrams (optional, Mermaid)
- The list of things we explicitly do NOT model
- Attribution + licenses

**`docs/ADRs/`** — 15 ADRs, one file each:
- `ADR-001-computeprices-canonical.md` through `ADR-015-raw-projection-pattern.md`
- Each ~100–300 words: decision, context, consequences, alternatives considered, references

**`docs/PRD.md`** — the full v2.1 ROADMAP, lightly edited for public consumption.

---

## Vertical slices

1. **Slice A: Test fixture cache directory** — `tests/fixtures/cp_gpus_2026-05-25.json`, `cp_gpu_prices_2026-05-25.json`, `aa_models_2026-05-25.json`, `hf_configs/qwen-3-coder-30b.json`, etc. Captured from live (or synthesized) responses.
2. **Slice B: FastMCP test client integration** — TDD: a single tool call works against the test client.
3. **Slice C: Golden-path test** — TDD per assertion in the test block above.
4. **Slice D: README polish** — confirm the README links work, the U3 example renders correctly in GitHub, attribution is present.
5. **Slice E: docs/MCP.md** — write per-client config blocks. Verify each manually with the actual client.
6. **Slice F: docs/TRUST.md** — write the deep-dive. Aim for ~1500 words.
7. **Slice G: ADRs** — port each from `SHARED.md` § ADRs to its own file.
8. **Slice H: docs/PRD.md** — copy v2.1 ROADMAP, strip implementation gossip, polish.

---

## Acceptance criteria

- [ ] Full test suite (`uv run pytest -ra`) runs in <30 seconds.
- [ ] Zero live network calls in test suite (verified via `respx.MockTransport` assertion).
- [ ] Golden-path test passes; CI gates release on it.
- [ ] Schema-evolution tests pass (`uv run pytest -m schema_evolution`).
- [ ] `README.md` renders cleanly on GitHub (preview before merging).
- [ ] `docs/MCP.md` config blocks work in Claude Desktop AND Claude Code (manual verification, recorded in commit message).
- [ ] `docs/TRUST.md` covers all 6 confidence domains, the 5 TPS tiers, the staleness policy.
- [ ] All 15 ADRs have their own file in `docs/ADRs/`.
- [ ] `docs/PRD.md` is the v2.1 roadmap, public-facing.

---

## Common pitfalls

- **Test fixtures going stale.** ComputePrices, AA, HF responses change over time. Pin fixtures to a specific snapshot date in the filename (`cp_gpus_2026-05-25.json`) so future contributors know when to regenerate.
- **Live network sneaking into tests.** A new test imports the real client class and forgets to patch `httpx.AsyncClient`. Add a session-scoped fixture that monkeypatches `httpx.AsyncClient.send` to raise; this catches accidental real-network usage.
- **README too long.** GitHub truncates README previews at ~1000 lines. Keep it focused. Deep content belongs in `docs/`.
- **Per-client config blocks subtly different.** Claude Desktop, Code, and Cursor have different config file locations and slightly different JSON shapes. Test each.

---

## When done

Commit:
> `M11: golden-path integration test + README, MCP.md, TRUST.md, 15 ADRs`

Mark M11 ✓ in `INDEX.md`. Continue with M12.
