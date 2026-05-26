# Test Fixtures

Captured upstream API responses, used by all tests in place of live network calls.

## Filename convention

`<source>_<endpoint>_<YYYY-MM-DD>.json` — date in filename indicates the snapshot. When upstream schemas drift meaningfully (per ADR-015's schema-evolution tests), regenerate fixtures with a current date and update tests that depend on specific values.

## Adding new fixtures

1. With real API keys in `.env`, capture a live response (curl or a one-off script).
2. Sanitize: strip any account-specific fields, replace API keys with `cp_live_REDACTED`.
3. Save with the dated filename.
4. Reference from tests by relative path.

## NEVER

- Commit a fixture containing a real API key.
- Commit a fixture from a production account with PII.
- Use live HTTP in tests (CI runs without keys; tests will silently 401).

## Current fixtures

| File | Source | Captured by | Notes |
|---|---|---|---|
| `cp_gpus_2026-05-26.json` | `GET https://www.computeprices.com/api/v1/gpus` (anonymous) | `scripts/capture_cp_gpus_fixture.py gpus` | 66 GPU rows. Used by M01 `tests/catalog/test_seed_join.py` to assert every `seeds/gpus_supplement.yaml` slug joins a real CP row. |
| `cp_gpu_prices_2026-05-26.json` | `GET https://www.computeprices.com/api/v1/gpu-prices` (anonymous) | `scripts/capture_cp_gpus_fixture.py gpu-prices` | 1000 (provider, GPU, pricing_type) price rows. Used by M02 `tests/pricing/test_projections.py` to project every row through `GpuPriceRow` and confirm the pricing_type Literal stays exhaustive. |
| `cp_llm_models_2026-05-26.json` | `GET https://www.computeprices.com/api/v1/llm-models` (anonymous) | `scripts/capture_cp_gpus_fixture.py llm-models` | 214 LLM rows (slug + name + creator + family + context_window + modalities + knowledge_cutoff). Used by M02 `tests/pricing/test_projections.py::TestLlmCatalogRow`. |
| `cp_llm_prices_2026-05-26.json` | `GET https://www.computeprices.com/api/v1/llm-prices` (anonymous) | `scripts/capture_cp_gpus_fixture.py llm-prices` | 498 (provider, model, pricing_type) per-1M-token price rows including the new `price_per_1m_cached_input_usd` field. Used by M02 `tests/pricing/test_projections.py::TestLlmPriceRow`. |

All CP fixtures: data © ComputePrices, public catalog; refer to <https://www.computeprices.com> for the canonical attribution language we propagate in TrustEnvelope.caveats (ADR-001).
