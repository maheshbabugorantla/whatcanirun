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
| `cp_gpus_2026-05-26.json` | `GET https://www.computeprices.com/api/v1/gpus` (anonymous) | `scripts/capture_cp_gpus_fixture.py` | 66 GPU rows. Used by M01 `tests/catalog/test_seed_join.py` to assert every `seeds/gpus_supplement.yaml` slug joins a real CP row. Data © ComputePrices, public catalog; refer to their site for the canonical attribution language we propagate in trust envelopes (ADR-001). |
