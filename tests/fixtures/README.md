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
