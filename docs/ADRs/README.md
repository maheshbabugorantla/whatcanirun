# Architectural Decision Records

15 ADRs are locked. Each is exactly the decision that, if reversed,
would break a load-bearing assumption elsewhere in the codebase.
Open them individually for the full decision, context, consequences,
alternatives considered, and references.

The canonical decision table also lives in
[`../../spec/SHARED.md`](../../spec/SHARED.md) § ADRs.

## Data sources & enrichment

- [ADR-001 — ComputePrices canonical for prices + GPU base catalog](ADR-001-computeprices-canonical.md)
- [ADR-002 — Hugging Face canonical for model architecture](ADR-002-huggingface-canonical-architecture.md)
- [ADR-003 — Artificial Analysis optional enrichment](ADR-003-aa-optional-enrichment.md)
- [ADR-005 — GPU supplement YAML for fields CP doesn't carry](ADR-005-gpu-supplement-yaml.md)
- [ADR-006 — Benchmark cells as Parquet on Hugging Face Datasets](ADR-006-benchmark-cells-parquet.md)

## Trust contract

- [ADR-004 — TrustEnvelope required on every numerical response](ADR-004-trust-envelope-required.md)
- [ADR-010 — TPS heuristic single-stream only (batch=1)](ADR-010-tps-single-stream.md)
- [ADR-013 — Snapshot fallback when ComputePrices is unreachable](ADR-013-snapshot-fallback.md)
- [ADR-015 — Raw + projection storage pattern for upstream APIs](ADR-015-raw-projection-pattern.md)

## Stack & transport

- [ADR-007 — v1 transport: stdio only](ADR-007-stdio-transport.md)
- [ADR-008 — v1 stack: FastMCP + Pydantic + httpx + DuckDB-on-files](ADR-008-v1-stack.md)
- [ADR-009 — v2 stack: Django + DRF + Postgres + Redis + Celery](ADR-009-v2-stack.md)
- [ADR-014 — DuckDB only for resource generation; Python joins for tools](ADR-014-duckdb-resource-only.md)

## Operations (v2)

- [ADR-011 — Observability in v2 only; v1 logs to stderr](ADR-011-observability-v2-only.md)
- [ADR-012 — v2 auth: email-OTP -> bearer API key via Resend](ADR-012-auth-email-otp.md)
