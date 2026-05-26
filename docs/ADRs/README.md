# Architectural Decision Records

15 ADRs are locked. Their summary table lives in `spec/SHARED.md`.

M11 populates this directory with one file per ADR. Until then, read the summary in SHARED.md.

| ADR | Decision |
|---|---|
| ADR-001 | ComputePrices canonical for prices + GPU base catalog |
| ADR-002 | Hugging Face canonical for model architecture |
| ADR-003 | Artificial Analysis optional enrichment |
| ADR-004 | Trust envelope on every numerical response |
| ADR-005 | 12-row GPU supplement YAML |
| ADR-006 | Benchmark cells as Parquet on HF Datasets |
| ADR-007 | v1 stdio; v2 bearer-token remote; no Claude.ai web |
| ADR-008 | v1 stack: FastMCP + Pydantic + httpx + DuckDB-on-files |
| ADR-009 | v2 stack: Django + Postgres + Redis + Celery |
| ADR-010 | TPS heuristic single-stream only |
| ADR-011 | Observability in v2 only |
| ADR-012 | Auth via email-OTP → bearer; no OAuth |
| ADR-013 | Fallback to local snapshot when upstream unreachable |
| ADR-014 | DuckDB only for resource generation, not tool calls |
| ADR-015 | Raw + projection storage for all upstream APIs |
