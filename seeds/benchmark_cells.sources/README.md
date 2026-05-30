# benchmark_cells.sources/

Archived HTML of every public benchmark source cited in
`seeds/benchmark_cells.parquet`. Defends the trust contract against link
rot — if the cited URL 404s later, the archive is the audit-trail of
record. Per locked M10 decision #3 (handoff doc § 5).

## Convention

One subdirectory per source, named by a short slug that matches the
`source_url`'s host + path stem. Examples:

- `spheron-llama-3-3-70b-fp8-h100-benchmark/`
- `vllm-blog-2024-llama-3-1-8b-h100-bf16/`
- `morphllm-tokens-per-second-survey/`

Each subdirectory contains:

- `article.html` — trimmed article body (strip nav, ads, sidebars; preserve
  tables, code blocks, byline, publish date).
- `meta.json` — `{ "url": "...", "fetched_at": "ISO-8601", "title": "...",
  "publisher": "..." }`. Lightweight metadata for human cross-reference.

## Why trimmed not full

Full blog HTML is often 200KB+ of analytics scripts and ad markup. The
benchmark numbers live in a small slice. Trimming keeps repo size sane
(~10–50KB per source) without losing the audit content. The accessible
URL in the cell remains the canonical source; the archive is fallback.

## V1 enforcement

`scripts/m10/sanity_check_cells.py` asserts that every committed cell has
a corresponding `benchmark_cells.sources/<slug>/article.html` reachable via
the path-stem mapping. New cells without an archived source fail with
exit code 2 (blocking).

## Bootstrap status

This directory is empty at PR-α merge time — the existing 8 seed cells
predate the archival convention and have a known gap. Backfill happens in
Slice D Phase 1: each per-source PR archives ONLY its own cells; the
8 grandfathered cells get archived as part of PR-δ (M10 close-out) if any
of their source URLs still 200, or removed if they don't.
