# M10 verification tooling

Two scripts that gate every `seeds/benchmark_cells.parquet` change with
machine-readable methodology + sanity checks. Build per spec/M10-benchmark-seeds.md
slices C, D, and the implicit "verification tooling" expansion captured in
the M10 handoff doc.

This tooling is throwaway-after-M10 by design — it exists to make Slice D's
per-source curation auditable and consistent. After M10 closes out, these
scripts may stay or be retired; they're not part of the package's public API.

## Tools

### `sanity_check_cells.py` (V1)

Validates a YAML/JSON file of candidate `BenchmarkCell` rows against:

- Catalog join keys: `gpu_slug`, `model_slug`, `quant_slug` resolve in
  CP gpu_catalog / merged tracked_models / `quantizations.yaml`.
- GPU SXM/PCIe disambiguation in `notes` (ADR-010 pitfall).
- Methodology completeness: `notes` ≥30 chars + mentions `engine_version` + `batch_size`.
- `engine_version` is semver-shaped (`\d+\.\d+(\.\d+|\.x)?`).
- `measured_at` within last 18 months (stale-numbers pitfall).
- `decode_tps` vs bandwidth-heuristic prediction (MoE-aware: uses
  `active_params_b` for sparse models when present).
- `batch_size > 1` cells do NOT scale linearly with batch (ADR-010 verified
  ~6× wrong at batch=128).
- `source_url` is a well-formed http(s) URL.
- Op-point `(gpu, model, quant, tp, batch, ctx)` not already present in the
  canonical parquet (prevents silent overrides).

Exit codes: `0` clean, `1` warnings, `2` blocking errors.

On exit-0, emits a `.sanity-passed` sidecar next to the input file that V2
checks for.

### `merge_candidate_to_parquet.py` (V2)

Refuses to run unless the sanity tool's `.sanity-passed` sidecar exists for
the input. Diffs existing parquet rows vs new on op-point key, reports rows
added, preserves all existing rows. Writes atomically via tmp+rename.

## Source HTML archival

Per locked M10 decision #3, every benchmark cell's source HTML is committed
to `seeds/benchmark_cells.sources/` at curation time, with the path stored
in the cell's notes. V1 asserts presence of the archive file for every
committed cell. This defends against link rot — if the source URL 404s
later, the archive is the audit-trail of record.

## Prototype findings (2026-05-30)

The pre-V1 `_sketch.py` prototype validated the bandwidth-heuristic check
against the existing 8 cells. Findings folded into V1's design:

1. The check IS valuable — 3 of 8 cells flagged for follow-up.
2. MoE handling is required — DeepSeek-V3's heuristic prediction was
   404% off because it naively used `total_params_b` (685B) instead of
   `active_params_b` (~37B). V1's check consults `active_params_b` when
   present and falls back to total only for dense models.
3. Two existing cells (phi-4 L40S, mistral-7b H100 FP8) deserve methodology
   re-review during Slice D Phase 1 — flagged in `seeds/benchmark_cells.README.md`.
