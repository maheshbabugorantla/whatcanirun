"""PROTOTYPE — throwaway sketch for M10 PR-alpha verification tooling.

Question this sketch answers: does a "predicted bandwidth-heuristic TPS
vs measured TPS" comparison surface anything interesting against the
existing 8 cells in `seeds/benchmark_cells.parquet`? If yes, the V1
sanity-check tool gets a `check_decode_tps_vs_bandwidth_heuristic`
function. If no (all predictions land within noise), drop the check.

Hardcodes the model/GPU/quant lookups — the V1 production version
will use the real loaders. This is a one-shot sketch.

Run: `uv run python scripts/m10/_sketch.py`

DELETE THIS FILE when PR-alpha lands V1 properly.
"""

from __future__ import annotations

import pyarrow.parquet as pq

# ---------------------------------------------------------------- inputs
# Hardcoded for the 8 cells in seeds/benchmark_cells.parquet as of
# 2026-05-30. Verified against:
# - seeds/gpus_supplement.yaml + CP gpu_catalog cache for bandwidth
# - seeds/quantizations.yaml for bits_per_weight
# - HF config.json values for total_params_b
# Numbers are approximate but in the right ballpark for sketch purposes.

_GPU_BANDWIDTH_GBPS = {
    "h100": 3350.0,  # H100 SXM, HBM3
    "l40s": 864.0,  # L40S, GDDR6
}

_QUANT_BITS_PER_WEIGHT = {
    "fp8": 8,
    "bf16": 16,
    "fp16": 16,
    "int4": 4,
}

_MODEL_TOTAL_PARAMS_B = {
    "llama-3-1-8b": 8.0,
    "llama-3-3-70b": 70.0,
    "mistral-7b": 7.0,
    "mixtral-8x7b": 46.7,  # total params; active is ~12.9B but heuristic uses total for memory bw
    "qwen-2-5-72b": 72.0,
    "deepseek-v3": 685.0,
    "phi-4": 14.0,
}

_KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75  # from tps_estimator.py


def _predict_tps(gpu_slug: str, model_slug: str, quant_slug: str) -> float | None:
    """Apply the same bandwidth-heuristic formula tps_estimator.py
    uses at Tier 3. Returns None if any input is missing."""
    bw = _GPU_BANDWIDTH_GBPS.get(gpu_slug)
    params_b = _MODEL_TOTAL_PARAMS_B.get(model_slug)
    bits = _QUANT_BITS_PER_WEIGHT.get(quant_slug)
    if bw is None or params_b is None or bits is None:
        return None
    weights_bytes_per_token = params_b * 1e9 * bits / 8.0
    peak_tps = bw * 1e9 / weights_bytes_per_token
    return peak_tps * _KERNEL_EFFICIENCY_SINGLE_STREAM


def _classify(ratio: float | None) -> str:
    """Map measured/predicted ratio → human label."""
    if ratio is None:
        return "SKIP (data gap)"
    if 0.5 <= ratio <= 1.5:
        return "OK (within ±50%)"
    if ratio < 0.5:
        return f"FLAG (under-performs by {(1 - ratio) * 100:.0f}%)"
    return f"FLAG (over-performs by {(ratio - 1) * 100:.0f}%)"


def main() -> None:
    t = pq.read_table("seeds/benchmark_cells.parquet")
    rows = [{col: t.column(col)[i].as_py() for col in t.column_names} for i in range(t.num_rows)]

    print(
        f"{'gpu':<6} {'model':<18} {'quant':<6} {'batch':<6} "
        f"{'actual':>8} {'predicted':>10} {'ratio':>7}  verdict"
    )
    print("-" * 95)

    flagged = 0
    for row in rows:
        if row["batch_size"] != 1:
            # Per the scope doc, the bandwidth heuristic check only
            # applies at batch=1. Batched cells route to the separate
            # `check_batch_scaling_not_linear` check instead.
            continue
        predicted = _predict_tps(row["gpu_slug"], row["model_slug"], row["quant_slug"])
        actual = row["decode_tps"]
        ratio = actual / predicted if predicted else None
        verdict = _classify(ratio)
        if "FLAG" in verdict:
            flagged += 1
        predicted_str = f"{predicted:.1f}" if predicted else "n/a"
        ratio_str = f"{ratio:.2f}" if ratio else "n/a"
        print(
            f"{row['gpu_slug']:<6} {row['model_slug']:<18} {row['quant_slug']:<6} "
            f"{row['batch_size']:<6} {actual:>8.1f} {predicted_str:>10} "
            f"{ratio_str:>7}  {verdict}"
        )

    print()
    print(f"summary: {flagged}/{len(rows)} cells flagged for review")
    print()
    print("interpretation:")
    print("- FLAGs at batch=1 mean the cell warrants a methodology re-check,")
    print("  not auto-rejection. They're informational signals during curation.")
    print("- If ≥1 flag surfaces something real, V1 gets")
    print("  `check_decode_tps_vs_bandwidth_heuristic` as a warning-tier check.")


if __name__ == "__main__":
    main()
