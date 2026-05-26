"""One-shot capture of ComputePrices public endpoints for test fixtures.

Run this when ComputePrices ships a new GPU/model/price you care about,
or when you want to refresh the ground-truth used by the cached-fixture
tests in tests/catalog/ (M01) and tests/pricing/ (M02 onward).

Anonymous tier (60 req/hr per IP) is sufficient — this script makes one
request per endpoint, four total. Each response is persisted verbatim
per ADR-015 (raw + projection: store the full payload, project only
what we currently consume).

Usage:
    uv run python scripts/capture_cp_gpus_fixture.py                 # all 4
    uv run python scripts/capture_cp_gpus_fixture.py gpus            # one
    uv run python scripts/capture_cp_gpus_fixture.py gpus llm-models # subset

Endpoints captured (key → URL → output file stem):
    gpus         -> /api/v1/gpus         -> cp_gpus_<date>.json
    llm-models   -> /api/v1/llm-models   -> cp_llm_models_<date>.json
    gpu-prices   -> /api/v1/gpu-prices   -> cp_gpu_prices_<date>.json
    llm-prices   -> /api/v1/llm-prices   -> cp_llm_prices_<date>.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import httpx

CP_BASE = "https://www.computeprices.com/api/v1"

# endpoint key -> (URL path, output filename stem)
ENDPOINTS: dict[str, tuple[str, str]] = {
    "gpus": ("gpus", "cp_gpus"),
    "llm-models": ("llm-models", "cp_llm_models"),
    "gpu-prices": ("gpu-prices", "cp_gpu_prices"),
    "llm-prices": ("llm-prices", "cp_llm_prices"),
}

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _capture(client: httpx.Client, key: str, date_iso: str) -> int:
    path, stem = ENDPOINTS[key]
    url = f"{CP_BASE}/{path}"
    out_path = OUT_DIR / f"{stem}_{date_iso}.json"

    print(f"GET {url}")
    try:
        response = client.get(url, timeout=30, follow_redirects=True)
    except httpx.HTTPError as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"  FAILED: HTTP {response.status_code}: {response.text[:500]}", file=sys.stderr)
        return 1

    payload = response.json()
    rows = payload.get("data", [])
    n_rows = len(rows) if isinstance(rows, list) else "n/a"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {n_rows} rows to {out_path}")
    return 0


def main(argv: list[str]) -> int:
    keys = argv if argv else list(ENDPOINTS)
    unknown = [k for k in keys if k not in ENDPOINTS]
    if unknown:
        print(f"unknown endpoint(s): {unknown}; valid: {list(ENDPOINTS)}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_iso = dt.datetime.now(tz=dt.UTC).date().isoformat()

    failures = 0
    with httpx.Client() as client:
        for key in keys:
            failures += _capture(client, key, date_iso)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
