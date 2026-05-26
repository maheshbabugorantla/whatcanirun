"""One-shot capture of ComputePrices /api/v1/gpus for the test fixture.

Run this when ComputePrices ships a new GPU you care about, or when you
want to refresh the slug ground-truth used by M01's join test.

Anonymous endpoint (60 req/hr per IP) is sufficient — this script makes
exactly one request. The response is persisted verbatim per ADR-015.

Usage:
    uv run python scripts/capture_cp_gpus_fixture.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import httpx

CP_URL = "https://www.computeprices.com/api/v1/gpus"
OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> int:
    today = dt.datetime.now(tz=dt.UTC).date().isoformat()
    out_path = OUT_DIR / f"cp_gpus_{today}.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GET {CP_URL}")
    try:
        response = httpx.get(CP_URL, timeout=30, follow_redirects=True)
    except httpx.HTTPError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(f"FAILED: HTTP {response.status_code}: {response.text[:500]}", file=sys.stderr)
        return 1

    payload = response.json()
    n_rows = len(payload.get("data", []))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {n_rows} GPU rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
