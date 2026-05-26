# M01 — Catalog Supplements (GPU + Quantization YAMLs)

**Status:** ⬜ Not started
**Effort:** 1h
**Dependencies:** M00
**Unblocks:** M06 (fit_check needs quant.kv_cache_bits, GPU fp8_tflops), M07 (tps_estimator needs supplement flags)

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

Two YAML files: 12 GPU supplement rows and 10 quantization rows. They cover what ComputePrices `/api/v1/gpus` does NOT expose — specifically `fp8_tflops_dense`, KV cache bit-widths per quantization scheme, GPU form factor, and the MLA-vs-GQA attention strategy flag.

These are facts from NVIDIA/AMD datasheets and the Inference Engineering book §5.1.1. They change ≤2×/year. Manual YAML is appropriate; no upstream API exists.

---

## Scope

### `seeds/gpus_supplement.yaml` (12 rows)

Joins ComputePrices `/api/v1/gpus` by `slug`. Covers only data-center GPUs we care about.

```yaml
- slug: h100              # joins ComputePrices "H100 SXM"
  fp8_tflops_dense: 1979  # NVIDIA H100 datasheet, page 6
  fp4_tflops_dense: null  # Hopper does not support FP4
  form_factor: SXM
  supports_fp8: true
  supports_fp4: false
  attention_kernels_supported: [flash_attention_2, paged_attention]
  notes: "Hopper architecture, HBM3, 80GB. Released March 2022."
  datasheet_url: "https://www.nvidia.com/en-us/data-center/h100/"

- slug: h200
  fp8_tflops_dense: 1979
  fp4_tflops_dense: null
  form_factor: SXM
  supports_fp8: true
  supports_fp4: false
  attention_kernels_supported: [flash_attention_2, paged_attention]
  notes: "Hopper architecture, HBM3e, 141GB. Same compute as H100, more memory."
  datasheet_url: "https://www.nvidia.com/en-us/data-center/h200/"

# ... 10 more rows for: h100pcie, h100nvl, b100, b200, gb200, gb300, l40s, l40, a100sxm, mi300x
```

### `seeds/quantizations.yaml` (10 rows)

From Inference Engineering book §5.1.1 "Number Formats" table on page 121.

```yaml
- slug: fp16
  bits_per_weight: 16
  kv_cache_bits_default: 16
  introduced_architecture: Pascal
  notes: "IEEE 754 half-precision; native on all data-center GPUs"

- slug: bf16
  bits_per_weight: 16
  kv_cache_bits_default: 16
  introduced_architecture: Ampere
  notes: "Brain float; wider dynamic range than fp16, same memory cost"

- slug: fp8
  bits_per_weight: 8
  kv_cache_bits_default: 8
  introduced_architecture: Hopper
  notes: "e4m3 / e5m2 dual-format. Native FP8 tensor cores from Hopper onward."

# ... 7 more rows: int8, int4, fp4, fp6 (stable);
#                  nvfp4, mxfp4, mxfp8 (experimental: true).
```

**Ship all 10 rows; mark the iffy ones with `experimental: true`.**
The earlier guidance ("skip experimental formats until measured cells
exist (M10)") contradicted the "exactly 10 rows" acceptance criterion.
Resolution: the `Quantization` schema carries `experimental: bool =
False`, and M07 (tps_estimator) plus M10 (benchmark cells) filter on
that flag so v1 plan output never anchors throughput on an unsettled
format. The seed YAML is single source of truth; downstream consumers
opt in to experimental rows explicitly.

### Pydantic schema validation

`src/whatcanirun/catalog/seed_schemas.py`:

```python
from pydantic import BaseModel, ConfigDict
from typing import Literal

class GpuSupplement(BaseModel):
    model_config = ConfigDict(extra="forbid")  # supplement YAML IS our own; strict here
    slug: str
    fp8_tflops_dense: float | None
    fp4_tflops_dense: float | None
    form_factor: Literal["SXM", "PCIe", "NVL", "OAM"]
    supports_fp8: bool
    supports_fp4: bool
    attention_kernels_supported: list[str]
    notes: str
    datasheet_url: str

class Quantization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    bits_per_weight: int
    kv_cache_bits_default: int
    introduced_architecture: str
    notes: str
    experimental: bool = False  # see "Ship all 10 rows" note above
```

### Loader

`src/whatcanirun/catalog/loaders.py`:

```python
def load_gpu_supplements(path: Path) -> list[GpuSupplement]: ...
def load_quantizations(path: Path) -> list[Quantization]: ...
```

Raises with line numbers on malformed YAML.

---

## Out of scope

- Joining with ComputePrices `/api/v1/gpus` data — that's M02's first integration.
- Supplements for consumer GPUs (RTX 4090, 5090) — not in v1 target set.
- AMD-specific attention kernel flags (Flash Attention 2 isn't yet on MI300X via vLLM) — note as caveat in the row, don't try to model.

---

## Vertical slices

1. **Slice A: Pydantic schemas** — write `GpuSupplement` and `Quantization` with one failing test (`test_seed_schemas.py::test_gpu_supplement_rejects_extra_field`). Implement. Green.
2. **Slice B: YAML loader** — write a failing test (`test_loader_loads_one_row`). Implement `load_gpu_supplements`. Green.
3. **Slice C: Full GPU YAML** — fill in all 12 rows. Add a fixture-based test verifying each row's slug joins to a real ComputePrices slug (use a captured fixture from May 2026; live network NOT allowed in tests).
4. **Slice D: Full Quantization YAML** — fill in all 10 rows. Test loads cleanly.
5. **Slice E: Round-trip integrity** — test that loading the YAMLs into Pydantic and dumping back produces the same dict.

---

## Acceptance criteria

- [ ] `seeds/gpus_supplement.yaml` has exactly 12 rows.
- [ ] `seeds/quantizations.yaml` has exactly 10 rows.
- [ ] Every GPU row has a publicly accessible `datasheet_url`.
- [ ] Every quantization row references its `introduced_architecture` per book §5.1.1.
- [ ] Pydantic schemas reject malformed YAML (missing required field, unknown field) with line numbers in the error.
- [ ] All slugs join to real entries in a captured ComputePrices `/api/v1/gpus` fixture (e.g. `tests/fixtures/cp_gpus_<YYYY-MM-DD>.json`; the actual file is dated by UTC at capture time, see `scripts/capture_cp_gpus_fixture.py`).
- [ ] `uv run pytest tests/catalog/` green.
- [ ] No live network calls in tests.
- [ ] Rows whose dense FP8/FP4 TFLOPS could not be confirmed against the linked vendor whitepaper at M01 capture time carry `fp8_tflops_dense: null` / `fp4_tflops_dense: null` rather than a guess. M07 treats null as `requires_measurement` per ADR-010 / TPS source enum, so the trust contract holds even before backfill.

---

## Schema-evolution note (ADR-015)

Unlike the upstream-data clients (M02, M03, M04), supplements are OUR own controlled data, so the Pydantic models use `extra="forbid"`. If the YAML grows a typo, we want to know.

---

## When this is done

Commit message:
> `M01: GPU supplement + quantization YAML seeds (12 + 10 rows)`

Mark M01 as ✓ in `spec/INDEX.md`. Move to M02 (critical path) or work M04/M05 in parallel.
