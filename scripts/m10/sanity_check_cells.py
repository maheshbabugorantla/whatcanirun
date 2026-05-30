"""V1 sanity-check tool for M10 candidate benchmark cells.

Validates a YAML/JSON file of candidate `BenchmarkCell` rows against
methodology + join-key + sanity rules before the rows are merged
into `seeds/benchmark_cells.parquet`. Exit code: 0 clean, 1 warning,
2 blocking error.

Each check is a pure function `(cell, ctx) -> CheckResult` that the
CLI iterates over. The check set is documented in `scripts/m10/README.md`
§ "Tools / V1"; this file owns the implementations.

The CLI surface (`main`) lands once enough checks exist to make the
tool useful. Until then, the module is just a check library that
tests import directly.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from whatcanirun.catalog.benchmark_cells import BenchmarkCell
from whatcanirun.catalog.seed_schemas import Quantization, TrackedModelRow
from whatcanirun.pricing.projections import GpuCatalogRow

# Stale-numbers threshold per spec's "Stale numbers" pitfall: a cell
# from 2024 with vLLM 0.4 is less applicable to 2026 stacks. 18 months
# is the rough lifetime over which an engine version stays current.
_RECENCY_THRESHOLD = dt.timedelta(days=30 * 18)

# Semver-ish: MAJOR.MINOR[.(PATCH|x)]. Rejects "latest", "main", "dev".
_ENGINE_VERSION_RE = re.compile(r"^\d+\.\d+(\.(\d+|x))?$")

# Form-factor disambiguation per M10 pitfall "Cross-pollinated GPUs":
# these data-center SKUs ship in multiple form factors with different
# bandwidth (e.g. H100 SXM5 3350 GB/s vs H100 PCIe 2000 GB/s). Cells
# whose notes mention any of these GPU names without disclosing which
# form factor was tested are ambiguous and get a blocking error.
_AMBIGUOUS_GPU_NAMES = ("h100", "h200", "a100")
_FORM_FACTOR_TOKENS = ("sxm", "pcie", "nvl", "oam")

# Same kernel-efficiency constant tps_estimator Tier 3 uses; kept here
# so the heuristic check stays in lock-step without an inter-module
# dependency on the production estimator.
_KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75

# Heuristic comparison band — anything outside [0.5, 1.5] (i.e. ±50%)
# is worth a curator's attention, but not a blocking error since the
# actual TPS may legitimately differ from the heuristic.
_HEURISTIC_RATIO_BAND_LOW = 0.5
_HEURISTIC_RATIO_BAND_HIGH = 1.5

# Batch-scaling sanity threshold: if batched_tps / (batch * single_tps)
# is >= this, the cell is probably reporting per-stream throughput as
# if it were per-batch. ADR-010's verified 6x wrong at batch=128 puts
# real sub-linear ratios closer to 0.4 in practice.
_LINEAR_BATCH_THRESHOLD = 0.85

Severity = Literal["pass", "warn", "error"]


@dataclass(frozen=True)
class CheckResult:
    """The protocol every check returns. Frozen so the CLI can
    safely thread results through aggregation without defensive
    copies; severity is a closed Literal so exit-code mapping is
    exhaustive at the type level."""

    severity: Severity
    message: str


@dataclass(frozen=True)
class CheckContext:
    """Side-data the catalog-aware checks need. Frozen so a single
    context can be threaded safely across every cell in a candidate
    file without checks mutating shared state.

    Fields:
    - `existing_cells` — the canonical parquet's current rows, used
      by `check_op_point_unique` (cycle 5) and
      `check_batch_scaling_not_linear` (cycle 10).
    - `gpu_catalog` — CP's gpu_catalog rows, used by
      `check_gpu_slug_exists` (cycle 6) and
      `check_gpu_form_factor_disambiguated` (cycle 9).
    - `quantizations` — seed quantizations, used by
      `check_quant_slug_exists` (cycle 7) and the MoE-aware
      bandwidth heuristic (cycle 11) for bits_per_weight.
    - `tracked_models` — merged tracked-models (project seed +
      user_models.yaml), used by `check_model_slug_exists`
      (cycle 8) and cycle 11 for active_params_b / total_params_b
      lookups."""

    existing_cells: list[BenchmarkCell]
    gpu_catalog: list[GpuCatalogRow]
    quantizations: list[Quantization]
    tracked_models: list[TrackedModelRow]


# ---------------------------------------------------------------- checks


def check_source_url_well_formed(cell: BenchmarkCell) -> CheckResult:
    """`source_url` must parse as an absolute http(s) URL — scheme
    must be `http` or `https`, and netloc must be non-empty. The
    Pydantic `min_length=1` on the field accepts strings like
    `javascript:alert(1)`, `example.com/article`, and `https://`
    which can't lead a reader to a methodology disclosure."""
    parsed = urlparse(cell.source_url)
    if parsed.scheme not in {"http", "https"}:
        return CheckResult(
            severity="error",
            message=(
                f"source_url scheme must be http or https, got "
                f"{parsed.scheme!r} for {cell.source_url!r}"
            ),
        )
    if not parsed.netloc:
        return CheckResult(
            severity="error",
            message=(f"source_url has no host (netloc); got {cell.source_url!r}"),
        )
    return CheckResult(severity="pass", message="source_url is well-formed")


def check_engine_version_format(cell: BenchmarkCell) -> CheckResult:
    """`engine_version` must be `MAJOR.MINOR[.(PATCH|x)]`. Rejects
    `latest`, `main`, `dev`, and empty strings, all of which break
    auditability — a cell tagged `latest` today means a different
    engine next quarter."""
    if not _ENGINE_VERSION_RE.match(cell.engine_version):
        return CheckResult(
            severity="error",
            message=(
                f"engine_version must be semver-shaped "
                f"(MAJOR.MINOR[.(PATCH|x)]); got {cell.engine_version!r}. "
                f"Floating refs like 'latest' / 'main' break the audit trail."
            ),
        )
    return CheckResult(
        severity="pass", message=f"engine_version {cell.engine_version!r} is well-formed"
    )


def check_measured_at_recency(cell: BenchmarkCell, *, _today: dt.date | None = None) -> CheckResult:
    """`measured_at` should be within ~18 months of today. Older
    numbers correspond to older engine/driver/PyTorch versions and
    warrant a warning rather than auto-rejection (the curator can
    still keep the cell if the methodology is solid). Future dates
    are hard errors — they can't be a real measurement.

    `_today` is injected for deterministic testing across the
    calendar; production callers leave it as None to use the real
    clock."""
    today = _today if _today is not None else dt.date.today()
    if cell.measured_at > today:
        return CheckResult(
            severity="error",
            message=(
                f"measured_at {cell.measured_at.isoformat()} is in the "
                f"future relative to today {today.isoformat()}; almost "
                f"certainly a typo."
            ),
        )
    age = today - cell.measured_at
    if age > _RECENCY_THRESHOLD:
        months = age.days // 30
        return CheckResult(
            severity="warn",
            message=(
                f"measured_at {cell.measured_at.isoformat()} is "
                f"~{months} months stale (>18 month threshold); the "
                f"engine version in this cell may no longer be representative."
            ),
        )
    return CheckResult(
        severity="pass",
        message=f"measured_at {cell.measured_at.isoformat()} is recent",
    )


def check_methodology_complete(cell: BenchmarkCell) -> CheckResult:
    """`notes` must be ≥30 chars AND mention both the cell's
    `engine_version` and its `batch_size`. The 30-char floor catches
    'see source' / 'reference' placeholder rows; the engine+batch
    mention catches the M10 pitfall of blog posts that don't
    disclose methodology."""
    if len(cell.notes) < 30:
        return CheckResult(
            severity="error",
            message=(
                f"notes is too short ({len(cell.notes)} chars < 30); "
                f"must include a 1-2 sentence methodology summary"
            ),
        )
    if cell.engine_version not in cell.notes:
        return CheckResult(
            severity="error",
            message=(
                f"notes does not mention engine_version "
                f"{cell.engine_version!r}; the curator must explain "
                f"which version was measured"
            ),
        )
    batch_token_a = f"batch={cell.batch_size}"
    batch_token_b = f"batch_size={cell.batch_size}"
    if batch_token_a not in cell.notes and batch_token_b not in cell.notes:
        return CheckResult(
            severity="error",
            message=(
                f"notes does not mention batch_size {cell.batch_size!r}; "
                f"expected literal 'batch={cell.batch_size}' or "
                f"'batch_size={cell.batch_size}' in notes"
            ),
        )
    return CheckResult(severity="pass", message="notes contains required methodology fields")


def _op_point_key(cell: BenchmarkCell) -> tuple[str, str, str, int, int, int]:
    """The six-tuple that BenchmarkCell uses as its primary key
    for tps_estimator Tier 1b matching. Two cells with the same
    key but different decode_tps are an ambiguity the tool path
    can't resolve without a tiebreaker."""
    return (
        cell.gpu_slug,
        cell.model_slug,
        cell.quant_slug,
        cell.tp_size,
        cell.batch_size,
        cell.context_length,
    )


def check_op_point_unique(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """Reject candidate cells whose op-point key already exists in
    the canonical parquet. The Tier 1b matcher takes the first
    match it finds, so silently shadowing an existing row is
    behavior the curator must opt into explicitly (by deleting the
    old row in the same PR). Errors here are blocking."""
    key = _op_point_key(cell)
    for existing in ctx.existing_cells:
        if _op_point_key(existing) == key:
            return CheckResult(
                severity="error",
                message=(
                    f"duplicate op-point {key!r} already in the canonical "
                    f"parquet (existing decode_tps={existing.decode_tps}, "
                    f"candidate decode_tps={cell.decode_tps}). If this is "
                    f"a re-measurement, delete the existing row in the same PR."
                ),
            )
    return CheckResult(severity="pass", message=f"op-point {key!r} is new to the parquet")


def check_gpu_slug_exists(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """`cell.gpu_slug` must resolve to a row in CP's gpu_catalog.
    Without an exact match, tps_estimator Tier 1b can't join the
    cell to a real GPU's bandwidth + form factor - the row is dead
    data. An empty catalog is also a blocking error (better to fail
    loud than silently pass every slug)."""
    known = {row.slug for row in ctx.gpu_catalog}
    if cell.gpu_slug not in known:
        return CheckResult(
            severity="error",
            message=(
                f"gpu_slug {cell.gpu_slug!r} does not resolve in CP's "
                f"gpu_catalog (known slugs: {sorted(known)[:5]}...). "
                f"Verify the slug matches CP's `gpus` endpoint."
            ),
        )
    return CheckResult(
        severity="pass", message=f"gpu_slug {cell.gpu_slug!r} resolves in gpu_catalog"
    )


def check_quant_slug_exists(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """`cell.quant_slug` must resolve to a row in
    `seeds/quantizations.yaml`. Without that, fit_check can't get
    `bits_per_weight` and tps_estimator can't compute memory
    traffic. Blocking error."""
    known = {q.slug for q in ctx.quantizations}
    if cell.quant_slug not in known:
        return CheckResult(
            severity="error",
            message=(
                f"quant_slug {cell.quant_slug!r} does not resolve in "
                f"seeds/quantizations.yaml (known slugs: {sorted(known)}). "
                f"Add the quantization to the YAML before adding cells "
                f"that reference it."
            ),
        )
    return CheckResult(
        severity="pass", message=f"quant_slug {cell.quant_slug!r} resolves in quantizations"
    )


def check_model_slug_exists(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """`cell.model_slug` must resolve to a row in merged tracked_models
    (project seed + user_models.yaml). Without that, HfModelSync
    can't find the HF repo and the cell's join keys are stranded.
    Blocking error; message hints at the fix path."""
    known = {row.slug for row in ctx.tracked_models}
    if cell.model_slug not in known:
        return CheckResult(
            severity="error",
            message=(
                f"model_slug {cell.model_slug!r} does not resolve in "
                f"merged tracked_models. Add an entry to "
                f"seeds/tracked_models.yaml (or user_models.yaml) "
                f"mapping the slug to its HF repo_id before adding "
                f"cells that reference it."
            ),
        )
    return CheckResult(
        severity="pass",
        message=f"model_slug {cell.model_slug!r} resolves in tracked_models",
    )


def check_gpu_form_factor_disambiguated(cell: BenchmarkCell) -> CheckResult:
    """If `notes` mentions an ambiguous data-center GPU name
    (H100, H200, A100) without disclosing the form factor (SXM,
    PCIe, NVL, OAM), the cell's bandwidth assumptions are unclear.
    Blocking error per M10 pitfall.

    Single-form-factor GPUs (L40S, RTX series, etc.) don't trigger
    the check. The check is case-insensitive and tolerates the
    common 'A100 80GB SXM' / 'H100 SXM5' variants."""
    notes_lower = cell.notes.lower()
    for gpu_name in _AMBIGUOUS_GPU_NAMES:
        if gpu_name in notes_lower and not any(
            token in notes_lower for token in _FORM_FACTOR_TOKENS
        ):
            return CheckResult(
                severity="error",
                message=(
                    f"notes mentions {gpu_name.upper()} but does not "
                    f"disclose a form factor (SXM, PCIe, NVL, or OAM). "
                    f"{gpu_name.upper()} ships in multiple form factors "
                    f"with different bandwidth; the cell is ambiguous "
                    f"without one of these tokens in notes."
                ),
            )
    return CheckResult(
        severity="pass",
        message="form factor either disambiguated or not required",
    )


def check_batch_scaling_not_linear(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """For batch>1 cells, find the single-stream (batch=1) peer with
    the same (gpu, model, quant, tp, ctx) and verify the batched
    decode_tps is sub-linear vs `batch * single_stream_tps`. A ratio
    above 0.85 likely means the source mislabeled per-stream
    throughput as per-batch — blocking error per ADR-010.

    If no single-stream peer exists in `ctx.existing_cells`, return
    warn ('cannot verify') rather than pass — the curator should
    add the single-stream peer to enable the check."""
    if cell.batch_size == 1:
        return CheckResult(severity="pass", message="batch=1; scaling check does not apply")
    peer = None
    for existing in ctx.existing_cells:
        if (
            existing.gpu_slug == cell.gpu_slug
            and existing.model_slug == cell.model_slug
            and existing.quant_slug == cell.quant_slug
            and existing.tp_size == cell.tp_size
            and existing.context_length == cell.context_length
            and existing.batch_size == 1
        ):
            peer = existing
            break
    if peer is None:
        return CheckResult(
            severity="warn",
            message=(
                f"no single-stream (batch=1) peer found in existing parquet "
                f"for ({cell.gpu_slug}, {cell.model_slug}, {cell.quant_slug}, "
                f"tp={cell.tp_size}, ctx={cell.context_length}); cannot "
                f"verify sub-linear batch scaling"
            ),
        )
    ratio = cell.decode_tps / (cell.batch_size * peer.decode_tps)
    if ratio >= _LINEAR_BATCH_THRESHOLD:
        return CheckResult(
            severity="error",
            message=(
                f"batched decode_tps {cell.decode_tps} ~= batch_size "
                f"{cell.batch_size} * single_stream_tps {peer.decode_tps} "
                f"(ratio {ratio:.2f} >= {_LINEAR_BATCH_THRESHOLD}); per "
                f"ADR-010 batched throughput is sub-linear. Source likely "
                f"reports per-stream as per-batch; re-check methodology."
            ),
        )
    return CheckResult(
        severity="pass",
        message=f"batched scaling ratio {ratio:.2f} is sub-linear (peer found)",
    )


def _params_b_for_traffic(model_row: TrackedModelRow) -> tuple[float, str] | None:
    """Returns the parameter count to use for memory-traffic prediction
    plus a short label ('active' or 'total') for diagnostic messages.
    Sparse / MoE models advertise both `total_params_b` and
    `active_params_b`; only the active subset is read per token, so
    the heuristic over-predicts memory traffic by total/active if you
    naively use total. The prototype caught DeepSeek-V3 at a 404%
    over-prediction this way."""
    if model_row.active_params_b is not None and model_row.active_params_b > 0:
        return (model_row.active_params_b, "active")
    if model_row.total_params_b is not None and model_row.total_params_b > 0:
        return (model_row.total_params_b, "total")
    return None


def check_decode_tps_vs_bandwidth_heuristic(cell: BenchmarkCell, ctx: CheckContext) -> CheckResult:
    """Compare cell.decode_tps against the bandwidth heuristic
    `(bandwidth_gbps / weights_bytes_per_token) * KERNEL_EFFICIENCY`.
    Only fires at batch=1 (the heuristic doesn't apply to batched
    throughput per ADR-010). Warns when ratio falls outside [0.5, 1.5];
    skips silently when bandwidth / params / bits_per_weight aren't
    available (data gap is the curator's signal, not their fault)."""
    if cell.batch_size != 1:
        return CheckResult(
            severity="pass",
            message="batch>1; bandwidth heuristic does not apply, skipped",
        )
    gpu_row = next((g for g in ctx.gpu_catalog if g.slug == cell.gpu_slug), None)
    quant_row = next((q for q in ctx.quantizations if q.slug == cell.quant_slug), None)
    model_row = next((m for m in ctx.tracked_models if m.slug == cell.model_slug), None)
    if gpu_row is None or quant_row is None or model_row is None:
        return CheckResult(
            severity="pass",
            message=(
                f"skipped: missing catalog row "
                f"(gpu={gpu_row is not None}, quant={quant_row is not None}, "
                f"model={model_row is not None})"
            ),
        )
    bw_raw = gpu_row.specs.get("memory_bandwidth_gbps")
    if not isinstance(bw_raw, (int, float)) or bw_raw <= 0:
        return CheckResult(
            severity="pass",
            message=f"skipped: gpu_slug {cell.gpu_slug!r} missing memory_bandwidth_gbps",
        )
    bandwidth_gbps = float(bw_raw)
    params_pair = _params_b_for_traffic(model_row)
    if params_pair is None:
        return CheckResult(
            severity="pass",
            message=(
                f"skipped: tracked_models row for {cell.model_slug!r} "
                f"missing both total_params_b and active_params_b"
            ),
        )
    params_b, params_label = params_pair
    weights_bytes_per_token = params_b * 1e9 * quant_row.bits_per_weight / 8.0
    peak_tps = bandwidth_gbps * 1e9 / weights_bytes_per_token
    predicted = peak_tps * _KERNEL_EFFICIENCY_SINGLE_STREAM
    ratio = cell.decode_tps / predicted
    if _HEURISTIC_RATIO_BAND_LOW <= ratio <= _HEURISTIC_RATIO_BAND_HIGH:
        return CheckResult(
            severity="pass",
            message=(
                f"actual {cell.decode_tps:.1f} tok/s within band of "
                f"predicted {predicted:.1f} tok/s "
                f"({params_label}_params_b={params_b}); ratio {ratio:.2f}"
            ),
        )
    return CheckResult(
        severity="warn",
        message=(
            f"actual {cell.decode_tps:.1f} tok/s vs predicted "
            f"{predicted:.1f} tok/s ({params_label}_params_b={params_b}, "
            f"bw={bandwidth_gbps:.0f} GB/s); ratio {ratio:.2f} outside "
            f"[{_HEURISTIC_RATIO_BAND_LOW}, {_HEURISTIC_RATIO_BAND_HIGH}] band. "
            f"Curator should re-verify the source's methodology."
        ),
    )
