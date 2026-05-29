"""Pydantic schema for `seeds/workload_profiles.yaml` rows.

A workload profile pins the shape of typical traffic for a use case
— `(avg_input_tokens, avg_output_tokens)`. M09's `budget_to_plan`
tool divides a dollar budget by the per-prompt cost computed from
these two values to produce `est_total_prompts`, so the trust
contract demands they're positive and explicit (not defaulted from
prior rows).

`extra="forbid"` follows the M01/M03 convention for our own
controlled-data schemas: a YAML typo (e.g. `is_defualt` instead of
`is_default`) fails loudly at load rather than silently dropping
into an unknown-field bucket where it would leave every row
non-default — exactly the kind of "every model has the same
defaulted property" failure mode that's expensive to debug after
the fact.

The cross-row "exactly one row has `is_default=True`" invariant is
enforced by `load_workload_profiles` (Slice C), not by Pydantic —
it spans the row list and only the loader sees the full set.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class WorkloadProfile(BaseModel):
    """One row of `seeds/workload_profiles.yaml`."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    display_name: str
    avg_input_tokens: int
    avg_output_tokens: int
    is_default: bool
    description: str

    @field_validator("avg_input_tokens", "avg_output_tokens")
    @classmethod
    def positive(cls, v: int) -> int:
        """Zero or negative token counts would let `budget_to_plan`
        divide by zero or produce a negative prompt count — surface
        the bad row at load time rather than as a non-finite
        `est_total_prompts` in the trust envelope."""
        if v <= 0:
            raise ValueError("token counts must be positive")
        return v
