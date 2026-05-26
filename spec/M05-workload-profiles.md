# M05 — Workload Profile Seeds

**Status:** ⬜ Not started
**Effort:** 1h
**Dependencies:** M00
**Unblocks:** M09 (`budget_to_plan` uses workload profiles to compute est_total_prompts)
**Parallel-safe:** yes

> Read [`SHARED.md`](SHARED.md) first.

---

## Goal

3 hand-curated `WorkloadProfile` rows that turn dollar budgets into prompt counts. A profile is `(avg_input_tokens, avg_output_tokens)` — the shape of typical traffic for that use case.

3 profiles, not 30. v1 keeps the option surface narrow. More profiles ship only if usage data shows we need them (see v2 trigger conditions in `INDEX.md`).

---

## Scope

### `seeds/workload_profiles.yaml`

```yaml
- slug: code_completion
  display_name: "Code completion (Copilot-style)"
  avg_input_tokens: 800              # surrounding code as context
  avg_output_tokens: 120             # one completion suggestion
  is_default: false
  description: "Editor-integrated completion. Heavy on context, light on output."

- slug: chat_assistant
  display_name: "Chat assistant (general Q&A)"
  avg_input_tokens: 400              # system prompt + question
  avg_output_tokens: 250             # paragraph-length answer
  is_default: true                   # the default when caller omits workload_profile
  description: "Conversational use. Balanced input/output. Used when caller does not specify."

- slug: batch_eval
  display_name: "Batch evaluation"
  avg_input_tokens: 1200             # long context
  avg_output_tokens: 200             # short structured output
  is_default: false
  description: "Eval suites, classification pipelines, retrieval grading. Long input, short output."
```

### Pydantic schema (`src/whatcanirun/catalog/workload.py`)

```python
class WorkloadProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # our own data; strict
    slug: str
    display_name: str
    avg_input_tokens: int
    avg_output_tokens: int
    is_default: bool
    description: str

    @field_validator("avg_input_tokens", "avg_output_tokens")
    @classmethod
    def positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("token counts must be positive")
        return v
```

### Loader (`src/whatcanirun/catalog/loaders.py` — extend M01)

```python
def load_workload_profiles(path: Path) -> list[WorkloadProfile]: ...
```

---

## Out of scope

- Token mix per language (Python vs Bash for code completion) — gated on usage data.
- Streaming vs non-streaming profile splits — gated on usage data.
- User-defined custom profiles passed through MCP tool args — v1 picks from the 3 seeded.

---

## Vertical slices

1. **Slice A: Pydantic model** — TDD: extra field rejected; negative token count rejected.
2. **Slice B: Loader** — TDD: loading 3 rows produces 3 WorkloadProfile objects.
3. **Slice C: Exactly-one default** — TDD: loader raises if `is_default=True` count ≠ 1.

---

## Acceptance criteria

- [ ] `seeds/workload_profiles.yaml` has exactly 3 rows.
- [ ] Exactly one row has `is_default: true` (`chat_assistant`).
- [ ] Pydantic schema is strict (`extra="forbid"`).
- [ ] Loader raises with line number on malformed YAML.
- [ ] `uv run pytest tests/catalog/test_workload.py` green.

---

## When done

Commit:
> `M05: workload profile seeds (3 rows)`

Mark M05 ✓ in `INDEX.md`. Continue with M06.
