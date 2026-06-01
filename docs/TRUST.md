# The Trust Contract

> **Placeholder.** M11 populates this with the full deep-dive.

Summary (full details in `spec/SHARED.md`):

Every numerical tool response from this server includes a `trust_envelope`:

```python
TrustEnvelope(
    sources=[...],                       # which upstreams contributed
    confidence=min(breakdown.values()),  # weakest-link rollup
    confidence_breakdown={               # per-domain
        "pricing": ...,
        "fit_check": ...,
        "throughput": ...,
        "model_architecture": ...,
        "gpu_specs": ...,
        "freshness": ...,
    },
    assumptions={...},                   # what we held fixed
    caveats=[...],                       # what we explicitly do NOT model
    freshness={...},                     # per-source last-updated timestamps
    verify_links=[...],                  # audit upstream
)
```

The trust contract is the product. Everything else is plumbing.

## Methodology citations

The TPS-estimator's Tier 3 bandwidth heuristic
(`KERNEL_EFFICIENCY_SINGLE_STREAM = 0.75`, applied as
`predicted_tps = bandwidth_gbps / weights_bytes_per_token * 0.75`)
follows Kiely 2026, *Inference Engineering*, §2.4.2 "LLM Inference
Bottlenecks." The book teaches the arithmetic-intensity analysis we
use to argue that LLM decode is memory-bound at low-to-medium batch
sizes, which is the load-bearing assumption for Tier 3's
single-stream estimates.

Recommended reading for understanding the heuristic + the GPU-spec
tables ComputePrices reports:

- Kiely, Philip. *Inference Engineering.* 2026.
  Specifically §2.4 (Calculating Inference Bottlenecks) and
  §3.2 (GPU Architecture Generations).
