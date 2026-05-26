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
