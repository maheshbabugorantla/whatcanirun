"""M09 Slice A + K: FastMCP server skeleton + instructions string.

These assertions are the public contract the LLM client sees on
every MCP `initialize` handshake. The server's `name` is what
shows up in client UI; the `instructions` string is the prose
that teaches the LLM how to speak trust-contract-respecting
language about every tool response.

The substring checks aren't testing the prose word-for-word
(that would couple the test to copy edits). They're asserting
the *concepts* the instructions must communicate — every one
of these is load-bearing per spec/M09 § "The FastMCP.instructions
string".
"""

from __future__ import annotations

from fastmcp import FastMCP

from whatcanirun.server import mcp


def test_mcp_is_fastmcp_instance() -> None:
    """Server is a `FastMCP` instance — not a thin wrapper or a
    raw dict. Future slices register tools / resources / prompts
    against this object, so the test asserts the type that
    decorator-based registration requires."""
    assert isinstance(mcp, FastMCP)


def test_server_name_is_whatcanirun() -> None:
    """`name` shows up in MCP client UI (Claude Desktop's server
    list, Cursor's MCP panel). Hard-code it to the project name
    so users can find it without guessing."""
    assert mcp.name == "whatcanirun"


def test_instructions_string_is_present_and_bounded() -> None:
    """Spec/M09 acceptance: 'instructions string present,
    length-checked (rough sanity bound — not empty, not
    multi-thousand-word)'. The 4K-char cap protects against
    client-side truncation; the 200-char floor catches a stub
    that forgot to actually write the prose."""
    assert mcp.instructions is not None
    assert 200 < len(mcp.instructions) < 4000


def test_instructions_string_mentions_trust_contract_concepts() -> None:
    """Every concept here is load-bearing in spec/M09 § 'The
    FastMCP.instructions string'. If a future edit drops one,
    this test goes red — the LLM client wouldn't know to surface
    the corresponding signal to the user."""
    instructions = mcp.instructions or ""
    assert "trust_envelope" in instructions
    assert "sources" in instructions
    assert "confidence_breakdown" in instructions
    assert "caveats" in instructions
    assert "freshness" in instructions
    assert "verify_links" in instructions


def test_instructions_string_calls_out_critical_relay_rules() -> None:
    """The instructions string isn't just a list of fields — it
    tells the LLM client *how* to relay them. These four rules
    are the spec's explicit per-domain instructions (refusal
    explanation, fits-doesn't-mean-sufficient, spot preemption,
    availability disclaimer). Each catches a specific failure
    mode where a less careful LLM would relay something the
    server can't defend."""
    instructions = mcp.instructions or ""
    assert "sufficiency_caveat" in instructions  # fits != sufficient
    assert "spot" in instructions  # preemption risk
    assert "availability" in instructions  # rentability disclaimer
    assert "workload_assumption" in instructions  # derived-count caveat


def test_instructions_string_mentions_honesty_posture() -> None:
    """The closing posture in spec/M09 — 'designed to be honest,
    not optimistic' — is the prose that anchors the LLM client's
    voice when two numbers disagree. Without it the model might
    silently pick the more optimistic one."""
    instructions = (mcp.instructions or "").lower()
    assert "honest" in instructions


def test_main_runs_server_over_stdio() -> None:
    """`uvx whatcanirun-mcp` (the project.scripts entry point in
    pyproject.toml) must launch the server on stdio. We don't
    actually start the asyncio loop here — that would block —
    we just assert `main` is wired up and callable. Slice A
    of M09 covers the wiring; an actual handshake roundtrip is
    M11's golden-path job (live stdio fixture)."""
    from whatcanirun.server import main

    assert callable(main)
