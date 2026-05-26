"""MCP server entry point. M00 stub; replaced by FastMCP server in M09."""

from __future__ import annotations

from whatcanirun import __version__


def main() -> None:
    """Entry point for `uvx whatcanirun-mcp`. M00 prints version; M09 wires FastMCP."""
    print(f"whatcanirun v{__version__} — MCP server not yet implemented (see spec/M09)")


if __name__ == "__main__":
    main()
