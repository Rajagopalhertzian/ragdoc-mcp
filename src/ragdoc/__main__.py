from __future__ import annotations

import argparse
import importlib.util
import sys

from . import __version__
from .config import base_dir


def _check() -> int:
    from .embedders import neural_available
    from .parsers import docling_available

    print(f"ragdoc-mcp {__version__}")
    print(f"base dir (RAGDOC_BASE_DIR): {base_dir()}")
    print(f"neural embedder (fastembed): {'yes' if neural_available() else 'no (offline hashing embedder is used)'}")
    print(f"docling: {'yes' if docling_available() else 'no (fast parser only)'}")
    print(f"mcp sdk: {importlib.metadata.version('mcp')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragdoc-mcp", description="RAG Doctor MCP server (stdio by default).")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default loopback). Put auth in front before exposing it.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--check", action="store_true", help="Print environment info and exit.")
    a = ap.parse_args(argv)
    if a.version:
        print(__version__)
        return 0
    if a.check:
        import importlib.metadata  # noqa: F401

        return _check()
    from .server import configure_logging, mcp

    configure_logging()
    if a.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
