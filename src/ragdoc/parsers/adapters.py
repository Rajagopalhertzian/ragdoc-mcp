"""Optional parser adapters. Both are thin and lazy-imported.

NOTE: these two adapters could not be exercised in the build sandbox (no model downloads / no network
access to cloud parsers). They follow the libraries' documented entry points; verify against the
installed version before relying on them (see docs/KNOWN_LIMITS.md)."""
from __future__ import annotations

import os
from pathlib import Path

from ..security import require_consent
from .base import ParsedDoc, ParserUnavailable


def parse_docling(path: Path, rel: str, file_hash: str) -> ParsedDoc:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ParserUnavailable("Docling is not installed. Run: pip install 'ragdoc-mcp[docling]'") from e
    result = DocumentConverter().convert(str(path))
    md = result.document.export_to_markdown()
    return ParsedDoc(
        rel, file_hash, "docling", "pdf" if path.suffix.lower() == ".pdf" else "text",
        [{"num": 1, "text": md}],
        {"n_pages": 1, "scanned_pages": [], "notes": "docling markdown export; page numbers not preserved"},
    )


def parse_llamaparse(path: Path, rel: str, file_hash: str, manifest: dict) -> ParsedDoc:
    require_consent(manifest, "cloud_parser")
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not key:
        raise ParserUnavailable("Set LLAMA_CLOUD_API_KEY in the environment to use the llamaparse backend.")
    try:
        from llama_parse import LlamaParse  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ParserUnavailable("llama-parse is not installed. Run: pip install llama-parse") from e
    docs = LlamaParse(api_key=key, result_type="markdown").load_data(str(path))
    pages = [{"num": i + 1, "text": d.text} for i, d in enumerate(docs)]
    return ParsedDoc(rel, file_hash, "llamaparse", "pdf", pages, {"n_pages": len(pages), "scanned_pages": []})
