"""Parser registry: fast (default), reference (second backend), docling (local, optional), llamaparse (cloud, opt-in)."""
from __future__ import annotations

from pathlib import Path

from ..errors import RagDocError
from .adapters import parse_docling, parse_llamaparse
from .base import CODE_EXTS, SUPPORTED_EXTS, ParsedDoc, ParserUnavailable
from .fast import parse_fast
from .faults import apply_parse_fault

BACKENDS = ("fast", "reference", "docling", "llamaparse")

__all__ = ["BACKENDS", "CODE_EXTS", "SUPPORTED_EXTS", "ParsedDoc", "ParserUnavailable", "docling_available", "parse_file"]


def docling_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("docling") is not None


def parse_file(path: Path, rel: str, file_hash: str, backend: str = "fast", *, manifest: dict | None = None, faults: dict | None = None) -> ParsedDoc:
    if backend == "fast":
        return apply_parse_fault(parse_fast(path, rel, file_hash), (faults or {}).get("parse"))
    if backend == "reference":
        return parse_fast(path, rel, file_hash, reference=True)
    if backend == "docling":
        return parse_docling(path, rel, file_hash)
    if backend == "llamaparse":
        return parse_llamaparse(path, rel, file_hash, manifest or {})
    raise RagDocError(f"Unknown parser backend {backend!r}.", hint=f"Choose one of {BACKENDS}.", code="BAD_BACKEND")
