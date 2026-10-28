"""Fast parser (pypdf, plain text, markdown, code) and the 'reference' second backend."""
from __future__ import annotations

from pathlib import Path

from ..config import Limits
from ..errors import RagDocError
from .base import CODE_EXTS, PDF_EXTS, TEXT_EXTS, ParsedDoc


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _pdf_pages(path: Path, layout: bool, max_pages: int | None = None) -> tuple[list[dict], list[int]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise RagDocError(f"{path.name} is password protected.", hint="Remove the password or skip this file.", code="ENCRYPTED")
        except RagDocError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RagDocError(f"{path.name} could not be decrypted: {e}", code="ENCRYPTED") from e
    pages: list[dict] = []
    scanned: list[int] = []
    n = min(len(reader.pages), Limits.max_pdf_pages, max_pages or Limits.max_pdf_pages)
    for i in range(n):
        page = reader.pages[i]
        try:
            text = page.extract_text(extraction_mode="layout") if layout else page.extract_text()
        except Exception:  # noqa: BLE001
            text = ""
        text = text or ""
        if not text.strip():
            try:
                has_img = len(page.images) > 0
            except Exception:  # noqa: BLE001
                has_img = False
            if has_img:
                scanned.append(i + 1)
        pages.append({"num": i + 1, "text": text})
    return pages, scanned


def parse_fast(path: Path, rel: str, file_hash: str, *, reference: bool = False, max_pages: int | None = None) -> ParsedDoc:
    """reference=True selects the alternate extraction (pypdf layout mode for PDFs).
    For text and code there is no alternate, so both backends read the file as-is."""
    ext = path.suffix.lower()
    name = "reference" if reference else "fast"
    if ext in PDF_EXTS:
        pages, scanned = _pdf_pages(path, layout=reference, max_pages=max_pages)
        meta = {"n_pages": len(pages), "scanned_pages": scanned}
        return ParsedDoc(rel, file_hash, name, "pdf", pages, meta)
    if ext in CODE_EXTS:
        return ParsedDoc(rel, file_hash, name, "code", [{"num": 1, "text": _read_text(path)}], {"language": CODE_EXTS[ext], "n_pages": 1, "scanned_pages": []})
    if ext in TEXT_EXTS:
        raw = _read_text(path)
        chunks = raw.split("\f") if "\f" in raw else [raw]
        pages = [{"num": i + 1, "text": c} for i, c in enumerate(chunks)]
        return ParsedDoc(rel, file_hash, name, "text", pages, {"n_pages": len(pages), "scanned_pages": []})
    raise RagDocError(f"Unsupported file type: {ext}", code="UNSUPPORTED")
