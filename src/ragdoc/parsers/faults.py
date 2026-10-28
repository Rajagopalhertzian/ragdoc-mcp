"""TEST-ONLY fault injectors used to build the failure fixture set (docs/FIXTURES.md).

They run only inside the 'fast' backend when a plan carries ``_faults.parse``. The diagnosis code never
reads them: it works from the parsed text, chunks, index and a second backend, exactly as in real use."""
from __future__ import annotations

import re

from .base import ParsedDoc

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_CID = lambda s: "".join(f"(cid:{ord(c)})" for c in s)  # noqa: E731


def _blank(text: str) -> str:
    return ""


def _scramble_columns(text: str) -> str:
    """Simulate a two-column PDF whose columns are interleaved word by word."""
    out = []
    for line in text.splitlines():
        w = line.split()
        if len(w) >= 4 and not line.lstrip().startswith("#"):
            h = len(w) // 2
            left, right = w[:h], w[h:]
            mixed = [x for pair in zip(left, right) for x in pair] + right[len(left):]
            out.append(" ".join(mixed))
        else:
            out.append(line)
    return "\n".join(out)


def _flatten_tables(text: str) -> str:
    """Simulate a table extracted column-major with the row structure lost."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW.match(lines[i]):
            j = i
            rows = []
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                if not _SEP_ROW.match(lines[j]):
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            ncol = max(len(r) for r in rows)
            for c in range(ncol):
                out.append(" ".join(r[c] for r in rows if c < len(r)))
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _garble_equations(text: str) -> str:
    def rep(m: re.Match) -> str:
        return "$$ " + _CID(m.group(1)) + " $$"

    return re.sub(r"\$\$\s*(.+?)\s*\$\$", rep, text, flags=re.S)


MODES = {"blank_pages": _blank, "scramble_columns": _scramble_columns, "flatten_tables": _flatten_tables, "garble_equations": _garble_equations}


def apply_parse_fault(doc: ParsedDoc, fault: dict | None) -> ParsedDoc:
    if not fault:
        return doc
    files = fault.get("files") or ["*"]
    if "*" not in files and doc.path not in files:
        return doc
    fn = MODES[fault["mode"]]
    for p in doc.pages:
        p["text"] = fn(p["text"])
    if fault["mode"] == "blank_pages":
        doc.meta["scanned_pages"] = [p["num"] for p in doc.pages]
        doc.meta["notes"] = "simulated: scanned pages, OCR disabled"
    return doc
