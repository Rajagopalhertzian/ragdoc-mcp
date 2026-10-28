"""Split page text into typed blocks: heading, prose, table, equation, code."""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_MD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
HEADING_NUM = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^\n]{2,80})$")
FENCE = re.compile(r"^\s*```")
EQ_BEGIN = re.compile(r"^\s*\\begin\{(equation\*?|align\*?|eqnarray\*?|gather\*?|multline\*?)\}")
TABLE_MD = re.compile(r"^\s*\|.*\|\s*$")
TABLE_WS = re.compile(r"\S(?:\t| {2,})\S.*(?:\t| {2,})\S")
SEP_ROW = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")


@dataclass
class Block:
    kind: str  # heading | prose | table | equation | code
    text: str
    heading_path: str


class HeadingState:
    """Heading stack carried across pages."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    @property
    def path(self) -> str:
        return " > ".join(t for _, t in self.stack)


def _is_heading(line: str, numeric: bool) -> tuple[int, str] | None:
    m = HEADING_MD.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if numeric:
        n = HEADING_NUM.match(line.strip())
        if n and len(n.group(2).split()) <= 12 and not n.group(2).rstrip().endswith("."):
            return n.group(1).count(".") + 1, line.strip()
    return None


def segment(text: str, state: HeadingState, numeric_headings: bool = False) -> list[Block]:
    lines = text.splitlines()
    out: list[Block] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def flush_para() -> None:
        if para:
            out.append(Block("prose", "\n".join(para).strip(), state.path))
            para.clear()

    while i < n:
        line = lines[i]
        if not line.strip():
            flush_para()
            i += 1
            continue
        if FENCE.match(line):
            flush_para()
            j = i + 1
            while j < n and not FENCE.match(lines[j]):
                j += 1
            out.append(Block("code", "\n".join(lines[i : min(j + 1, n)]), state.path))
            i = j + 1
            continue
        s = line.strip()
        if s.startswith("$$"):
            flush_para()
            if s.count("$$") >= 2 and len(s) > 4:
                out.append(Block("equation", line, state.path))
                i += 1
                continue
            j = i + 1
            while j < n and "$$" not in lines[j]:
                j += 1
            out.append(Block("equation", "\n".join(lines[i : min(j + 1, n)]), state.path))
            i = j + 1
            continue
        m = EQ_BEGIN.match(line)
        if m:
            flush_para()
            end = re.compile(r"\\end\{" + re.escape(m.group(1)) + r"\}")
            j = i
            while j < n and not end.search(lines[j]):
                j += 1
            out.append(Block("equation", "\n".join(lines[i : min(j + 1, n)]), state.path))
            i = j + 1
            continue
        h = _is_heading(line, numeric_headings)
        if h:
            flush_para()
            state.push(*h)
            out.append(Block("heading", line.strip(), state.path))
            i += 1
            continue
        if TABLE_MD.match(line) or (TABLE_WS.search(line) and i + 1 < n and TABLE_WS.search(lines[i + 1])):
            flush_para()
            md = bool(TABLE_MD.match(line))
            j = i
            while j < n and ((TABLE_MD.match(lines[j]) if md else TABLE_WS.search(lines[j]))):
                j += 1
            out.append(Block("table", "\n".join(lines[i:j]), state.path))
            i = j
            continue
        para.append(line)
        i += 1
    flush_para()
    return out


def has_markdown_headings(text: str) -> bool:
    return any(HEADING_MD.match(line) for line in text.splitlines())


def table_header_and_rows(table_text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (header_lines, row_lines, column_names) for a markdown or whitespace table."""
    lines = [ln for ln in table_text.splitlines() if ln.strip()]
    if not lines:
        return [], [], []
    header = [lines[0]]
    rest = lines[1:]
    if rest and SEP_ROW.match(rest[0]) and TABLE_MD.match(lines[0]):
        header.append(rest[0])
        rest = rest[1:]
    if TABLE_MD.match(lines[0]):
        cols = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    else:
        cols = [c for c in re.split(r"\t| {2,}", lines[0].strip()) if c]
    return header, rest, cols
