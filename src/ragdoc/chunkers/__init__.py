"""Chunkers: heading-aware, token window, table-per-chunk, formula-preserving, AST code."""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

from ..parsers.base import ParsedDoc
from ..textutil import count_tokens, sha1_hex
from .blocks import HeadingState, has_markdown_headings, segment, table_header_and_rows

DEFAULT_CHUNKING = {
    "kind": "heading",  # heading | window
    "max_tokens": 600,
    "overlap_pct": 10,
    "prepend_heading": True,
    "keep_tables_whole": True,
    "keep_formulas_whole": True,
    "table_summary": True,
}


@dataclass
class Chunk:
    chunk_id: str
    file: str
    file_hash: str
    page: int
    heading_path: str
    ctype: str  # prose | table | equation | code
    text: str
    embed_text: str
    tokens: int
    seq: int

    def to_dict(self) -> dict:
        return asdict(self)


class _Out:
    def __init__(self, doc: ParsedDoc, cfg: dict):
        self.doc, self.cfg, self.items = doc, cfg, []

    def emit(self, text: str, ctype: str, path: str, page: int, extra_prefix: str = "") -> None:
        text = text.strip()
        if not text:
            return
        seq = len(self.items)
        prefix = ""
        if self.cfg.get("prepend_heading") and path:
            prefix += path + "\n"
        prefix += extra_prefix
        cid = sha1_hex(f"{self.doc.path}|{self.doc.file_hash}|{seq}|{text[:80]}")
        self.items.append(Chunk(cid, self.doc.path, self.doc.file_hash, page, path, ctype, text, prefix + text, count_tokens(text), seq))


def classify_text(text: str) -> str:
    lines = text.splitlines()
    if sum(1 for ln in lines if ln.strip().startswith("|")) >= 2:
        return "table"
    if "$$" in text or "\\begin{" in text or "\\frac" in text:
        return "equation"
    if "```" in text:
        return "code"
    return "prose"


def window_words(words: list[str], size: int, overlap: int) -> list[str]:
    w = [count_tokens(x) for x in words]
    n, i, out = len(words), 0, []
    while i < n:
        j, used = i, 0
        while j < n and (used + w[j] <= size or j == i):
            used += w[j]
            j += 1
        out.append(" ".join(words[i:j]))
        if j >= n:
            break
        k, back = j, 0
        while k > i + 1 and back + w[k - 1] <= overlap:
            back += w[k - 1]
            k -= 1
        i = k if k > i else j
    return out


def _split_prose(text: str, max_t: int, ov: int) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces: list[str] = []
    cur: list[str] = []
    used = 0
    for s in sents:
        t = count_tokens(s)
        if t > max_t:
            if cur:
                pieces.append(" ".join(cur))
                cur, used = [], 0
            pieces.extend(window_words(s.split(), max_t, ov))
            continue
        if used + t > max_t and cur:
            pieces.append(" ".join(cur))
            carry: list[str] = []
            back = 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if back + pt > ov:
                    break
                carry.insert(0, prev)
                back += pt
            cur, used = carry, back
        cur.append(s)
        used += t
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _tail(text: str, ov: int) -> str:
    if ov <= 0:
        return ""
    words = text.split()
    out, used = [], 0
    for w in reversed(words):
        c = count_tokens(w)
        if used + c > ov:
            break
        out.insert(0, w)
        used += c
    return " ".join(out)


def chunk_heading(doc: ParsedDoc, cfg: dict) -> list[Chunk]:
    max_t = int(cfg["max_tokens"])
    ov = int(max_t * cfg["overlap_pct"] / 100)
    state = HeadingState()
    numeric = not any(has_markdown_headings(p["text"]) for p in doc.pages)
    out = _Out(doc, cfg)
    buf: list[tuple[str, str]] = []
    st = {"tokens": 0, "path": "", "page": 1}

    def flush(carry_overlap: bool = False) -> None:
        if not buf:
            return
        text = "\n\n".join(t for _, t in buf)
        kinds = {k for k, _ in buf}
        ctype = "equation" if "equation" in kinds else "code" if "code" in kinds else "prose"
        out.emit(text, ctype, st["path"], st["page"])
        last_prose = next((t for k, t in reversed(buf) if k == "prose"), "")
        buf.clear()
        st["tokens"] = 0
        if carry_overlap and ov > 0 and last_prose:
            tail = _tail(last_prose, ov)
            if tail:
                buf.append(("prose", tail))
                st["tokens"] = count_tokens(tail)

    def add(kind: str, text: str, path: str, page: int) -> None:
        if not buf:
            st["path"], st["page"] = path, page
        buf.append((kind, text))
        st["tokens"] += count_tokens(text)

    for page in doc.pages:
        for blk in segment(page["text"], state, numeric):
            if blk.kind == "heading":
                flush()
                continue
            if buf and blk.heading_path != st["path"]:
                flush()
            if blk.kind == "table" and cfg.get("keep_tables_whole", True):
                flush()
                _emit_table(out, blk.text, blk.heading_path, page["num"], max_t, cfg)
                continue
            t = count_tokens(blk.text)
            protected = (blk.kind == "equation" and cfg.get("keep_formulas_whole", True)) or blk.kind == "code"
            if protected:
                if buf and st["tokens"] + t > max_t:
                    flush(carry_overlap=False)
                add(blk.kind, blk.text, blk.heading_path, page["num"])
                continue
            if t > max_t:
                flush()
                for piece in _split_prose(blk.text, max_t, ov):
                    out.emit(piece, classify_text(piece) if blk.kind != "prose" else "prose", blk.heading_path, page["num"])
                continue
            if buf and st["tokens"] + t > max_t:
                flush(carry_overlap=True)
                if not buf:
                    st["path"], st["page"] = blk.heading_path, page["num"]
            add("prose" if blk.kind != "table" else "prose", blk.text, blk.heading_path, page["num"])
    flush()
    return out.items


def _emit_table(out: _Out, table_text: str, path: str, page: int, max_t: int, cfg: dict) -> None:
    header, rows, cols = table_header_and_rows(table_text)
    summary = ""
    if cfg.get("table_summary", True) and cols:
        where = f" in '{path}'" if path else ""
        summary = f"Table{where}: columns {', '.join(cols)}.\n"
    limit = max(max_t * 3, 200)
    if count_tokens(table_text) <= limit:
        out.emit(table_text, "table", path, page, extra_prefix=summary)
        return
    group: list[str] = []
    used = count_tokens("\n".join(header))
    for r in rows:
        t = count_tokens(r)
        if group and used + t > limit:
            out.emit("\n".join(header + group), "table", path, page, extra_prefix=summary)
            group, used = [], count_tokens("\n".join(header))
        group.append(r)
        used += t
    if group:
        out.emit("\n".join(header + group), "table", path, page, extra_prefix=summary)


def chunk_window(doc: ParsedDoc, cfg: dict) -> list[Chunk]:
    size = int(cfg["max_tokens"])
    ov = int(size * cfg["overlap_pct"] / 100)
    out = _Out(doc, {**cfg, "prepend_heading": False})
    for page in doc.pages:
        for piece in window_words(page["text"].split(), size, ov):
            out.emit(piece, classify_text(piece), "", page["num"])
    return out.items


_DEF_RE = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\b|class\b|def\b|fn\b|func\b|public\b|private\b|protected\b|static\b|interface\b|struct\b|impl\b|const\s+\w+\s*=\s*(?:async\s*)?\()")


def chunk_code(doc: ParsedDoc, cfg: dict) -> list[Chunk]:
    src = doc.pages[0]["text"] if doc.pages else ""
    max_t = int(cfg["max_tokens"])
    out = _Out(doc, {**cfg, "prepend_heading": False})
    lang = doc.meta.get("language", "")
    if lang == "python":
        try:
            _chunk_python(doc, src, max_t, out)
            if out.items:
                return out.items
        except SyntaxError:
            out.items = []
    _chunk_regex(doc, src, max_t, out)
    return out.items


def _emit_code(out: _Out, text: str, name: str, imports: str) -> None:
    max_t = int(out.cfg["max_tokens"])
    path = f"{out.doc.path}::{name}" if name else out.doc.path
    prefix = f"# file: {out.doc.path}\n" + (f"# imports: {imports}\n" if imports else "")
    pieces = [text] if count_tokens(text) <= max_t * 2 else window_words(text.split("\n"), max_t * 2, 0)
    for p in pieces:
        out.emit(p, "code", path, 1, extra_prefix=prefix)


def _chunk_python(doc: ParsedDoc, src: str, max_t: int, out: _Out) -> None:
    tree = ast.parse(src)
    lines = src.splitlines()
    imports: list[str] = []
    covered: set[int] = set()
    defs = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(", ".join(a.name for a in node.names) if isinstance(node, ast.Import) else f"{node.module}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            defs.append((node, start, node.end_lineno or node.lineno))
    imp = ", ".join(dict.fromkeys(i for i in imports if i))[:200]
    for node, start, end in defs:
        covered.update(range(start, end + 1))
        text = "\n".join(lines[start - 1 : end])
        if isinstance(node, ast.ClassDef) and count_tokens(text) > max_t:
            head_end = node.body[0].end_lineno if node.body else node.lineno
            _emit_code(out, "\n".join(lines[start - 1 : head_end]), node.name, imp)
            for m in node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)) else node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ms = min([m.lineno] + [d.lineno for d in m.decorator_list])
                    _emit_code(out, "\n".join(lines[ms - 1 : m.end_lineno]), f"{node.name}.{m.name}", imp)
        else:
            _emit_code(out, text, node.name, imp)
    rest = "\n".join(ln for i, ln in enumerate(lines, 1) if i not in covered).strip()
    if rest:
        _emit_code(out, rest, "<module>", "")


def _chunk_regex(doc: ParsedDoc, src: str, max_t: int, out: _Out) -> None:
    lines = src.splitlines()
    starts = [i for i, ln in enumerate(lines) if _DEF_RE.match(ln)]
    if not starts:
        for piece in window_words(src.split("\n"), max_t * 2, 0):
            out.emit(piece, "code", doc.path, 1, extra_prefix=f"# file: {doc.path}\n")
        return
    bounds = ([0] if starts[0] > 0 else []) + starts + [len(lines)]
    for a, b in zip(bounds, bounds[1:]):
        seg = "\n".join(lines[a:b]).strip()
        if seg:
            name = lines[a].strip()[:60] if a in starts else "<header>"
            _emit_code(out, seg, name, "")


def chunk_document(doc: ParsedDoc, cfg: dict | None = None) -> list[Chunk]:
    c = {**DEFAULT_CHUNKING, **(cfg or {})}
    if doc.kind == "code":
        return chunk_code(doc, c)
    if c["kind"] == "window":
        return chunk_window(doc, c)
    return chunk_heading(doc, c)
