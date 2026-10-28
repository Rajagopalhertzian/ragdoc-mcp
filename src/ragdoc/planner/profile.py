"""profile_corpus: sample the corpus and measure what kinds of content it holds. Read-only."""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from pathlib import Path

from ..build import discover_files
from ..chunkers.blocks import HeadingState, segment
from ..parsers.base import CODE_EXTS, PDF_EXTS
from ..parsers.fast import parse_fast
from ..textutil import count_tokens, sha1_hex, tokens

_STOP_LANG = {
    "en": {"the", "and", "of", "to", "is", "in", "that", "for", "with", "this", "are"},
    "de": {"der", "die", "und", "das", "ist", "nicht", "mit", "von", "den", "zu", "auf"},
    "fr": {"le", "la", "les", "et", "des", "est", "que", "une", "dans", "pour", "pas"},
    "es": {"el", "la", "los", "y", "que", "es", "en", "una", "por", "con", "para"},
}
_MATH_LINE = re.compile(r"[∑∫√≈≤≥±×÷αβγδθλμπσω]|\\(?:frac|sum|int|sqrt|alpha|beta)|\$\$")
PROFILES: dict[str, dict] = {}


def detect_language(text: str) -> str:
    toks = tokens(text)[:2000]
    if len(toks) < 20:
        return "unknown"
    scores = {lang: sum(1 for t in toks if t in sw) for lang, sw in _STOP_LANG.items()}
    lang, hits = max(scores.items(), key=lambda kv: kv[1])
    return lang if hits >= 3 else "unknown"


def profile_corpus(src: Path, sample_size: int = 200, seed: int = 0) -> dict:
    files, skipped = discover_files(src)
    by_ext: Counter = Counter(p.suffix.lower() for p, _ in files)
    rng = random.Random(seed)
    sample = files if len(files) <= sample_size else rng.sample(files, sample_size)
    tok_by_kind: Counter = Counter()
    tables = equations = scanned_pages = sampled_pdf_pages = blank_pages = 0
    pdf_pages_seen: list[int] = []
    langs: Counter = Counter()
    sample_tokens = 0
    errors = 0
    for p, rel in sample:
        ext = p.suffix.lower()
        try:
            doc = parse_fast(p, rel, "profile", max_pages=20 if ext in PDF_EXTS else None)
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        if ext in PDF_EXTS:
            try:
                from pypdf import PdfReader

                pdf_pages_seen.append(len(PdfReader(str(p)).pages))
            except Exception:  # noqa: BLE001
                pdf_pages_seen.append(len(doc.pages))
            sampled_pdf_pages += len(doc.pages)
            scanned_pages += len(doc.meta.get("scanned_pages", []))
            blank_pages += sum(1 for pg in doc.pages if not pg["text"].strip())
            try:  # layout sample to see table structure in digital PDFs (first 5 pages)
                lay = parse_fast(p, rel, "profile", reference=True, max_pages=5)
                for pg in lay.pages:
                    tables += sum(1 for b in segment(pg["text"], HeadingState()) if b.kind == "table")
            except Exception:  # noqa: BLE001
                pass
        text = doc.full_text()
        n = count_tokens(text)
        sample_tokens += n
        if doc.kind == "code":
            tok_by_kind["code"] += n
            continue
        st = HeadingState()
        blocks = []
        for pg in doc.pages:
            blocks.extend(segment(pg["text"], st))
        for b in blocks:
            t = count_tokens(b.text)
            kind = {"heading": "prose"}.get(b.kind, b.kind)
            tok_by_kind[kind] += t
            if b.kind == "table" and ext not in PDF_EXTS:
                tables += 1
            if b.kind == "equation":
                equations += 1
        if ext in PDF_EXTS:
            equations += sum(1 for ln in text.splitlines() if len(_MATH_LINE.findall(ln)) >= 2)
        langs[detect_language(text)] += 1
    n_sampled = max(len(sample) - errors, 1)
    scale = len(files) / n_sampled if n_sampled else 1.0
    avg_pdf_pages = (sum(pdf_pages_seen) / len(pdf_pages_seen)) if pdf_pages_seen else 0.0
    n_pdf = by_ext.get(".pdf", 0)
    text_tokens_est = int(sample_tokens * scale)
    est_pages = int(n_pdf * avg_pdf_pages + (text_tokens_est / 500 if len(files) - n_pdf else 0))
    k_tok = max(sample_tokens, 1) / 1000
    scanned_share = (scanned_pages / sampled_pdf_pages) if sampled_pdf_pages else 0.0
    total_kind = max(sum(tok_by_kind.values()), 1)
    mix = {k: round(tok_by_kind.get(k, 0) / total_kind, 3) for k in ("prose", "table", "equation", "code")}
    kinds = {"code": sum(by_ext.get(e, 0) for e in CODE_EXTS), "pdf": n_pdf}
    kinds["text"] = len(files) - kinds["code"] - kinds["pdf"]
    signals = []
    if tables / k_tok >= 0.5 or mix["table"] >= 0.1:
        signals.append("tables")
    if equations / k_tok >= 0.5:
        signals.append("equations")
    if mix["code"] >= 0.3:
        signals.append("code")
    if scanned_share >= 0.05:
        signals.append("scanned")
    if sum(1 for v in kinds.values() if len(files) and v / len(files) >= 0.1) >= 2:
        signals.append("mixed")
    pid = "prof_" + hashlib.sha1(f"{src}|{len(files)}|{sample_size}|{seed}".encode()).hexdigest()[:10]
    prof = {
        "profile_id": pid, "path": str(src), "files_total": len(files), "files_sampled": n_sampled,
        "by_extension": dict(by_ext.most_common(12)), "kinds": kinds, "content_mix": mix,
        "tables_per_1k_tokens": round(tables / k_tok, 3), "equations_per_1k_tokens": round(equations / k_tok, 3),
        "scanned_share": round(scanned_share, 3), "blank_page_share": round(blank_pages / sampled_pdf_pages, 3) if sampled_pdf_pages else 0.0,
        "languages": {k: round(v / max(sum(langs.values()), 1), 2) for k, v in langs.items()},
        "est_total_pages": est_pages, "est_total_tokens": text_tokens_est, "signals": signals or ["clean_text"],
        "skipped": skipped[:10], "skipped_count": len(skipped), "parse_errors_in_sample": errors,
        "note": "Table/equation detection in PDFs is heuristic (layout sample of the first 5 pages, math-symbol lines).",
    }
    PROFILES[pid] = prof
    return prof


def slug(path: Path) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", path.name).strip("-") or "corpus"
    return (s[:40] + "-" + sha1_hex(str(path), 6)).lstrip("-_")
