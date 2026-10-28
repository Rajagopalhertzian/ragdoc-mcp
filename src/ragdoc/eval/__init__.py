"""Evaluation: question generation, metrics, domain_eval."""
from __future__ import annotations

import json
import random
import re
import uuid
from collections import Counter

from ..collection import Collection, atomic_write_json, now_iso
from ..retrieval.pipeline import Pipeline, Trace
from ..textutil import STOP, content_words, find_answer, is_number, snippet, tokens

CTYPES = ("prose", "table", "equation", "code")
# Fixed chunking used ONLY to generate questions, so the question set does not depend on the chunking under test.
REFERENCE_CHUNKING = {"kind": "heading", "max_tokens": 400, "overlap_pct": 0, "prepend_heading": False, "keep_tables_whole": True, "keep_formulas_whole": True, "table_summary": False}
_SENT = re.compile(r"(?<=[.!?])\s+")
_DEF = re.compile(r"^\s*(?:async\s+)?def\s+\w+\([^)]*\)")
_EQ = re.compile(r"\$\$\s*(.+?)\s*\$\$", re.S)


def _heading_last(path: str) -> str:
    return path.split(" > ")[-1] if path else ""


def reference_rows(coll: Collection, max_files: int = 300) -> list[dict]:
    """Chunk the PARSED text with a fixed reference chunker (independent of the live chunking) to source test questions."""
    from ..chunkers import chunk_document

    m = coll.load_manifest()
    rows: list[dict] = []
    for rel in sorted(m.get("files", {}))[:max_files]:
        doc = coll.load_parsed(rel)
        if doc:
            rows.extend(c.to_dict() for c in chunk_document(doc, REFERENCE_CHUNKING))
    return rows


def generate_questions(rows: list[dict], types: list[str] | None = None, n_per_type: int = 20, seed: int = 0) -> list[dict]:
    """Deterministic extractive question generation (no LLM, no network).

    Prose questions are built from the rarest non-numeric words of one sentence plus the section heading; they are
    keyword-style and therefore optimistic for lexical retrievers. Tables, equations and code use structure."""
    types = [t for t in (types or CTYPES) if t in CTYPES]
    rng = random.Random(seed)
    df: Counter = Counter()
    for r in rows:
        df.update(set(content_words(r["text"])))
    out: list[dict] = []
    by_type: dict[str, list[dict]] = {t: [] for t in types}
    for r in sorted(rows, key=lambda r: r["chunk_id"]):
        if r["ctype"] in by_type:
            by_type[r["ctype"]].append(r)
    for t in types:
        cands = by_type[t][:]
        rng.shuffle(cands)
        seen: set[str] = set()
        for r in cands:
            q = _make(t, r, df)
            if not q or q["expected_answer"] in seen:
                continue
            seen.add(q["expected_answer"])
            out.append({"id": "q_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{r['chunk_id']}|{t}").hex[:8], "ctype": t, **q,
                        "expected_source": r["file"], "generator": "extractive"})
            if sum(1 for o in out if o["ctype"] == t) >= n_per_type:
                break
    return out


def _make(t: str, r: dict, df: Counter) -> dict | None:
    text, head = r["text"], _heading_last(r["heading_path"])
    where = f"In '{head}', " if head else ""
    if t == "prose":
        sents = [s for s in _SENT.split(text) if len(content_words(s)) >= 6]
        if not sents:
            return None
        s = max(sents, key=lambda x: sum(1 for w in content_words(x) if is_number(w)))
        hw = set(tokens(head))
        words = [w for w in dict.fromkeys(content_words(s)) if not is_number(w) and w not in hw and w not in STOP]
        words.sort(key=lambda w: (df.get(w, 0), w))
        terms = words[:5]
        if len(terms) < 2:
            return None
        return {"question": f"{where}what does it say about {' '.join(terms)}?", "expected_answer": s.strip()}
    if t == "table":
        lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
        if len(lines) < 3:
            return None
        cols = [c.strip() for c in lines[0].strip().strip("|").split("|")]
        data = [ln for ln in lines[1:] if not re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", ln)]
        if not data:
            return None
        row = data[len(data) // 2]
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2 or len(cols) < 2:
            return None
        return {"question": f"{where}what are the {', '.join(cols[1:])} for {cells[0]}?", "expected_answer": " | ".join(cells)}
    if t == "equation":
        m = _EQ.search(text)
        eq = m.group(1) if m else next((ln.strip() for ln in text.splitlines() if "\\" in ln), "")
        if not eq:
            return None
        first = _SENT.split(text.strip())[0]
        first = re.sub(r"\$\$.*?\$\$", "", first, flags=re.S).strip()
        words = content_words(first)[:6]
        return {"question": f"{where}what is the equation for {' '.join(words) or 'this relationship'}?", "expected_answer": eq.strip()}
    if t == "code":
        m = next((_DEF.match(ln) for ln in text.splitlines() if _DEF.match(ln)), None)
        if not m:
            return None
        sig = m.group(0).strip()
        name = re.sub(r"^(?:async\s+)?def\s+", "", sig).split("(")[0]
        doc = re.search(r'"""(.+?)"""', text, re.S)
        phrase = snippet(doc.group(1).strip(), 90) if doc else " ".join(name.split("_"))
        return {"question": f"Which function implements: {phrase} (in {r['file']})?", "expected_answer": sig}
    return None


def locate_gold(rows: list[dict], case: dict) -> list[dict]:
    ans, src = case.get("expected_answer"), case.get("expected_source")
    cand = rows
    if src:
        s = src.replace("\\", "/").lstrip("./")
        cand = [r for r in rows if r["file"] == s or r["file"].endswith("/" + s) or s.endswith("/" + r["file"])]
    if not ans:
        return cand
    return [r for r in cand if find_answer(ans, r["text"])["found"]]


def evaluate_cases(pipe: Pipeline, cases: list[dict], k: int = 8, max_tokens: int | None = None) -> list[dict]:
    res = []
    for c in cases:
        gold = locate_gold(pipe.rows, c)
        gids = {g["chunk_id"] for g in gold}
        tr = pipe.run(c["question"], k=k, max_tokens=max_tokens)
        ranks = [r for r in (Trace.rank_of(tr.reranked, g) for g in gids) if r]
        rank = min(ranks) if ranks else None
        ans = c.get("expected_answer")
        if ans:
            hit = any(find_answer(ans, p["text"])["found"] for p in tr.packed)
            at = set(tokens(" ".join(p["text"] for p in tr.packed)))
            a = [x for x in tokens(ans)]
            recall = (sum(1 for x in a if x in at) / len(a)) if a else 0.0
        else:
            hit = any(p["chunk_id"] in gids for p in tr.packed)
            recall = 1.0 if hit else 0.0
        res.append({"id": c.get("id"), "ctype": c.get("ctype", "unknown"), "file": c.get("expected_source"), "question": c["question"],
                    "gold_chunks": len(gids), "rank": rank, "hit": bool(hit), "rr": (1.0 / rank) if rank else 0.0, "context_recall": round(recall, 3)})
    return res


def summarize(results: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        return {"n": n, "hit_at_k": round(sum(r["hit"] for r in rs) / n, 3) if n else None,
                "mrr": round(sum(r["rr"] for r in rs) / n, 3) if n else None,
                "context_recall": round(sum(r["context_recall"] for r in rs) / n, 3) if n else None}

    by = {t: agg([r for r in results if r["ctype"] == t]) for t in sorted({r["ctype"] for r in results})}
    return {"overall": agg(results), "by_type": by}


def domain_eval(coll: Collection, types: list[str] | None = None, n_per_type: int = 20, k: int = 8, seed: int = 0,
                questions: list[dict] | None = None, diagnose_failures: int = 20) -> dict:
    from ..diagnose import diagnose

    pipe = Pipeline(coll)
    qs = questions or generate_questions(reference_rows(coll), types, n_per_type, seed)
    results = evaluate_cases(pipe, qs, k=k)
    summary = summarize(results)
    fails = [(q, r) for q, r in zip(qs, results) if not r["hit"]]
    codes: Counter = Counter()
    samples = []
    for q, r in fails[:diagnose_failures]:
        try:
            d = diagnose(coll, q["question"], q.get("expected_answer"), q.get("expected_source"), k=k)
            codes[d["failure_code"]] += 1
            if len(samples) < 5:
                samples.append({"ctype": r["ctype"], "file": r["file"], "question": snippet(r["question"], 140), "failure_code": d["failure_code"], "first_failing_stage": d["first_failing_stage"]})
        except Exception as e:  # noqa: BLE001
            codes["diagnosis_error"] += 1
            if len(samples) < 5:
                samples.append({"ctype": r["ctype"], "question": snippet(r["question"], 140), "error": str(e)[:120]})
    per_file: dict[str, list[bool]] = {}
    for r in results:
        per_file.setdefault(r["file"] or "?", []).append(r["hit"])
    weakest = sorted(({"file": f, "n": len(v), "miss_rate": round(1 - sum(v) / len(v), 2)} for f, v in per_file.items() if not all(v)),
                     key=lambda x: (-x["miss_rate"], -x["n"]))[:5]
    run_id = "run_" + uuid.uuid4().hex[:8]
    report = {"run_id": run_id, "created": now_iso(), "k": k, "generator": "extractive" if not questions else "provided", "n_questions": len(qs),
              **summary, "failure_breakdown": dict(codes), "weakest_documents": weakest, "failure_samples": samples,
              "caveat": "Extractive questions are keyword-style and optimistic for lexical retrieval; use provided questions or an LLM generator for harder tests."}
    coll.eval_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(coll.eval_dir / "runs" / f"{run_id}.json", {**report, "questions": qs, "results": results})
    atomic_write_json(coll.eval_dir / "questions.json", qs)
    return report


def load_report(coll: Collection, run_id: str) -> dict:
    if not re.match(r"^run_[0-9a-f]{8}$", run_id or ""):
        from ..errors import RagDocError
        raise RagDocError("Invalid run_id.", code="BAD_RUN_ID")
    p = coll.eval_dir / "runs" / f"{run_id}.json"
    from ..errors import RagDocError
    if not p.exists():
        raise RagDocError(f"Run '{run_id}' not found.", code="NO_RUN")
    d = json.loads(p.read_text("utf-8"))
    d.pop("results", None)
    d.pop("questions", None)
    return d
