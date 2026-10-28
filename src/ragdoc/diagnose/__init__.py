"""diagnose_failure: trace a failing question through every stage and name the first one that lost the answer.

Stages: 1 parse, 2 chunk, 3 index, 4 retrieve, 5 rerank, 6 pack, 7 downstream.
Rules: never blame a stage without evidence; digits and units are compared exactly; confidence is lower when the
expected answer is only approximately known. Everything is computed from the parsed text, the stored chunks,
the vector index and a second parser backend. The fault-injection hooks used for fixtures are never read here."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..collection import Collection
from ..config import base_dir
from ..errors import RagDocError
from ..parsers import ParsedDoc, docling_available, parse_file
from ..retrieval.pipeline import Pipeline, Trace
from ..security import is_within, scan_injection
from ..textutil import content_words, find_answer, is_number, sha256_file, snippet, tokens

STAGES = ["parse", "chunk", "index", "retrieve", "rerank", "pack", "downstream"]
IMPACT = {"high": 0, "medium": 1, "low": 2}
_MATHY = re.compile(r"[\\=+\-*/^_{}∫∑√≤≥≈]|\d")


def _norm_rel(s: str) -> str:
    return s.replace("\\", "/").lstrip("./").strip("/")


def _match_source(files: list[str], source: str) -> list[str]:
    s = _norm_rel(source)
    return [f for f in files if f == s or f.endswith("/" + s) or s.endswith("/" + f)]


def _disk_path(manifest: dict, rel: str) -> Path | None:
    src = Path(manifest["source_path"])
    p = src if src.is_file() else src / rel
    return p if p.exists() and is_within(p, base_dir()) else None


def _context(text: str, answer: str, width: int = 260) -> str:
    low = text.lower()
    for w in tokens(answer):
        i = low.find(w)
        if i >= 0:
            return snippet(text[max(0, i - 60): i + width], width)
    return snippet(text, width)


def _answer_line_is_tabular(raw: str, answer: str) -> bool:
    """Does the line of the source that best matches the answer look like a table row?"""
    want = set(tokens(answer))
    if not want:
        return False
    best, best_cov = "", 0.0
    for ln in raw.splitlines():
        cov = len(want & set(tokens(ln))) / len(want)
        if cov > best_cov:
            best, best_cov = ln, cov
    return best_cov >= 0.5 and (best.count("|") >= 2 or bool(re.search(r"\S(?:\t| {2,})\S.*(?:\t| {2,})\S", best)))


def _file_changed(manifest: dict, rel: str, p: Path) -> bool:
    e = manifest["files"][rel]
    st = p.stat()
    if st.st_size == e.get("size") and abs(st.st_mtime - e.get("mtime", 0)) < 1e-6:
        return False
    return sha256_file(p) != e.get("hash")


def _fix(fid: str, ftype: str, impact: str, why: str, **kw) -> dict:
    return {"id": fid, "type": ftype, "expected_impact": impact, "why": why, **kw}


def diagnose(coll: Collection, question: str, expected_answer: str | None = None, expected_source: str | None = None,
             k: int = 8, max_tokens: int | None = None) -> dict:
    if not expected_answer and not expected_source:
        raise RagDocError(
            "diagnose_failure needs expected_answer or expected_source.",
            hint="Pass a short text that must appear in the right answer (best), or the file that holds it.", code="NEED_EXPECTED")
    pipe = Pipeline(coll)
    manifest, rows, cfg = pipe.manifest, pipe.rows, pipe.retrieval
    files = list(manifest.get("files", {}))
    stages: list[dict] = []
    notes: list[str] = []
    approx_only = not expected_answer

    def done(stage: str, code: str, conf: float, evidence: dict, fixes: list[dict], sub: str | None = None, extra_notes: list[str] | None = None) -> dict:
        for s in STAGES[len(stages):]:
            stages.append({"stage": s, "status": "not_reached"})
        fixes = sorted(fixes, key=lambda f: IMPACT[f["expected_impact"]])
        if approx_only:
            conf = min(conf, 0.5)
        flags = scan_injection(json.dumps(evidence, default=str, ensure_ascii=False))  # nested snippets included
        out = {
            "first_failing_stage": stage, "failure_code": code, "subcause": sub, "confidence": round(conf, 2),
            "confidence_note": ("Only expected_source was given, so the gold chunk is approximate; confidence is capped at 0.5."
                                if approx_only else ("Answer located by fuzzy match, not exact text." if conf < 0.8 else "Answer located by exact text match.")),
            "evidence": evidence, "stages": stages, "fixes": fixes, "notes": (notes + (extra_notes or []))[:6],
            "untrusted_evidence": True,
        }
        if flags:
            out["instruction_like_text_flags"] = flags
        return out

    # ---- source resolution --------------------------------------------------------------
    if expected_source:
        cands = _match_source(files, expected_source)
        if not cands:
            on_disk = None
            src = Path(manifest["source_path"])
            if src.is_dir() and (src / _norm_rel(expected_source)).exists() and is_within(src / _norm_rel(expected_source), base_dir()):
                on_disk = expected_source
            if on_disk:
                stages.extend([{"stage": "parse", "status": "pass"}, {"stage": "chunk", "status": "pass"}])
                stages.append({"stage": "index", "status": "fail"})
                return done("index", "INDEX_MISSING", 0.9, {"file": expected_source, "detail": "The file exists on disk but was never indexed (not in the manifest)."},
                            [_fix("fix_reindex", "reindex", "high", "Index the missing file.", files=[expected_source], force=True)], sub="file_not_indexed")
            raise RagDocError(f"expected_source '{expected_source}' is not in collection or on disk.", hint="Use a path relative to the corpus root.", code="BAD_SOURCE")
    else:
        cands = files

    # ---- staleness (checked first: a stale parse cache would otherwise look like parse loss) -----
    stale: list[str] = []
    for rel in cands:
        p = _disk_path(manifest, rel)
        if p is not None and rel in manifest["files"] and _file_changed(manifest, rel, p):
            stale.append(rel)
    if stale:
        for rel in stale:
            p = _disk_path(manifest, rel)
            cur = parse_file(p, rel, sha256_file(p), "fast")
            hit = any(find_answer(expected_answer, pg["text"])["found"] for pg in cur.pages) if expected_answer else True
            old = coll.load_parsed(rel)
            old_hit = any(find_answer(expected_answer, pg["text"])["found"] for pg in old.pages) if (old and expected_answer) else False
            if hit and not old_hit:
                stages.extend([{"stage": "parse", "status": "skipped"}, {"stage": "chunk", "status": "skipped"}, {"stage": "index", "status": "fail"}])
                ev = {"file": rel, "indexed_hash": manifest["files"][rel]["hash"][:12], "current_hash": sha256_file(p)[:12],
                      "detail": "The file changed on disk after it was indexed; the answer is in the current file but not in the index."}
                if expected_answer:
                    ev["current_text"] = _context(cur.full_text(), expected_answer)
                return done("index", "INDEX_STALE", 0.92 if expected_answer else 0.6, ev,
                            [_fix("fix_reindex", "reindex", "high", "Re-index the changed file(s) incrementally.", files=stale)])

    # ---- stage 1: parse -----------------------------------------------------------------------
    parse_hits: list[tuple[str, int, dict]] = []
    parsed_docs: dict[str, ParsedDoc] = {}
    if expected_answer:
        for rel in cands:
            d = coll.load_parsed(rel)
            if not d:
                continue
            parsed_docs[rel] = d
            for pg in d.pages:
                r = find_answer(expected_answer, pg["text"])
                if r["found"]:
                    parse_hits.append((rel, pg["num"], r))
                    break
        if not parse_hits:
            return _parse_failure(coll, manifest, cands, parsed_docs, expected_answer, stages, done)
    else:
        blank = [rel for rel in cands if (d := coll.load_parsed(rel)) and not d.full_text().strip()]
        if blank and len(blank) == len(cands):
            stages.append({"stage": "parse", "status": "fail"})
            scanned = any(manifest["files"][r].get("scanned_pages") for r in blank)
            return done("parse", "PARSE_LOSS", 0.7, {"files": blank[:5], "detail": "The parser extracted no text from the expected source."},
                        [_fix("fix_reparse", "reparse", "high", "Re-parse with OCR/layout analysis.", files=blank[:5], backend="docling" if docling_available() else "reference")],
                        sub="scanned_no_ocr" if scanned else "no_text")
    stages.append({"stage": "parse", "status": "pass"})
    conf_scale = 1.0 if (not parse_hits or parse_hits[0][2]["mode"] == "exact") else 0.8

    # ---- stage 2: chunk --------------------------------------------------------------------------
    hit_files = [h[0] for h in parse_hits] or cands
    if expected_answer:
        gold = [r for r in rows if r["file"] in hit_files and find_answer(expected_answer, r["text"])["found"]]
    else:
        gold = [r for r in rows if r["file"] in cands]
    if not gold:
        return _chunk_failure(rows, hit_files, expected_answer, cfg, pipe, stages, done, parse_hits)
    stages.append({"stage": "chunk", "status": "pass"})
    gold_ids = [g["chunk_id"] for g in gold]
    best = min(gold, key=lambda g: g["tokens"])
    ans_tokens = len(tokens(expected_answer)) if expected_answer else 0
    share = ans_tokens / max(best["tokens"], 1) if ans_tokens else 1.0
    dilution_risk = bool(ans_tokens) and share < 0.05 and best["tokens"] >= 300
    qw = content_words(question)
    emb_tokens = set(tokens(best["embed_text"]))
    missing = [w for w in qw if w not in emb_tokens]
    from_heading = [w for w in missing if w in set(tokens(best["heading_path"]))]
    context_risk = bool(from_heading) and (len(from_heading) >= 2 or any(any(ch.isdigit() for ch in w) for w in from_heading))

    # ---- stage 3: index ----------------------------------------------------------------------------
    present = [cid for cid in gold_ids if pipe.store.has_id(cid)]
    index_gap = not present
    index_gap_result = lambda: done(  # noqa: E731
        "index", "INDEX_MISSING", 0.9 * conf_scale,
        {"chunk_ids": gold_ids[:3], "file": best["file"],
         "detail": "The chunk holding the answer is in chunks.parquet but absent from the vector index, and retrieval failed."},
        [_fix("fix_reindex", "reindex", "high", "Rebuild the index rows for this file.", files=sorted({g["file"] for g in gold}), force=True)])
    stages.append({"stage": "index", "status": "warn" if index_gap else "pass"})

    # ---- stages 4-6: run the real pipeline and read its trace ---------------------------------------------
    tr = pipe.run(question, k=k, max_tokens=max_tokens, diagnostic=True)
    gset = set(gold_ids)

    def rank(lst: list) -> int | None:
        rs = [r for r in (Trace.rank_of(lst, g) for g in gset) if r]
        return min(rs) if rs else None

    r_dense, r_bm25, r_hyb, r_rr = rank(tr.dense), rank(tr.bm25), rank(tr.hybrid), rank(tr.reranked)
    r_final = rank(tr.final)
    rerank_on = cfg["rerank"] not in ("off", "none", "", None)
    cand = int(cfg["candidates"])
    table = {"dense_only": r_dense, "bm25_only": r_bm25, "hybrid": r_hyb, "after_rerank": r_rr if rerank_on else None, "k": k,
             "retrievers_enabled": {"dense": bool(cfg["dense"]), "bm25": bool(cfg["bm25"])}, "rerank": cfg["rerank"], "candidates": cand}
    top_instead = [{"rank": i + 1, "file": pipe.by_id[cid]["file"], "heading": pipe.by_id[cid]["heading_path"], "text": snippet(pipe.by_id[cid]["text"], 120)}
                   for i, cid in enumerate(tr.final[:3]) if cid not in gset]
    ev_rank = {"gold_chunk": {"id": best["chunk_id"], "file": best["file"], "heading": best["heading_path"], "tokens": best["tokens"], "text": snippet(best["text"], 200)},
               "ranks": table, "retrieved_instead": top_instead}

    fail: tuple[str, str] | None = None
    if r_hyb is None:
        fail = ("retrieve", "RETRIEVAL_MISS")
    elif rerank_on:
        if r_hyb > cand:
            fail = ("retrieve", "RANK_LOW")
        elif r_rr is None or r_rr > k:
            fail = ("rerank", "RERANK_DEMOTED") if r_hyb <= k else ("retrieve", "RANK_LOW")
    elif r_hyb > k:
        fail = ("retrieve", "RANK_LOW")

    if fail and index_gap:
        # The vector index lacks the gold chunk and retrieval really failed, so the index gap is the cause.
        stages[2]["status"] = "fail"
        stages.append({"stage": "retrieve", "status": "symptom"})
        return index_gap_result()
    if fail:
        stage, code = fail
        # Chunk-stage causes are blamed only when retrieval actually failed (a big or heading-less chunk that still ranks well is not a failure).
        # Missing context also outranks a rerank blame: no reranker can order chunks that lack the identifying terms.
        if (code in ("RETRIEVAL_MISS", "RANK_LOW") and (context_risk or dilution_risk)) or (code == "RERANK_DEMOTED" and context_risk):
            return _deferred_chunk(best, stages, done, context_risk, dilution_risk, from_heading, share, ev_rank, cfg, conf_scale, r_dense, r_bm25, r_hyb)
        return _retrieval_failure(stage, code, ev_rank, cfg, k, r_dense, r_bm25, r_hyb, r_rr, rerank_on, cand, question, best, stages, done, conf_scale, share)

    stages.extend([{"stage": "retrieve", "status": "pass"}, {"stage": "rerank", "status": "pass" if rerank_on else "skipped"}])
    in_pack = [c for c in tr.packed if c["chunk_id"] in gset]
    cut = [c for c in in_pack if c["truncated"] and expected_answer and not find_answer(expected_answer, c["text"])["found"]]
    if not in_pack or (cut and len(cut) == len(in_pack)):
        stages.append({"stage": "pack", "status": "fail"})
        used, need = 0, 0
        for cid in tr.final:
            used += int(pipe.by_id[cid]["tokens"])
            if cid in gset:
                need = used
                break
        budget = tr.max_tokens
        ev = {**ev_rank, "budget_tokens": budget, "tokens_needed_to_include_gold": need, "final_rank": r_final, "pack_notes": tr.notes[:3]}
        fixes = [_fix("fix_budget", "pack_config", "high", "Raise the context token budget so the gold chunk survives.", pack={"max_tokens": min(int(need * 1.1) + 10, 8000)}),
                 _fix("fix_shorter_chunks", "rechunk", "medium", "Use shorter chunks so more of them fit the budget.", chunking={"max_tokens": max(150, int(best['tokens'] * 0.6))})]
        return done("pack", "BUDGET_TRUNCATED", 0.9 * conf_scale, ev, fixes)
    stages.extend([{"stage": "pack", "status": "pass"}])

    # ---- stage 7: downstream ---------------------------------------------------------------------------------
    in_ctx = any(find_answer(expected_answer, c["text"])["found"] for c in tr.packed) if expected_answer else True
    stages.append({"stage": "downstream", "status": "fail" if in_ctx else "pass"})
    ev = {**ev_rank, "final_rank": r_final, "answer_in_final_context": in_ctx}
    extra = ["The answer text is in the context this pipeline returns. This is not a retrieval problem: check the prompt, the model, or how the answer is checked."]
    if index_gap:
        extra.append("Warning: the gold chunk is missing from the vector index (dense retrieval cannot see it) but BM25 still recovers it here. Re-index soon.")
    return done("downstream", "GENERATION_SIDE", 0.85 * conf_scale, ev, [], extra_notes=extra)


def _parse_failure(coll, manifest, cands, parsed_docs, answer, stages, done) -> dict:
    stages.append({"stage": "parse", "status": "fail"})
    ref_hits, scattered = [], []
    check = cands if len(cands) <= 200 else cands[:200]
    for rel in check:
        p = _disk_path(manifest, rel)
        if p is None:
            continue
        ref = parse_file(p, rel, manifest["files"][rel]["hash"], "reference")
        for pg in ref.pages:
            r = find_answer(answer, pg["text"])
            if r["found"]:
                ref_hits.append((rel, pg["num"], pg["text"], r))
                break
        old = parsed_docs.get(rel)
        if old:
            for pg in old.pages:
                r = find_answer(answer, pg["text"])
                if r["mode"] == "scattered":
                    scattered.append((rel, pg["num"], pg["text"]))
                    break
    answer_mathy = len(_MATHY.findall(answer)) >= max(3, len(answer) // 6) and not scattered
    if ref_hits:
        rel, pgn, rtxt, r = ref_hits[0]
        old = parsed_docs.get(rel)
        old_text = old.full_text() if old else ""
        scanned = manifest["files"][rel].get("scanned_pages") or []
        a_tok = tokens(answer)
        if not old_text.strip() or (scanned and pgn in scanned):
            sub, why = "scanned_no_ocr", "The pipeline parse is empty for a scanned page; text exists only in images or the second backend."
        elif "(cid:" in old_text or "\ufffd" in old_text or (answer_mathy and not find_answer(answer, old_text)["found"] and any(is_number(x) for x in a_tok)):
            sub, why = "equation_garbled", "Equation characters were garbled in the pipeline parse but are intact in the second backend."
        elif scattered and ("|" in answer or _answer_line_is_tabular(rtxt, answer)):
            sub, why = "table_flattened", "All answer tokens are present but not contiguous, and the source shows a table here: row/column structure was lost."
        elif scattered:
            sub, why = "column_order", "All answer tokens are present but in the wrong order: likely multi-column reading order."
        else:
            sub, why = "unknown", "The second backend has the answer and the pipeline parse does not."
        ev = {"file": rel, "page": pgn, "second_backend_text": _context(rtxt, answer), "pipeline_parse_text": _context(old_text, answer) if old_text else "(empty)",
              "second_backend": "reference (alternate extraction)", "detail": why}
        be = "docling" if docling_available() else "reference"
        fixes = [_fix("fix_reparse", "reparse", "high", f"Re-parse '{rel}' with {be}.", files=[rel], backend=be)]
        if sub == "scanned_no_ocr" and not docling_available():
            fixes[0]["requires"] = "pip install 'ragdoc-mcp[docling]' (OCR). Without it only a layout re-parse is possible."
        return done("parse", "PARSE_LOSS", 0.9 if sub != "unknown" else 0.7, ev, fixes, sub=sub)
    if scattered:
        rel, pgn, txt = scattered[0]
        sub = "table_flattened" if "|" in answer else "column_order"
        return done("parse", "PARSE_LOSS", 0.6, {"file": rel, "page": pgn, "pipeline_parse_text": _context(txt, answer),
                    "detail": "All answer tokens are present but not contiguous. The second backend could not confirm (it has no better extraction for this file type)."},
                    [_fix("fix_reparse", "reparse", "high", f"Re-parse '{rel}' with a layout-aware parser.", files=[rel], backend="docling" if docling_available() else "reference")], sub=sub)
    scanned_any = [r for r in cands if manifest["files"][r].get("scanned_pages")]
    ev = {"detail": "Neither the pipeline parse nor the second backend contains the expected answer text.",
          "checked_files": len(check), "scanned_files": scanned_any[:3]}
    sub = "scanned_no_ocr" if scanned_any else "unknown"
    return done("parse", "PARSE_LOSS", 0.45, ev,
                [_fix("fix_reparse", "reparse", "medium", "Try OCR/layout analysis on the scanned files.", files=scanned_any[:5] or check[:1], backend="docling" if docling_available() else "reference"),
                 _fix("fix_check_expected", "advice", "low", "Check that expected_answer matches the document wording (digits and units must match exactly).")],
                sub=sub, extra_notes=["Low confidence: the answer text could not be found in any parse. It may be an image-only page, or expected_answer may not match the document wording."])


def _chunk_failure(rows, hit_files, answer, cfg, pipe, stages, done, parse_hits) -> dict:
    stages.append({"stage": "chunk", "status": "fail"})
    by_file: dict[str, list[dict]] = {}
    for r in rows:
        if r["file"] in hit_files:
            by_file.setdefault(r["file"], []).append(r)
    ch = pipe.plan.get("chunking", {})
    cur = int(ch.get("max_tokens", 400))
    for rel, rs in by_file.items():
        rs = sorted(rs, key=lambda r: r["seq"])
        for a, b in zip(rs, rs[1:]):
            if find_answer(answer, a["text"] + " " + b["text"])["found"]:
                ev = {"file": rel, "chunk_a": {"id": a["chunk_id"], "tokens": a["tokens"], "ends_with": snippet(a["text"][-120:], 120)},
                      "chunk_b": {"id": b["chunk_id"], "tokens": b["tokens"], "starts_with": snippet(b["text"][:120], 120)},
                      "configured": {"kind": ch.get("kind"), "max_tokens": cur, "overlap_pct": ch.get("overlap_pct")},
                      "detail": "The answer text is split across two adjacent chunks; no single chunk contains it."}
                fixes = [_fix("fix_rechunk_heading", "rechunk", "high", "Use heading-aware chunks that keep tables and formulas whole.",
                              chunking={"kind": "heading", "max_tokens": max(cur, 400), "overlap_pct": max(int(ch.get("overlap_pct", 0)), 10), "keep_tables_whole": True, "keep_formulas_whole": True}),
                         _fix("fix_more_overlap", "rechunk", "medium", "Increase chunk size and overlap so the span stays intact.",
                              chunking={"max_tokens": cur * 2, "overlap_pct": max(int(ch.get("overlap_pct", 0)), 20)})]
                return done("chunk", "CHUNK_SPLIT", 0.9, ev, fixes)
    ev = {"files": list(by_file)[:3], "detail": "The answer is in the parsed text but no chunk contains it intact (and it is not a simple two-chunk split)."}
    return done("chunk", "CHUNK_SPLIT", 0.55, ev,
                [_fix("fix_rechunk_heading", "rechunk", "high", "Re-chunk with heading-aware chunking; keep tables and formulas whole.",
                      chunking={"kind": "heading", "max_tokens": max(cur, 400), "overlap_pct": 10, "keep_tables_whole": True, "keep_formulas_whole": True})])


def _deferred_chunk(best, stages, done, context_risk, dilution_risk, from_heading, share, ev_rank, cfg, conf_scale, r_dense, r_bm25, r_hyb) -> dict:
    stages[1]["status"] = "fail"  # chunk stage: blamed because retrieval failed and the chunk shape explains it
    stages.append({"stage": "retrieve", "status": "symptom"})
    if context_risk:
        ev = {**ev_rank, "detail": "Retrieval failed and the question's key terms appear only in the heading path, which is not part of the indexed chunk text.",
              "terms_only_in_heading": from_heading[:6], "heading_path": best["heading_path"]}
        fixes = [_fix("fix_prepend_heading", "rechunk", "high", "Prepend the heading path to each chunk before embedding and BM25.", chunking={"prepend_heading": True})]
        return done("chunk", "CONTEXT_MISSING", 0.7 * conf_scale, ev, fixes)
    ev = {**ev_rank, "detail": f"Retrieval failed and the answer is only {share:.1%} of a {best['tokens']}-token chunk, so it is diluted by surrounding text.",
          "answer_share_of_chunk": round(share, 4)}
    fixes = [_fix("fix_smaller_chunks", "rechunk", "high", "Use smaller, heading-aware chunks.",
                  chunking={"kind": "heading", "max_tokens": max(150, min(400, best["tokens"] // 3)), "overlap_pct": 10})]
    return done("chunk", "CHUNK_DILUTED", 0.65 * conf_scale, ev, fixes)


def _retrieval_failure(stage, code, ev_rank, cfg, k, r_dense, r_bm25, r_hyb, r_rr, rerank_on, cand, question, best, stages, done, conf_scale, share=1.0) -> dict:
    if stage == "retrieve":
        stages.append({"stage": "retrieve", "status": "fail"})
    else:
        stages.extend([{"stage": "retrieve", "status": "pass"}, {"stage": "rerank", "status": "fail"}])
    fixes: list[dict] = []
    ev = dict(ev_rank)
    if code == "RERANK_DEMOTED":
        ev["detail"] = f"The chunk ranked #{r_hyb} before reranking (inside the top {k}) and #{r_rr} after."
        fixes = [_fix("fix_rerank_off", "rerank_config", "high", "Turn reranking off, or swap the reranker, for this collection.", retrieval={"rerank": "off"}),
                 _fix("fix_rerank_wider", "rerank_config", "medium", "Rerank more candidates.", retrieval={"candidates": min(cand * 2, 100)})]
        if share < 0.05:
            fixes.append(_fix("fix_smaller_chunks", "rechunk", "medium",
                              f"The answer is only {share:.1%} of this {best['tokens']}-token chunk; rerankers often under-score diluted chunks, so smaller chunks may fix the cause.",
                              chunking={"kind": "heading", "max_tokens": max(150, min(400, best["tokens"] // 2)), "overlap_pct": 10}))
        return done("rerank", code, 0.9 * conf_scale, ev, fixes)
    only = []
    if r_bm25 and not cfg["bm25"]:
        only.append("BM25 (disabled in this collection)")
    if r_dense and not cfg["dense"]:
        only.append("dense (disabled in this collection)")
    better = [(n, r) for n, r in (("dense", r_dense), ("bm25", r_bm25)) if r and r <= k and (r_hyb is None or r < r_hyb)]
    if code == "RETRIEVAL_MISS":
        ev["detail"] = "The chunk is not in the top 100 of the configured retrieval." + (f" It would be found by {', '.join(only)}." if only else "")
    else:
        ev["detail"] = (f"The chunk ranks #{r_hyb} in the hybrid list (cutoff for reranking is {cand}, k is {k})." if rerank_on
                        else f"The chunk ranks #{r_hyb}, below k={k}.")
    if r_bm25 and not cfg["bm25"]:
        fixes.append(_fix("fix_enable_bm25", "retrieval_config", "high", f"Enable BM25: it ranks the chunk #{r_bm25}.", retrieval={"bm25": True}))
    if r_dense and not cfg["dense"]:
        fixes.append(_fix("fix_enable_dense", "retrieval_config", "high", f"Enable dense retrieval: it ranks the chunk #{r_dense}.", retrieval={"dense": True}))
    if better and not only:
        n, r = min(better, key=lambda x: x[1])
        w = {"dense": 1.0, "bm25": 1.0}
        w[n] = 3.0
        fixes.append(_fix("fix_weights", "retrieval_config", "medium", f"{n} alone ranks it #{r}; weight {n} higher in the fusion.", retrieval={"weights": w}))
    if code == "RANK_LOW" and r_hyb and r_hyb <= 100:
        fixes.append(_fix("fix_wider", "retrieval_config", "medium", f"Rerank a wider candidate pool and return more results (the chunk is at rank {r_hyb}).", retrieval={"candidates": min(max(r_hyb + 10, cand), 100)}, k=min(r_hyb + 2, 100)))
    heading_terms = [w for w in tokens(best["heading_path"]) if w not in set(tokens(question)) and len(w) > 2][:6]
    fixes.append(_fix("fix_query_rewrite", "query_rewrite", "medium",
                      "Rewrite the query using document vocabulary" + (f" (e.g. add: {' '.join(heading_terms)})" if heading_terms else "") + ".",
                      suggested_query=(question + " " + " ".join(heading_terms)).strip(), verifiable=True))
    fixes.append(_fix("fix_embedder", "advice", "low", "Try a different embedder (ragdoc-mcp[neural]) if semantic matches are the issue."))
    return done("retrieve", code, 0.9 * conf_scale, ev, fixes)
