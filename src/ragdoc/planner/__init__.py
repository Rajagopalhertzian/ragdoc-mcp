"""plan_rag: choose parser / chunker / embedder / retrieval from a corpus profile, then measure time and size.

The rules below are starting hypotheses taken from the product spec, to be tuned by measurement
(benchmarks/compare_baseline.py), not facts."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from ..chunkers import DEFAULT_CHUNKING, chunk_document
from ..collection import atomic_write_json, now_iso
from ..config import state_dir
from ..embedders import get_embedder, load_models, neural_available
from ..errors import RagDocError
from ..parsers import docling_available
from ..parsers.base import PDF_EXTS
from ..parsers.fast import parse_fast
from ..retrieval.pipeline import DEFAULT_PACK, DEFAULT_RETRIEVAL
from ..security import safe_path
from .plans import baseline_plan, default_plan
from .profile import PROFILES, profile_corpus, slug

__all__ = ["baseline_plan", "default_plan", "plan_rag", "load_plan", "profile_corpus", "PROFILES"]

_PLAN_ID = re.compile(r"^plan_[0-9a-f]{10}$")
LARGE_RUN_SECONDS = 120


def plans_dir() -> Path:
    return state_dir() / "plans"


def load_plan(plan_id: str) -> dict:
    if not _PLAN_ID.match(plan_id or ""):
        raise RagDocError("Invalid plan_id.", hint="Use the plan_id returned by plan_rag.", code="BAD_PLAN_ID")
    p = plans_dir() / f"{plan_id}.json"
    if not p.exists():
        raise RagDocError(f"Plan '{plan_id}' not found.", hint="Run plan_rag again.", code="NO_PLAN")
    return json.loads(p.read_text("utf-8"))


def count_pages_text(doc) -> int:
    return max(1, int(sum(len(p["text"]) for p in doc.pages) / 3000))


def _measure(src: Path, prof: dict, plan: dict, sample_pages: int = 50) -> dict:
    """Parse about 50 sampled pages and embed a sample of chunks on this machine, then extrapolate."""
    from ..build import discover_files

    files, _ = discover_files(src)
    pages = 0
    t0 = time.perf_counter()
    docs = []
    for p, rel in files:
        if pages >= sample_pages:
            break
        try:
            d = parse_fast(p, rel, "measure", max_pages=max(1, sample_pages - pages))
        except Exception:  # noqa: BLE001
            continue
        docs.append(d)
        pages += max(len(d.pages), 1) if p.suffix.lower() in PDF_EXTS else count_pages_text(d)
    parse_s = max(time.perf_counter() - t0, 1e-6)
    chunks = [c for d in docs for c in chunk_document(d, plan["chunking"])]
    sample = chunks[:200]
    try:
        emb = get_embedder(plan["embedder"]["name"])
    except RagDocError:
        emb = get_embedder("hashing-384")
    t1 = time.perf_counter()
    if sample:
        emb.embed_docs([c.embed_text for c in sample])
    embed_s = max(time.perf_counter() - t1, 1e-6)
    pps = pages / parse_s if pages else None
    cps = len(sample) / embed_s if sample else None
    avg_tok = (sum(c.tokens for c in chunks) / len(chunks)) if chunks else float(plan["chunking"]["max_tokens"])
    est_chunks = int(prof["est_total_tokens"] / max(avg_tok, 1.0)) if prof["est_total_tokens"] else len(chunks)
    est_chunks = max(est_chunks, len(chunks))
    workers = 1  # the build is single-process; do not promise parallel speedups
    parse_est = (prof["est_total_pages"] / (pps * workers)) if pps else 0.0
    embed_est = (est_chunks / (cps * workers)) if cps else 0.0
    dim = int(plan["embedder"].get("dim", 384))
    return {
        "measured": {"parse_pages_per_s": round(pps, 2) if pps else None, "embed_chunks_per_s": round(cps, 1) if cps else None,
                     "sample_pages": pages, "sample_chunks": len(sample), "avg_chunk_tokens": round(avg_tok, 1), "workers": workers},
        "est_chunks": est_chunks,
        "est_seconds": round(parse_est + embed_est, 1),
        "est_index_bytes": int(est_chunks * dim * 4 * 1.5),
        "basis": "Measured on a sample on this machine, then divided by workers=1. Real runs vary with file mix and disk speed.",
    }


def plan_rag(path: str, constraints: dict | None = None, collection: str | None = None, sample_size: int = 200, profile: dict | None = None) -> dict:
    constraints = constraints or {}
    privacy = constraints.get("privacy", "local_only")
    if privacy not in ("local_only", "cloud_ok"):
        raise RagDocError("constraints.privacy must be 'local_only' or 'cloud_ok'.", hint="Use privacy='local_only' (default) or 'cloud_ok'.", code="BAD_CONSTRAINT")
    src = safe_path(path)
    prof = profile or profile_corpus(src, sample_size=sample_size)
    if prof["files_total"] == 0:
        raise RagDocError("No supported files found.", hint="Supported: .md .txt .rst .pdf and common source-code files.", code="EMPTY_CORPUS")
    sig = set(prof["signals"])
    log: list[dict] = []
    warnings: list[str] = []
    consent: dict = {}

    def decide(choice: str, value, reason: str) -> None:
        log.append({"choice": choice, "value": value, "reason": reason})

    # parsing -----------------------------------------------------------------------------------------
    parsers = {"*": "fast"}
    if prof["by_extension"].get(".pdf"):
        hard = bool(sig & {"scanned", "tables", "equations"})
        hard_list = sorted(sig & {"scanned", "tables", "equations"})
        if hard and docling_available():
            parsers[".pdf"] = "docling"
            decide("pdf_parser", "docling", f"PDF signals {hard_list} need layout/OCR/formula parsing.")
        elif hard and privacy == "cloud_ok" and os.getenv("LLAMA_CLOUD_API_KEY"):
            parsers[".pdf"] = "llamaparse"
            consent = {"cloud_parser": True, "consent_recorded_at": now_iso()}
            decide("pdf_parser", "llamaparse", "Hard PDFs, cloud allowed and an API key is present. Consent is recorded in the manifest.")
        else:
            parsers[".pdf"] = "fast"
            if hard:
                warnings.append(f"PDFs show {hard_list} but Docling is not installed (pip install 'ragdoc-mcp[docling]'): using the fast parser, which cannot OCR scans or rebuild tables and equations.")
            decide("pdf_parser", "fast", "Digital PDFs without hard signals." if not hard else "Fallback: Docling unavailable.")
    else:
        decide("parser", "fast", "No PDFs; text, markdown and code are read directly.")

    # chunking ----------------------------------------------------------------------------------------
    chunking = dict(DEFAULT_CHUNKING)
    chunking.update({"kind": "heading", "max_tokens": 600, "overlap_pct": 12, "prepend_heading": True})
    if "tables" in sig:
        chunking.update({"keep_tables_whole": True, "table_summary": True})
        decide("tables", "one chunk per table with a column summary; header repeated per row group", "Table signal in profile.")
    if "equations" in sig:
        chunking.update({"keep_formulas_whole": True})
        decide("equations", "never split inside a formula; keep surrounding explanation", "Equation signal in profile.")
    if "code" in sig or prof["kinds"].get("code"):
        decide("code", "AST chunks by function/class (Python AST; regex split for other languages), path and imports as metadata", "Source files present.")
    decide("chunking", f"heading-aware, {chunking['max_tokens']} tokens, {chunking['overlap_pct']}% overlap, heading path prepended", "Starting rule for text/markdown; tune with domain_eval.")

    # embedder ------------------------------------------------------------------------------------------
    models = load_models()["embedders"]
    want = constraints.get("embedder", "offline")
    if want not in ("offline", "neural"):
        raise RagDocError("constraints.embedder must be 'offline' or 'neural'.", hint="Use embedder='offline' (default) or 'neural'.", code="BAD_CONSTRAINT")
    if want == "neural" and neural_available():
        # Neural models need a one-time network download, so this is opt-in (the server makes no network calls by default).
        pick = models["code"] if ("code" in sig and prof["content_mix"]["code"] >= 0.5) else models["neural_small"]
        embedder = {"name": pick["name"], "dim": pick["dim"]}
        decide("embedder", pick["name"], "You opted in to a neural embedder; the model downloads once on first use. Verify the model is still offered. If it cannot load, the build falls back to the offline embedder and says so.")
    else:
        embedder = {"name": "hashing-384", "dim": 384}
        if want == "neural":
            warnings.append("constraints.embedder='neural' needs fastembed: pip install 'ragdoc-mcp[neural]'. Using the offline embedder.")
            decide("embedder", "hashing-384", "Neural requested but fastembed is not installed.")
        else:
            decide("embedder", "hashing-384", "Default: offline and deterministic, no downloads. It is lexical-fuzzy, not semantic; BM25 compensates.")
            warnings.append("Using the offline hashing embedder (no network). For semantic recall opt in with constraints.embedder='neural'"
                            + (" (fastembed is installed)." if neural_available() else " after: pip install 'ragdoc-mcp[neural]'."))

    retrieval = {**DEFAULT_RETRIEVAL, "rerank": "lexical_overlap", "candidates": 50}
    decide("retrieval", "dense + BM25 fused with RRF (k=60), rerank top 50", "Spec default; hybrid protects exact identifiers and numbers.")

    plan = default_plan(str(src), collection or slug(src))
    plan.update({"parsers": parsers, "chunking": chunking, "embedder": embedder, "retrieval": retrieval, "pack": dict(DEFAULT_PACK),
                 "consent": consent, "constraints": constraints})
    plan["id"] = "plan_" + uuid.uuid4().hex[:10]

    est = _measure(src, prof, plan)
    tier = "lancedb" if est["est_chunks"] < 100_000 else "qdrant_or_pgvector"
    if tier != "lancedb":
        warnings.append(f"~{est['est_chunks']:,} chunks exceeds the LanceDB tier (100k). Qdrant/pgvector adapters are not included in v0.1; LanceDB is used and may be slow or large.")
    decide("store", "lancedb", f"~{est['est_chunks']:,} chunks (tier rule: <100k LanceDB).")
    cloud_cost = None
    if parsers.get(".pdf") == "llamaparse":
        per_page = load_models().get("cloud_parser_cost_per_page_usd")
        cloud_cost = {"pages": prof["est_total_pages"], "usd": None if per_page is None else round(prof["est_total_pages"] * per_page, 2),
                      "note": "No price is configured; check the provider's current pricing." if per_page is None else None}
    budget = constraints.get("budget_usd")
    if budget is not None and cloud_cost and cloud_cost["usd"] is not None and cloud_cost["usd"] > budget:
        raise RagDocError(f"Estimated cloud parsing cost ${cloud_cost['usd']} exceeds budget ${budget}.", hint="Use privacy='local_only' or raise the budget.", code="OVER_BUDGET")
    plan["estimates"] = {**est, "cloud_cost": cloud_cost, "store_tier": tier}
    plan["requires_approval"] = est["est_seconds"] > LARGE_RUN_SECONDS or bool(cloud_cost)
    plan["decision_log"] = log
    plan["warnings"] = warnings
    plan["profile_id"] = prof["profile_id"]
    plan["created"] = now_iso()
    atomic_write_json(plans_dir() / f"{plan['id']}.json", plan)
    return {"plan": plan, "profile": prof}
