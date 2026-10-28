"""verify_fix: apply a proposed fix to a shadow copy, re-run the failing cases plus a regression sample, report,
and promote only when asked and only when nothing regressed."""
from __future__ import annotations

import copy
import os
import shutil
import time
import uuid

from ..build import build_collection
from ..collection import Collection
from ..config import shadow_dir
from ..errors import RagDocError
from ..retrieval.pipeline import Pipeline
from . import evaluate_cases, generate_questions, reference_rows, summarize

CHUNK_KEYS = {"kind": str, "max_tokens": int, "overlap_pct": int, "prepend_heading": bool, "keep_tables_whole": bool, "keep_formulas_whole": bool, "table_summary": bool}
RETR_KEYS = {"dense": bool, "bm25": bool, "rerank": str, "candidates": int, "rrf_k": int, "weights": dict}
FIX_TYPES = {"rechunk", "retrieval_config", "rerank_config", "pack_config", "reparse", "reindex", "query_rewrite"}


def _clean(d: dict, allowed: dict, what: str) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if k not in allowed:
            raise RagDocError(f"Unsupported {what} key '{k}'.", hint=f"Allowed: {sorted(allowed)}", code="BAD_FIX")
        if not isinstance(v, allowed[k]) or (allowed[k] is int and isinstance(v, bool)):
            raise RagDocError(f"{what}.{k} must be {allowed[k].__name__}.", code="BAD_FIX")
        out[k] = v
    return out


def _apply_to_plan(plan: dict, fix: dict) -> set[str]:
    """Mutates plan; returns files to force-reprocess."""
    t = fix.get("type")
    if t not in FIX_TYPES:
        raise RagDocError(f"Unknown fix type {t!r}.", hint=f"Use one of {sorted(FIX_TYPES)}.", code="BAD_FIX")
    force: set[str] = set()
    if t == "rechunk":
        c = _clean(fix.get("chunking", {}), CHUNK_KEYS, "chunking")
        if "max_tokens" in c and not 8 <= c["max_tokens"] <= 8000:
            raise RagDocError("chunking.max_tokens must be 8-8000.", code="BAD_FIX")
        if "overlap_pct" in c and not 0 <= c["overlap_pct"] <= 50:
            raise RagDocError("chunking.overlap_pct must be 0-50.", code="BAD_FIX")
        if c.get("kind") not in (None, "heading", "window"):
            raise RagDocError("chunking.kind must be 'heading' or 'window'.", code="BAD_FIX")
        plan["chunking"] = {**plan["chunking"], **c}
    elif t in ("retrieval_config", "rerank_config"):
        r = _clean(fix.get("retrieval", {}), RETR_KEYS, "retrieval")
        if "candidates" in r and not 1 <= r["candidates"] <= 100:
            raise RagDocError("retrieval.candidates must be 1-100.", code="BAD_FIX")
        plan["retrieval"] = {**plan.get("retrieval", {}), **r}
    elif t == "pack_config":
        p = _clean(fix.get("pack", {}), {"max_tokens": int}, "pack")
        if not 50 <= p.get("max_tokens", 1500) <= 8000:
            raise RagDocError("pack.max_tokens must be 50-8000.", code="BAD_FIX")
        plan["pack"] = {**plan.get("pack", {}), **p}
    elif t == "reparse":
        backend = fix.get("backend")
        if backend not in ("fast", "reference", "docling", "llamaparse"):
            raise RagDocError("reparse.backend must be fast, reference, docling or llamaparse.", code="BAD_FIX")
        fb = dict(plan.get("file_backends") or {})
        for rel in fix.get("files") or []:
            fb[rel] = backend
            force.add(rel)
        plan["file_backends"] = fb
    elif t == "reindex":
        force = set(fix.get("files") or [])
    return force


def verify_fix(coll: Collection, fix: dict, cases: list[dict], regression_cases: list[dict] | None = None, k: int = 8,
               apply: bool = False, regression_sample: int = 20) -> dict:
    t0 = time.time()
    if not cases:
        raise RagDocError("Pass at least one failing case.", hint="cases=[{question, expected_answer?, expected_source?}]", code="NEED_CASES")
    cases = [{**c, "id": c.get("id") or f"case{i}", "ctype": c.get("ctype", "target")} for i, c in enumerate(cases)]
    fix = copy.deepcopy(fix)
    if fix.get("type") == "query_rewrite":
        if len(cases) != 1 or not fix.get("suggested_query"):
            raise RagDocError("query_rewrite needs exactly one case and suggested_query.", code="BAD_FIX")
    live = Pipeline(coll)
    if regression_cases is None:
        target_ans = {c.get("expected_answer") for c in cases}
        pool = [q for q in generate_questions(reference_rows(coll), None, regression_sample, seed=1) if q["expected_answer"] not in target_ans]
        regression_cases = pool
    reg = [{**c, "id": c.get("id") or f"reg{i}", "ctype": c.get("ctype", "regression")} for i, c in enumerate(regression_cases)]
    before_t = evaluate_cases(live, cases, k)
    before_r = evaluate_cases(live, reg, k)
    reg = [c for c, r in zip(reg, before_r) if r["hit"]]  # regression = things that work today
    before_r = [r for r in before_r if r["hit"]]

    sid = uuid.uuid4().hex[:8]
    shadow = Collection(shadow_dir() / f"{coll.dir.name}-{sid}")
    shadow.dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(coll.dir, shadow.dir)
    try:
        manifest = shadow.load_manifest()
        plan = manifest["plan"]
        force = _apply_to_plan(plan, fix)
        rep = build_collection(plan, collection_dir=shadow.dir, force_files=force)
        unavailable = [w for w in rep["warnings"] if "Using the fast parser instead" in w or "Falling back" in w]
        if fix.get("type") == "reparse" and unavailable:
            return {"verdict": "fix_unavailable", "applied": False, "reason": unavailable[0],
                    "next_step": "Install the missing backend (see 'requires' on the fix) and run verify_fix again.", "seconds": round(time.time() - t0, 1)}
        sp = Pipeline(shadow)
        live_files = coll.load_manifest().get("files", {})
        changed = {rel for rel, e in shadow.load_manifest().get("files", {}).items() if live_files.get(rel, {}).get("hash") != e.get("hash")}
        n_reg_before = len(reg)
        keep = [i for i, c in enumerate(reg) if (c.get("expected_source") or "") not in changed]
        reg, before_r = [reg[i] for i in keep], [before_r[i] for i in keep]  # ground truth of files edited on disk legitimately changed
        excluded = n_reg_before - len(reg)
        after_cases = [{**c, "question": fix["suggested_query"]} for c in cases] if fix.get("type") == "query_rewrite" else cases
        after_t = evaluate_cases(sp, after_cases, k)
        after_r = evaluate_cases(sp, reg, k)

        fixed = [b["id"] for b, a in zip(before_t, after_t) if not b["hit"] and a["hit"]]
        still = [b["id"] for b, a in zip(before_t, after_t) if not a["hit"]]
        regressions = [b["id"] for b, a in zip(before_r, after_r) if b["hit"] and not a["hit"]]
        mrr = lambda rs: (sum(r["rr"] for r in rs) / len(rs)) if rs else 0.0  # noqa: E731
        if fixed and not regressions:
            verdict = "improves_no_regression"
        elif fixed:
            verdict = "improves_with_regressions"
        elif regressions:
            verdict = "worse"
        else:
            verdict = "no_improvement"
        out = {
            "verdict": verdict, "fix_type": fix.get("type"),
            "target_cases": {"n": len(cases), "fixed": len(fixed), "still_failing": len(still),
                             "before": [{"id": r["id"], "rank": r["rank"], "hit": r["hit"]} for r in before_t],
                             "after": [{"id": r["id"], "rank": r["rank"], "hit": r["hit"]} for r in after_t]},
            "regression_check": {"n": len(reg), "regressions": len(regressions), "regressed_cases": regressions[:5],
                                 "mrr_before": round(mrr(before_r), 3), "mrr_after": round(mrr(after_r), 3),
                                 "excluded_source_changed_on_disk": excluded,
                                 "basis": "extractive questions generated from the parsed text with a fixed reference chunker, keeping only those that pass today"},
            "metrics_after": summarize(after_t + after_r)["overall"],
            "applied": False, "rebuild": {"files_reprocessed": rep["files_indexed"], "files_reused": rep["files_unchanged"], "seconds": rep["seconds"]},
        }
        if apply:
            if verdict != "improves_no_regression":
                out["apply_refused"] = f"Not applied: verdict is '{verdict}'. Only fixes that improve the target cases with no regressions are promoted."
            elif fix.get("type") == "query_rewrite":
                out["apply_refused"] = "A query rewrite is advice for the caller; there is nothing to apply to the collection."
            else:
                backup = shadow_dir() / f"{coll.dir.name}-backup-{sid}"
                os.replace(coll.dir, backup)
                try:
                    os.replace(shadow.dir, coll.dir)
                except Exception:
                    os.replace(backup, coll.dir)
                    raise
                out["applied"] = True
                out["backup"] = str(backup)
                out["rollback"] = f"To undo: replace the collection directory with the backup at {backup}."
        out["next_step"] = ("Fix promoted. Re-run the original query to confirm." if out["applied"] else
                            ("Call verify_fix again with apply=true to promote this fix." if verdict == "improves_no_regression" and not apply
                             else "Try the next fix from diagnose_failure."))
        out["seconds"] = round(time.time() - t0, 1)
        return out
    finally:
        if shadow.dir.exists():
            shutil.rmtree(shadow.dir, ignore_errors=True)
