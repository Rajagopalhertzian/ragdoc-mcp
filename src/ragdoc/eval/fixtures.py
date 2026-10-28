"""Failure fixtures: collections with a KNOWN injected fault plus questions that fail because of it.

Labeling rules (so the true failing stage is known, not guessed):
  * Fault families (parse, chunk, rerank, pack): a case is kept only if the intact pipeline answers it
    ("good passes") and the faulted pipeline does not ("faulty fails"). The injected fault is therefore the cause.
  * INDEX_STALE: the file is edited after indexing; the question asks about the new text.
  * INDEX_MISSING: new files are added after indexing and never indexed.
  * RETRIEVAL_MISS / RANK_LOW: upstream stages are intact; the label comes from MEASURING the gold chunk's rank in the
    healthy pipeline (this is partly circular with the diagnosis rule, by design: these two codes mainly guard against
    upstream faults being mis-blamed on retrieval).
  * GENERATION_SIDE controls: the healthy pipeline already puts the answer in context. Any other blame is a false blame.
This is SYNTHETIC data. Numbers measured on it are a regression guard, not a claim about real documents."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..build import build_collection
from ..collection import Collection
from ..config import base_dir
from ..planner.plans import default_plan
from ..retrieval.pipeline import Pipeline, Trace
from . import evaluate_cases, locate_gold
from .corpus import case_for, make_corpus

CODE_STAGE = {
    "PARSE_LOSS": "parse", "CHUNK_SPLIT": "chunk", "CHUNK_DILUTED": "chunk", "CONTEXT_MISSING": "chunk",
    "INDEX_STALE": "index", "INDEX_MISSING": "index", "RETRIEVAL_MISS": "retrieve", "RANK_LOW": "retrieve",
    "RERANK_DEMOTED": "rerank", "BUDGET_TRUNCATED": "pack", "GENERATION_SIDE": "downstream",
}
PARSE_MODES = {
    "blank_pages": ("scanned_no_ocr", ("prose", "table", "equation")),
    "scramble_columns": ("column_order", ("prose",)),
    "flatten_tables": ("table_flattened", ("table",)),
    "garble_equations": ("equation_garbled", ("equation",)),
}
PARAPHRASE = {
    "prose": ["How hot can the unit get while it is running?", "What heat level is tolerated during service?",
              "Up to which warmth may the device operate?"],
}


def _passes(pipe: Pipeline, case: dict, k: int = 8) -> bool:
    return evaluate_cases(pipe, [case], k)[0]["hit"]


def _best_rank(ranked: list, gold: set[str]) -> int | None:
    rs = [r for r in (Trace.rank_of(ranked, g) for g in gold) if r]
    return min(rs) if rs else None


def _number(cases: list[dict]) -> list[dict]:
    for i, c in enumerate(cases):
        c["id"] = f"{c['family']}-{i:02d}"
    return cases


def _build(name: str, src: Path, **ov) -> Collection:
    plan = default_plan(str(src), name, **ov)
    build_collection(plan)
    return Collection.open(name)


def _select(cands: list[dict], good: Pipeline, bad: Pipeline, n: int) -> list[dict]:
    out = []
    for c in cands:
        if len(out) >= n:
            break
        if _passes(good, c) and not _passes(bad, c):
            out.append(c)
    return out


def build_suite(seed: int = 7, n_docs: int = 60, n_per_code: int = 8) -> list[dict]:
    """Build all fixture collections under RAGDOC_BASE_DIR and return the labeled cases."""
    base = base_dir()
    rng = random.Random(seed)
    corpus = base / "fixture_corpus"
    facts = make_corpus(corpus, seed=seed, n_docs=n_docs)
    good_coll = _build("fx_good", corpus)
    good = Pipeline(good_coll)
    cases: list[dict] = []
    ctypes = ["prose", "table", "equation", "code"]

    def pool(types=ctypes, shuffle=True) -> list[dict]:
        order = list(range(n_docs))
        if shuffle:
            rng.shuffle(order)
        return [case_for(facts[i], types[j % len(types)]) for j, i in enumerate(order)]

    # ---- parse faults (4 sub-causes) ------------------------------------------------------------------
    per_sub = max(2, n_per_code // 2)
    for mode, (sub, types) in PARSE_MODES.items():
        cands = [c for c in pool(list(types)) if _passes(good, c)][: per_sub * 3]
        files = sorted({c["expected_source"] for c in cands})
        coll = _build(f"fx_parse_{mode}", corpus, _faults={"parse": {"mode": mode, "files": files}})
        bad = Pipeline(coll)
        for c in _select(cands, good, bad, per_sub):
            cases.append({**c, "collection": coll.dir.name, "true_code": "PARSE_LOSS", "true_stage": "parse", "true_subcause": sub, "family": f"parse_{mode}"})

    # ---- chunk split ----------------------------------------------------------------------------------
    coll = _build("fx_split", corpus, chunking={"kind": "window", "max_tokens": 6, "overlap_pct": 0})
    for c in _select(pool(["prose", "table", "equation"]), good, Pipeline(coll), n_per_code):
        cases.append({**c, "collection": "fx_split", "true_code": "CHUNK_SPLIT", "true_stage": "chunk", "true_subcause": None, "family": "chunk_split"})

    # ---- context missing (heading path not embedded) -----------------------------------------------------
    coll = _build("fx_ctx", corpus, chunking={"prepend_heading": False})
    for c in _select(pool(["prose", "equation"]), good, Pipeline(coll), n_per_code):
        cases.append({**c, "collection": "fx_ctx", "true_code": "CONTEXT_MISSING", "true_stage": "chunk", "true_subcause": None, "family": "context_missing"})

    # ---- chunk diluted: one huge chunk per file + anchorless questions ---------------------------------------
    coll = _build("fx_dilute", corpus, chunking={"kind": "window", "max_tokens": 4000, "overlap_pct": 0})
    anchorless = [{"ctype": "prose", "question": rng.choice(PARAPHRASE["prose"]), "expected_answer": f"maximum operating temperature of {f['t_max']} degrees Celsius",
                   "expected_source": f["md"]} for f in facts]
    for c in _select(anchorless, good, Pipeline(coll), n_per_code):
        cases.append({**c, "collection": "fx_dilute", "true_code": "CHUNK_DILUTED", "true_stage": "chunk", "true_subcause": None, "family": "chunk_diluted"})

    # ---- rerank demoted -------------------------------------------------------------------------------------
    coll = _build("fx_rerank", corpus, retrieval={"rerank": "_reverse"})
    for c in _select(pool(), good, Pipeline(coll), n_per_code):
        cases.append({**c, "collection": "fx_rerank", "true_code": "RERANK_DEMOTED", "true_stage": "rerank", "true_subcause": None, "family": "rerank_demoted"})

    # ---- budget truncated ---------------------------------------------------------------------------------------
    coll = _build("fx_budget", corpus, pack={"max_tokens": 45})
    for c in _select(pool(), good, Pipeline(coll), n_per_code):
        cases.append({**c, "collection": "fx_budget", "true_code": "BUDGET_TRUNCATED", "true_stage": "pack", "true_subcause": None, "family": "budget_truncated"})

    # ---- index stale: edit files after indexing ---------------------------------------------------------------------
    stale_dir = base / "fixture_corpus_stale"
    sfacts = make_corpus(stale_dir, seed=seed, n_docs=max(n_per_code, 8))
    _build("fx_stale", stale_dir)
    for f in sfacts[:n_per_code]:
        p = stale_dir / f["md"]
        new_t = f["t_max"] + 7
        p.write_text(p.read_text("utf-8").replace(f"maximum operating temperature of {f['t_max']} degrees", f"maximum operating temperature of {new_t} degrees"), "utf-8")
        c = case_for({**f, "t_max": new_t}, "prose")
        cases.append({**c, "collection": "fx_stale", "true_code": "INDEX_STALE", "true_stage": "index", "true_subcause": None, "family": "index_stale"})

    # ---- index missing: new files never indexed ----------------------------------------------------------------------
    miss_dir = base / "fixture_corpus_missing"
    make_corpus(miss_dir, seed=seed, n_docs=max(n_per_code, 8))
    _build("fx_missing", miss_dir)
    tmp = base / "fixture_corpus_new"
    nfacts = make_corpus(tmp, seed=seed + 1, n_docs=n_per_code)
    for f in nfacts:
        new_rel = "docs/new_" + Path(f["md"]).name
        shutil.copy(tmp / f["md"], miss_dir / new_rel)
        cases.append({**case_for({**f, "md": new_rel}, "prose"), "collection": "fx_missing", "true_code": "INDEX_MISSING", "true_stage": "index", "true_subcause": None, "family": "index_missing"})

    # ---- retrieval miss / rank low: measured on the healthy pipeline -------------------------------------------------------
    miss, low = [], []
    generic = [{"ctype": "prose", "question": "What is the maximum operating temperature of the unit?", "expected_answer": f"maximum operating temperature of {f['t_max']} degrees Celsius",
                "expected_source": f["md"]} for f in facts]
    for c in anchorless + generic:
        gold = {g["chunk_id"] for g in locate_gold(good.rows, c)}
        if not gold:
            continue
        tr = good.run(c["question"], k=8, diagnostic=True)
        r_h, r_f = _best_rank(tr.hybrid, gold), _best_rank(tr.reranked, gold)
        if r_h is None and len(miss) < n_per_code:
            miss.append(c)
        elif r_h is not None and r_h > 8 and (r_f is None or r_f > 8) and len(low) < n_per_code:
            low.append(c)
    cases += [{**c, "collection": "fx_good", "true_code": "RETRIEVAL_MISS", "true_stage": "retrieve", "true_subcause": None, "family": "retrieval_miss"} for c in miss]
    cases += [{**c, "collection": "fx_good", "true_code": "RANK_LOW", "true_stage": "retrieve", "true_subcause": None, "family": "rank_low"} for c in low]

    # ---- controls: the healthy pipeline already has the answer in context --------------------------------------------------------
    ctl = [c for c in pool() if _passes(good, c)][: n_per_code]
    cases += [{**c, "collection": "fx_good", "true_code": "GENERATION_SIDE", "true_stage": "downstream", "true_subcause": None, "family": "control"} for c in ctl]
    return _number(cases)
