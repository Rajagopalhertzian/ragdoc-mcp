"""The retrieval pipeline. search() and diagnose_failure() both run this and read the same Trace."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

from ..collection import Collection
from ..embedders import get_embedder
from ..errors import RagDocError
from ..stores import LanceStore
from . import BM25, pack, rerank_scores, rrf

DEFAULT_RETRIEVAL = {"dense": True, "bm25": True, "rerank": "lexical_overlap", "candidates": 50, "rrf_k": 60, "weights": {"dense": 1.0, "bm25": 1.0}}
DEFAULT_PACK = {"max_tokens": 1500}
DEPTH = 100
_BM25: dict[str, tuple[tuple, BM25]] = {}
_EMB: dict[str, object] = {}
FILTER_KEYS = {"ctype", "file", "page"}


@dataclass
class Trace:
    query: str
    k: int
    max_tokens: int
    cfg: dict
    dense: list[tuple[str, float]] = field(default_factory=list)
    bm25: list[tuple[str, float]] = field(default_factory=list)
    hybrid: list[tuple[str, float]] = field(default_factory=list)  # pre-rerank, up to DEPTH
    candidates: list[str] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)
    final: list[str] = field(default_factory=list)  # top-k ids
    packed: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def rank_of(ranked: list, cid: str) -> int | None:
        for i, item in enumerate(ranked, 1):
            if (item[0] if isinstance(item, tuple) else item) == cid:
                return i
        return None


class Pipeline:
    def __init__(self, coll: Collection):
        self.coll = coll
        self.manifest = coll.load_manifest()
        self.plan = self.manifest["plan"]
        self.rows = coll.read_rows()
        self.by_id = {r["chunk_id"]: r for r in self.rows}
        name = self.plan["embedder"]["name"]
        if name not in _EMB:
            _EMB[name] = get_embedder(name)
        self.embedder = _EMB[name]
        self.store = LanceStore(coll.lance_dir, self.embedder.dim)
        self.retrieval = {**DEFAULT_RETRIEVAL, **self.plan.get("retrieval", {})}
        self.pack_cfg = {**DEFAULT_PACK, **self.plan.get("pack", {})}
        st = coll.rows_path.stat() if coll.rows_path.exists() else None
        key = (st.st_mtime_ns, st.st_size) if st else (0, 0)
        ck = str(coll.rows_path)
        if ck not in _BM25 or _BM25[ck][0] != key:
            _BM25[ck] = (key, BM25([r["chunk_id"] for r in self.rows], [r["embed_text"] for r in self.rows]))
        self.bm25 = _BM25[ck][1]

    def _allowed(self, filters: dict | None) -> set[str] | None:
        if not filters:
            return None
        bad = set(filters) - FILTER_KEYS
        if bad:
            raise RagDocError(f"Unsupported filter(s): {sorted(bad)}", hint=f"Allowed filters: {sorted(FILTER_KEYS)}.", code="BAD_FILTER")
        ct = filters.get("ctype")
        cts = {ct} if isinstance(ct, str) else set(ct) if ct else None
        f = filters.get("file")
        pg = filters.get("page")
        out = set()
        for r in self.rows:
            if cts and r["ctype"] not in cts:
                continue
            if f and not (fnmatch.fnmatch(r["file"].lower(), str(f).lower()) or str(f).lower() in r["file"].lower()):
                continue
            if pg is not None and int(r["page"]) != int(pg):
                continue
            out.add(r["chunk_id"])
        return out

    def run(self, query: str, k: int = 8, max_tokens: int | None = None, filters: dict | None = None,
            diagnostic: bool = False, overrides: dict | None = None) -> Trace:
        cfg = {**self.retrieval, **(overrides or {})}
        mt = int(max_tokens or self.pack_cfg["max_tokens"])
        tr = Trace(query, k, mt, cfg)
        allowed = self._allowed(filters)
        if allowed is not None and not allowed:
            tr.notes.append("filters matched no chunks")
            return tr
        if cfg["dense"] or diagnostic:
            qv = self.embedder.embed_query(query)
            lim = DEPTH if allowed is None else max(DEPTH, min(self.store.count(), 500))
            tr.dense = [(i, s) for i, s in self.store.search(qv, lim) if i in self.by_id and (allowed is None or i in allowed)][:DEPTH]
        if cfg["bm25"] or diagnostic:
            tr.bm25 = self.bm25.search(query, DEPTH, allowed)
        rankings = {}
        if cfg["dense"]:
            rankings["dense"] = [i for i, _ in tr.dense]
        if cfg["bm25"]:
            rankings["bm25"] = [i for i, _ in tr.bm25]
        tr.hybrid = rrf(rankings, int(cfg["rrf_k"]), cfg.get("weights"))[:DEPTH]
        tr.candidates = [i for i, _ in tr.hybrid[: int(cfg["candidates"])]]
        scores = rerank_scores(cfg["rerank"], query, [self.by_id[i]["embed_text"] for i in tr.candidates])
        if scores is None:
            tr.reranked = [(i, s) for i, s in tr.hybrid[: len(tr.candidates)]]
        else:
            order = sorted(range(len(tr.candidates)), key=lambda j: (-scores[j], j))
            tr.reranked = [(tr.candidates[j], scores[j]) for j in order]
        tr.final = [i for i, _ in tr.reranked[:k]]
        tr.packed, pn = pack([self.by_id[i] for i in tr.final], mt)
        tr.notes.extend(pn)
        return tr
