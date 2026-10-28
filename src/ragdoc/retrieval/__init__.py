"""BM25, reciprocal rank fusion, rerankers and token-budget packing."""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from ..errors import RagDocError
from ..textutil import STOP, content_words, count_tokens, tokens, truncate_words


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.dl: list[int] = []
        for i, t in enumerate(texts):
            toks = [w for w in tokens(t) if w not in STOP]
            self.dl.append(len(toks))
            for term, c in Counter(toks).items():
                self.inv[term].append((i, c))
        self.n = len(ids)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.inv.items()}

    def search(self, query: str, limit: int = 100, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = [w for w in tokens(query) if w not in STOP] or tokens(query)
        scores: dict[int, float] = defaultdict(float)
        for term in set(q):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.inv[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        items = [(self.ids[i], s) for i, s in scores.items() if allowed is None or self.ids[i] in allowed]
        return heapq.nlargest(limit, items, key=lambda x: x[1])


def rrf(rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion."""
    score: dict[str, float] = defaultdict(float)
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for r, cid in enumerate(ids, 1):
            score[cid] += w / (k + r)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


def _lexical_overlap(query: str, docs: list[str]) -> list[float]:
    qw = list(dict.fromkeys(content_words(query)))
    if not qw or not docs:
        return [0.0] * len(docs)
    dtoks = [tokens(d) for d in docs]
    dsets = [set(t) for t in dtoks]
    n = len(docs)
    idf = {w: math.log(1 + (n - sum(1 for s in dsets if w in s) + 0.5) / (sum(1 for s in dsets if w in s) + 0.5)) for w in qw}
    tot = sum(idf.values()) or 1.0
    bigrams = list(zip(qw, qw[1:]))
    out = []
    for toks, s in zip(dtoks, dsets):
        cov = sum(idf[w] for w in qw if w in s) / tot
        joined = " " + " ".join(toks) + " "
        bi = (sum(1 for a, b in bigrams if f" {a} {b} " in joined) / len(bigrams)) if bigrams else 0.0
        out.append(cov + 0.25 * bi + 0.05 / (1 + len(toks) / 200))
    return out


_CROSS = {}


def rerank_scores(name: str, query: str, docs: list[str]) -> list[float] | None:
    """Scores aligned with docs (higher = better), or None to keep the incoming order."""
    if name in ("off", "none", "", None):
        return None
    if name == "lexical_overlap":
        return _lexical_overlap(query, docs)
    if name == "_reverse":  # TEST-ONLY fault injector: breaks reranking by inverting the order
        return [float(i) for i in range(len(docs))]
    if name.startswith("cross:"):
        model = name.split(":", 1)[1]
        try:
            if model not in _CROSS:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _CROSS[model] = TextCrossEncoder(model_name=model)
            return [float(s) for s in _CROSS[model].rerank(query, docs)]
        except Exception as e:  # noqa: BLE001
            raise RagDocError(f"Cross-encoder reranker unavailable: {e}", hint="pip install 'ragdoc-mcp[neural]' or use rerank='lexical_overlap'.", code="NO_RERANKER") from e
    raise RagDocError(f"Unknown reranker {name!r}.", hint="Use 'lexical_overlap', 'off' or 'cross:<fastembed model>'.", code="BAD_RERANKER")


def pack(rows: list[dict], max_tokens: int) -> tuple[list[dict], list[str]]:
    """Fit ranked chunks into a token budget. Shared by search and diagnose so both see the same cut."""
    out: list[dict] = []
    notes: list[str] = []
    used = 0
    for i, r in enumerate(rows):
        t = int(r["tokens"])
        if used + t <= max_tokens:
            out.append({**r, "truncated": False, "packed_tokens": t})
            used += t
            continue
        remaining = max_tokens - used
        if remaining >= 40:
            cut = truncate_words(r["text"], remaining)
            out.append({**r, "text": cut, "truncated": True, "packed_tokens": count_tokens(cut)})
            notes.append(f"chunk {r['chunk_id']} truncated to fit the {max_tokens}-token budget")
            dropped = len(rows) - i - 1
        else:
            dropped = len(rows) - i
        if dropped:
            notes.append(f"{dropped} lower-ranked chunk(s) dropped by the token budget")
        break
    return out, notes
