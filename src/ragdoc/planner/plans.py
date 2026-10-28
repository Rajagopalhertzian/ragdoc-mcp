"""Plan construction helpers."""
from __future__ import annotations

import copy
import uuid

from ..chunkers import DEFAULT_CHUNKING
from ..retrieval.pipeline import DEFAULT_PACK, DEFAULT_RETRIEVAL


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_plan(source_path: str, collection: str, **overrides) -> dict:
    plan = {
        "id": "plan_" + uuid.uuid4().hex[:10],
        "collection": collection,
        "source_path": source_path,
        "parsers": {"*": "fast"},
        "chunking": dict(DEFAULT_CHUNKING, max_tokens=400),
        "embedder": {"name": "hashing-384", "dim": 384},
        "store": {"kind": "lancedb"},
        "retrieval": copy.deepcopy(DEFAULT_RETRIEVAL),
        "pack": dict(DEFAULT_PACK),
        "consent": {},
        "constraints": {},
        "decision_log": [],
        "estimates": {},
    }
    return deep_merge(plan, overrides)


def baseline_plan(source_path: str, collection: str, **overrides) -> dict:
    """Naive default used as the plan_rag comparison baseline: fixed 512-token windows, dense only, no rerank."""
    return default_plan(
        source_path, collection,
        chunking={"kind": "window", "max_tokens": 512, "overlap_pct": 10, "prepend_heading": False, "keep_tables_whole": False, "keep_formulas_whole": False},
        retrieval={"dense": True, "bm25": False, "rerank": "off"},
        **overrides,
    )
