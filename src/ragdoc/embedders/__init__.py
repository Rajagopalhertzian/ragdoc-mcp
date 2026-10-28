"""Embedders. Default is an offline deterministic hashing embedder (no downloads).
Neural embedders are optional via fastembed (`pip install ragdoc-mcp[neural]`)."""
from __future__ import annotations

import json
import math
import zlib
from collections import Counter
from functools import lru_cache
from importlib import resources

import numpy as np

from ..errors import RagDocError
from ..textutil import STOP, tokens

HASH_DIM = 384


@lru_cache(maxsize=1)
def load_models() -> dict:
    try:
        return json.loads(resources.files("ragdoc").joinpath("models.json").read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {"embedders": {"offline": {"name": "hashing-384", "kind": "hashing", "dim": HASH_DIM}}, "rerankers": {}}


class HashingEmbedder:
    """Signed feature hashing over word unigrams, bigrams and character 4-grams.

    Deterministic and fully offline. It is lexical-fuzzy rather than semantic, so it is weaker than a neural
    model; the hybrid BM25 leg compensates. Use `ragdoc-mcp[neural]` for a real semantic embedder."""

    kind = "hashing"

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = [t for t in tokens(text) if t not in STOP]
        feats: Counter = Counter()
        for t in toks:
            feats[t] += 1.0
            if len(t) >= 5:
                for i in range(len(t) - 3):
                    feats["#" + t[i : i + 4]] += 0.3
        for a, b in zip(toks, toks[1:]):
            feats[a + " " + b] += 0.5
        for f, w in feats.items():
            b = f.encode()
            idx = zlib.crc32(b) % self.dim
            sign = 1.0 if (zlib.crc32(b, 7) & 1) else -1.0
            v[idx] += sign * (1.0 + math.log(w)) if w >= 1 else sign * w
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32) if texts else np.zeros((0, self.dim), np.float32)

    embed_query_batch = embed_docs

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FastEmbedder:
    kind = "fastembed"

    def __init__(self, model_name: str):
        try:
            from fastembed import TextEmbedding
        except Exception as e:  # noqa: BLE001
            raise RagDocError("fastembed is not installed.", hint="pip install 'ragdoc-mcp[neural]'", code="NO_FASTEMBED") from e
        self.name = model_name
        try:
            self._m = TextEmbedding(model_name=model_name)
            self.dim = int(len(next(iter(self._m.embed(["probe"])))))
        except Exception as e:  # noqa: BLE001  (offline, unknown model, corrupt cache ...)
            raise RagDocError(
                f"Could not load embedding model '{model_name}': {str(e)[:160]}",
                hint="The first use downloads the model once. Check the network and the model name, or use the offline embedder (plan_rag constraints.embedder='offline').",
                code="MODEL_UNAVAILABLE",
            ) from e

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return (a / np.where(n == 0, 1, n)).astype(np.float32)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self._norm(np.vstack([np.asarray(v) for v in self._m.passage_embed(texts)]))

    def embed_query(self, text: str) -> np.ndarray:
        return self._norm(np.asarray(next(iter(self._m.query_embed([text])))).reshape(1, -1))[0]


def get_embedder(name: str):
    """name is 'hashing-384' (default) or any fastembed model id such as 'BAAI/bge-small-en-v1.5'."""
    if name.startswith("hashing"):
        try:
            return HashingEmbedder(int(name.split("-")[1]))
        except (IndexError, ValueError):
            return HashingEmbedder()
    return FastEmbedder(name)


def neural_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("fastembed") is not None
