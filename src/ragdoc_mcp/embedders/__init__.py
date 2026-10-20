"""Embedding models for RAG Doctor MCP using fastembed and sentence-transformers."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    embeddings: list[list[float]]
    model_name: str
    dimensions: int
    token_counts: list[int]
    total_tokens: int
    time_ms: float


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Embedder identifier."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum tokens per input."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        pass


class FastEmbedEmbedder(BaseEmbedder):
    """FastEmbed-based local embedder."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dimensions = None
        self._max_tokens = 512

    @property
    def name(self) -> str:
        return f"fastembed:{self._model_name}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
            # Get dimensions from a test embedding
            test_emb = list(self._model.embed(["test"]))[0]
            self._dimensions = len(test_emb)
        except ImportError:
            raise RuntimeError("fastembed not installed. Install with: pip install fastembed")

    def embed(self, texts: list[str]) -> EmbeddingResult:
        import time
        start = time.perf_counter()

        self._load_model()

        # Process in batches for memory efficiency
        embeddings = []
        token_counts = []

        for text in texts:
            # Truncate if needed (rough estimate)
            if len(text) > self._max_tokens * 4:
                text = text[:self._max_tokens * 4]

            emb = list(self._model.embed([text]))[0]
            embeddings.append(emb.tolist())
            token_counts.append(len(text) // 4)

        total_time = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._model_name,
            dimensions=self._dimensions,
            token_counts=token_counts,
            total_tokens=sum(token_counts),
            time_ms=total_time,
        )

    def embed_query(self, query: str) -> list[float]:
        self._load_model()
        emb = list(self._model.embed([query]))[0]
        return emb.tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-Transformers based embedder (alternative to fastembed)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dimensions = None
        self._max_tokens = 512

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            self._max_tokens = self._model.max_seq_length
        except ImportError:
            raise RuntimeError("sentence-transformers not installed. Install with: pip install sentence-transformers")

    def embed(self, texts: list[str]) -> EmbeddingResult:
        import time
        start = time.perf_counter()

        self._load_model()

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        token_counts = [len(t) // 4 for t in texts]

        total_time = (time.perf_counter() - start) * 1000

        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            model_name=self._model_name,
            dimensions=self._dimensions,
            token_counts=token_counts,
            total_tokens=sum(token_counts),
            time_ms=total_time,
        )

    def embed_query(self, query: str) -> list[float]:
        self._load_model()
        emb = self._model.encode([query], convert_to_numpy=True)[0]
        return emb.tolist()


class EmbedderRegistry:
    """Registry for managing embedders."""

    def __init__(self):
        self._embedders: dict[str, BaseEmbedder] = {}

    def register(self, name: str, embedder: BaseEmbedder) -> None:
        self._embedders[name] = embedder

    def get(self, name: str) -> BaseEmbedder:
        if name not in self._embedders:
            raise ValueError(f"Embedder '{name}' not registered. Available: {list(self._embedders.keys())}")
        return self._embedders[name]

    def get_default(self) -> BaseEmbedder:
        """Get default embedder (fastembed bge-small)."""
        if "default" not in self._embedders:
            self._embedders["default"] = FastEmbedEmbedder()
        return self._embedders["default"]


# Global embedder registry
_embedder_registry: EmbedderRegistry | None = None


def get_embedder_registry() -> EmbedderRegistry:
    global _embedder_registry
    if _embedder_registry is None:
        _embedder_registry = EmbedderRegistry()
        _embedder_registry.register("default", FastEmbedEmbedder())
        _embedder_registry.register("bge-small", FastEmbedEmbedder("BAAI/bge-small-en-v1.5"))
        _embedder_registry.register("bge-base", FastEmbedEmbedder("BAAI/bge-base-en-v1.5"))
        _embedder_registry.register("bge-large", FastEmbedEmbedder("BAAI/bge-large-en-v1.5"))
        _embedder_registry.register("nomic", FastEmbedEmbedder("nomic-ai/nomic-embed-text-v1.5"))
        _embedder_registry.register("e5-small", FastEmbedEmbedder("intfloat/e5-small-v2"))
        _embedder_registry.register("jina-code", FastEmbedEmbedder("jinaai/jina-embeddings-v2-base-code"))
    return _embedder_registry
