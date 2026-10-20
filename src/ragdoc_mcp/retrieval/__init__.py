"""Retrieval module for RAG Doctor MCP - hybrid search with RRF and cross-encoder reranking."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

from ragdoc_mcp.chunkers import Chunk
from ragdoc_mcp.embedders import BaseEmbedder, get_embedder_registry
from ragdoc_mcp.schemas import ContentType, RerankerChoice, RetrieverChoice
from ragdoc_mcp.schemas import SearchResult as SchemaSearchResult
from ragdoc_mcp.stores import BaseStore, get_store_registry
from ragdoc_mcp.stores import SearchResult as StoreSearchResult


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    chunk: Chunk
    dense_score: float
    bm25_score: float
    fused_score: float
    rank_dense: int
    rank_bm25: int
    rank_fused: int


class HybridRetriever:
    """Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        retrieve_top_k: int = 50,
        rerank_top_k: int = 8,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        self._reranker = None

    def retrieve(self, query: str, filters: dict | None = None) -> list[RetrievalResult]:
        """Perform hybrid retrieval: dense + BM25 fused with RRF."""
        # Dense search
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.store.search(query_embedding, self.retrieve_top_k, filters)

        # BM25 search
        bm25_results = self.store.bm25_search(query, self.retrieve_top_k, filters)

        # Convert to dict for rank lookup
        dense_ranks = {r.chunk_id: i + 1 for i, r in enumerate(dense_results)}
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}

        # All unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        # Get full chunk data for all candidates
        chunk_map = {}
        for r in dense_results + bm25_results:
            if r.chunk_id not in chunk_map:
                chunk_map[r.chunk_id] = r

        # Compute RRF scores
        fused_results = []
        for chunk_id in all_chunk_ids:
            rank_d = dense_ranks.get(chunk_id, self.retrieve_top_k + 1)
            rank_b = bm25_ranks.get(chunk_id, self.retrieve_top_k + 1)

            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (self.rrf_k + rank_d) + 1.0 / (self.rrf_k + rank_b)

            chunk_data = chunk_map[chunk_id]
            fused_results.append(RetrievalResult(
                chunk=Chunk(
                    chunk_id=chunk_data.chunk_id,
                    text=chunk_data.text,
                    file_path=chunk_data.metadata.get("file_path", ""),
                    page=chunk_data.metadata.get("page"),
                    line_start=chunk_data.metadata.get("line_start"),
                    line_end=chunk_data.metadata.get("line_end"),
                    heading_path=chunk_data.metadata.get("heading_path"),
                    content_type=ContentType(chunk_data.metadata.get("content_type", "text")),
                    token_count=chunk_data.metadata.get("token_count", 0),
                    metadata=chunk_data.metadata.get("metadata", {}),
                ),
                dense_score=1.0 / rank_d if rank_d <= self.retrieve_top_k else 0.0,
                bm25_score=1.0 / rank_b if rank_b <= self.retrieve_top_k else 0.0,
                fused_score=rrf_score,
                rank_dense=rank_d,
                rank_bm25=rank_b,
                rank_fused=0,  # Will be set after sorting
            ))

        # Sort by fused score
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)

        # Assign fused ranks
        for i, r in enumerate(fused_results):
            r.rank_fused = i + 1

        return fused_results[:self.retrieve_top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using fastembed."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from fastembed import CrossEncoder
            self._model = CrossEncoder(model_name=self._model_name)
        except ImportError:
            raise RuntimeError("fastembed not installed. Install with: pip install fastembed")

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        self._load_model()

        # Prepare pairs
        pairs = [(query, r.chunk.text) for r in results]

        # Get scores
        scores = list(self._model.score(pairs))

        # Update results with rerank scores
        for i, r in enumerate(results):
            r.rerank_score = scores[i]

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        return results[:top_k]


class RetrievalPipeline:
    """Complete retrieval pipeline: hybrid search + optional reranking."""

    def __init__(
        self,
        store: BaseStore,
        embedder: BaseEmbedder,
        retriever_choice: RetrieverChoice = RetrieverChoice.HYBRID_RRF,
        reranker_choice: RerankerChoice = RerankerChoice.CROSS_ENCODER,
        reranker_model: str = "BAAI/bge-reranker-base",
        retrieve_top_k: int = 50,
        rerank_top_k: int = 8,
        rrf_k: int = 60,
        max_tokens_context: int = 1500,
    ):
        self.store = store
        self.embedder = embedder
        self.retriever_choice = retriever_choice
        self.reranker_choice = reranker_choice
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        self.max_tokens_context = max_tokens_context

        self.hybrid_retriever = HybridRetriever(
            store, embedder, retrieve_top_k, rerank_top_k, rrf_k
        )

        self.reranker = None
        if reranker_choice == RerankerChoice.CROSS_ENCODER:
            self.reranker = CrossEncoderReranker(reranker_model)

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict | None = None,
        detail_level: str = "brief",
    ) -> tuple[list[SchemaSearchResult], dict[str, float]]:
        """
        Execute retrieval pipeline.
        Returns (results, timing_info).
        """
        timings = {}
        effective_k = k or self.rerank_top_k

        # Hybrid retrieval
        start = time.perf_counter()
        fused_results = self.hybrid_retriever.retrieve(query, filters)
        timings["retrieval_time_ms"] = (time.perf_counter() - start) * 1000

        # Rerank if enabled
        start = time.perf_counter()
        if self.reranker and len(fused_results) > 1:
            reranked = self.reranker.rerank(query, fused_results, min(effective_k * 2, len(fused_results)))
            final_results = reranked[:effective_k]
        else:
            final_results = fused_results[:effective_k]
        timings["rerank_time_ms"] = (time.perf_counter() - start) * 1000

        # Pack into context window
        packed_results = self._pack_context(final_results)

        # Convert to schema format
        schema_results = []
        for r in packed_results:
            schema_results.append(SchemaSearchResult(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                score=r.fused_score if not hasattr(r, 'rerank_score') else r.rerank_score,
                file_path=r.chunk.file_path,
                page=r.chunk.page,
                line_start=r.chunk.line_start,
                line_end=r.chunk.line_end,
                heading_path=r.chunk.heading_path,
                content_type=r.chunk.content_type,
                stage_scores={
                    "dense": r.dense_score,
                    "bm25": r.bm25_score,
                    "fused": r.fused_score,
                    "rerank": getattr(r, 'rerank_score', 0.0),
                },
                truncated=r.truncated if hasattr(r, 'truncated') else False,
                untrusted=True,
            ))

        return schema_results, timings

    def _pack_context(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Pack results into token budget, truncating if needed."""
        packed = []
        total_tokens = 0

        for r in results:
            chunk_tokens = r.chunk.token_count

            if total_tokens + chunk_tokens <= self.max_tokens_context:
                r.truncated = False
                packed.append(r)
                total_tokens += chunk_tokens
            else:
                # Truncate this chunk to fit
                remaining = self.max_tokens_context - total_tokens
                if remaining > 50:  # Minimum viable chunk
                    # Truncate text (rough)
                    words = r.chunk.text.split()
                    keep_words = int(remaining * 4 / 5)  # ~4 chars per token
                    if keep_words > 20:
                        r.chunk.text = ' '.join(words[:keep_words]) + " [...] [truncated]"
                        r.chunk.token_count = remaining
                        r.truncated = True
                        packed.append(r)
                        total_tokens += remaining
                break

        return packed


def create_retrieval_pipeline(
    store_choice: str,
    embedder_name: str,
    collection_path: str,
    **kwargs
) -> RetrievalPipeline:
    """Factory function to create a retrieval pipeline."""
    store_registry = get_store_registry()
    embedder_registry = get_embedder_registry()

    # Create store
    from ragdoc_mcp.schemas import StoreChoice
    store = store_registry.create_store(StoreChoice(store_choice), path=collection_path)
    # Note: store.initialize() should be called separately with dimensions

    # Get embedder
    embedder = embedder_registry.get(embedder_name)

    return RetrievalPipeline(store=store, embedder=embedder, **kwargs)
