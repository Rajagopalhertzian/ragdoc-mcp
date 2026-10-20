"""Vector stores for RAG Doctor MCP - LanceDB default with Qdrant/pgvector adapters."""

import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from ragdoc_mcp.chunkers import Chunk
from ragdoc_mcp.schemas import StoreChoice


class SearchResult(BaseModel):
    """Result from vector search."""
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class BaseStore(ABC):
    """Abstract base class for vector stores."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Store identifier."""
        pass

    @abstractmethod
    def initialize(self, collection_name: str, dimensions: int) -> None:
        """Initialize a collection."""
        pass

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with embeddings."""
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], k: int, filters: dict | None = None) -> list[SearchResult]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    def bm25_search(self, query: str, k: int, filters: dict | None = None) -> list[SearchResult]:
        """BM25 keyword search."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        """Get a specific chunk by ID."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total chunk count."""
        pass


class LanceDBStore(BaseStore):
    """LanceDB local vector store."""

    name = "lancedb"

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db = None
        self._table = None
        self._collection_name = None

    def _get_lancedb(self):
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(str(self._db_path))
        return self._db

    def initialize(self, collection_name: str, dimensions: int) -> None:
        self._collection_name = collection_name
        db = self._get_lancedb()

        # Define schema
        import pyarrow as pa
        schema = pa.schema([
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dimensions), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("page", pa.int32(), nullable=True),
            pa.field("line_start", pa.int32(), nullable=True),
            pa.field("line_end", pa.int32(), nullable=True),
            pa.field("heading_path", pa.string(), nullable=True),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("token_count", pa.int32(), nullable=False),
            pa.field("metadata", pa.string(), nullable=True),  # JSON string
        ])

        # Create or open table
        if collection_name in db.table_names():
            self._table = db.open_table(collection_name)
        else:
            self._table = db.create_table(collection_name, schema=schema)

        # Create BM25 index (full-text search)
        try:
            self._table.create_fts_index("text", replace=True)
        except Exception:
            pass  # FTS may not be available in all versions

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if self._table is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        import pyarrow as pa
        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append({
                "chunk_id": chunk.chunk_id,
                "vector": emb,
                "text": chunk.text,
                "file_path": chunk.file_path,
                "page": chunk.page,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "heading_path": chunk.heading_path,
                "content_type": chunk.content_type.value,
                "token_count": chunk.token_count,
                "metadata": json.dumps(chunk.metadata) if chunk.metadata else None,
            })

        self._table.add(data)

    def search(self, query_embedding: list[float], k: int, filters: dict | None = None) -> list[SearchResult]:
        if self._table is None:
            raise RuntimeError("Store not initialized.")

        # Build query
        query = self._table.search(query_embedding).limit(k)

        # Apply filters if any
        if filters:
            for key, value in filters.items():
                if key == "content_type":
                    query = query.where(f"content_type = '{value}'")
                elif key == "file_path":
                    query = query.where(f"file_path = '{value}'")

        results = query.to_list()

        return [
            SearchResult(
                chunk_id=r["chunk_id"],
                score=float(r.get("_distance", 1.0 - r.get("score", 0.0))),
                text=r["text"],
                metadata={
                    "file_path": r["file_path"],
                    "page": r["page"],
                    "line_start": r["line_start"],
                    "line_end": r["line_end"],
                    "heading_path": r["heading_path"],
                    "content_type": r["content_type"],
                    "token_count": r["token_count"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                }
            )
            for r in results
        ]

    def bm25_search(self, query: str, k: int, filters: dict | None = None) -> list[SearchResult]:
        if self._table is None:
            raise RuntimeError("Store not initialized.")

        try:
            # Use FTS index for BM25-like search
            query_obj = self._table.search(query, query_type="fts").limit(k)

            if filters:
                for key, value in filters.items():
                    if key == "content_type":
                        query_obj = query_obj.where(f"content_type = '{value}'")

            results = query_obj.to_list()

            return [
                SearchResult(
                    chunk_id=r["chunk_id"],
                    score=float(r.get("score", 0.0)),
                    text=r["text"],
                    metadata={
                        "file_path": r["file_path"],
                        "page": r["page"],
                        "line_start": r["line_start"],
                        "line_end": r["line_end"],
                        "heading_path": r["heading_path"],
                        "content_type": r["content_type"],
                        "token_count": r["token_count"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                    }
                )
                for r in results
            ]
        except Exception:
            # Fallback: simple text search
            return self._fallback_text_search(query, k, filters)

    def _fallback_text_search(self, query: str, k: int, filters: dict | None) -> list[SearchResult]:
        """Simple text search fallback."""
        if self._table is None:
            return []

        query_terms = query.lower().split()
        all_results = self._table.to_pandas()

        scored = []
        for _, row in all_results.iterrows():
            text_lower = row["text"].lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, row in scored[:k]:
            results.append(SearchResult(
                chunk_id=row["chunk_id"],
                score=float(score),
                text=row["text"],
                metadata={
                    "file_path": row["file_path"],
                    "page": row["page"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "heading_path": row["heading_path"],
                    "content_type": row["content_type"],
                    "token_count": row["token_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
            ))
        return results

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        if self._table is None:
            return None

        try:
            result = self._table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()
            if not result:
                return None
            r = result[0]
            return SearchResult(
                chunk_id=r["chunk_id"],
                score=1.0,
                text=r["text"],
                metadata={
                    "file_path": r["file_path"],
                    "page": r["page"],
                    "line_start": r["line_start"],
                    "line_end": r["line_end"],
                    "heading_path": r["heading_path"],
                    "content_type": r["content_type"],
                    "token_count": r["token_count"],
                    "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                }
            )
        except Exception:
            return None

    def delete_collection(self) -> None:
        if self._table is not None:
            db = self._get_lancedb()
            db.drop_table(self._collection_name)
            self._table = None

    def count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()


class QdrantStore(BaseStore):
    """Qdrant vector store (for medium/large scale)."""

    name = "qdrant"

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        self._url = url
        self._api_key = api_key
        self._client = None
        self._collection_name = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    def initialize(self, collection_name: str, dimensions: int) -> None:
        self._collection_name = collection_name
        client = self._get_client()

        from qdrant_client.models import Distance, VectorParams

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )

        # Create payload index for filtering
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="content_type",
                field_schema="keyword",
            )
        except Exception:
            pass

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        client = self._get_client()

        from qdrant_client.models import PointStruct

        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=emb,
                payload={
                    "text": chunk.text,
                    "file_path": chunk.file_path,
                    "page": chunk.page,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "heading_path": chunk.heading_path,
                    "content_type": chunk.content_type.value,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
            ))

        client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query_embedding: list[float], k: int, filters: dict | None = None) -> list[SearchResult]:
        client = self._get_client()

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=str(r.id),
                score=r.score,
                text=r.payload["text"],
                metadata={k: v for k, v in r.payload.items() if k != "text"}
            )
            for r in results
        ]

    def bm25_search(self, query: str, k: int, filters: dict | None = None) -> list[SearchResult]:
        # Qdrant doesn't have native BM25, use text search
        client = self._get_client()

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        # Use scroll with text search (limited)
        results, _ = client.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=k * 10,  # Get more for filtering
            with_payload=True,
            with_vectors=False,
        )

        # Simple text matching
        query_terms = query.lower().split()
        scored = []
        for r in results:
            text = r.payload.get("text", "").lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                chunk_id=str(r.id),
                score=float(score),
                text=r.payload["text"],
                metadata={k: v for k, v in r.payload.items() if k != "text"}
            )
            for score, r in scored[:k]
        ]

    def get_chunk(self, chunk_id: str) -> SearchResult | None:
        client = self._get_client()
        try:
            results = client.retrieve(
                collection_name=self._collection_name,
                ids=[chunk_id],
                with_payload=True,
            )
            if not results:
                return None
            r = results[0]
            return SearchResult(
                chunk_id=str(r.id),
                score=1.0,
                text=r.payload["text"],
                metadata={k: v for k, v in r.payload.items() if k != "text"}
            )
        except Exception:
            return None

    def delete_collection(self) -> None:
        client = self._get_client()
        client.delete_collection(self._collection_name)

    def count(self) -> int:
        client = self._get_client()
        info = client.get_collection(self._collection_name)
        return info.points_count


class StoreRegistry:
    """Registry for managing vector stores."""

    def __init__(self):
        self._stores: dict[str, BaseStore] = {}

    def register(self, name: str, store: BaseStore) -> None:
        self._stores[name] = store

    def get(self, name: str) -> BaseStore:
        if name not in self._stores:
            raise ValueError(f"Store '{name}' not registered. Available: {list(self._stores.keys())}")
        return self._stores[name]

    def create_store(self, choice: StoreChoice, **kwargs) -> BaseStore:
        """Factory method to create store instances."""
        if choice == StoreChoice.LANCEDB:
            path = kwargs.get("path", Path.cwd() / ".ragdoc" / "lancedb")
            return LanceDBStore(Path(path))
        if choice == StoreChoice.QDRANT:
            url = kwargs.get("url", os.getenv("QDRANT_URL", "http://localhost:6333"))
            api_key = kwargs.get("api_key", os.getenv("QDRANT_API_KEY"))
            return QdrantStore(url, api_key)
        raise ValueError(f"Store {choice} not yet implemented")


# Global store registry
_store_registry: StoreRegistry | None = None


def get_store_registry() -> StoreRegistry:
    global _store_registry
    if _store_registry is None:
        _store_registry = StoreRegistry()
    return _store_registry
