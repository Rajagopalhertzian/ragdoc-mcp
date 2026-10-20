"""Collection management for RAG Doctor MCP - directory layout, manifest, and indexing."""

import json
import hashlib
import os
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
import pyarrow as pa

from ragdoc_mcp.config import settings
from ragdoc_mcp.parsers import get_parser_registry, ParsedDocument
from ragdoc_mcp.chunkers import get_chunker_registry, Chunk
from ragdoc_mcp.embedders import get_embedder_registry
from ragdoc_mcp.stores import get_store_registry, StoreChoice, BaseStore
from ragdoc_mcp.planner import RagPlan, StageConfig
from ragdoc_mcp.schemas import (
    ContentType, CollectionManifest, CorpusProfile, ChunkerChoice, ParserChoice,
    RetrieverChoice, StoreChoice as SchemaStoreChoice, RagConstraints,
    PrivacyMode, HardwareTarget
)
from ragdoc_mcp.security import get_path_allowlist


@dataclass
class IndexProgress:
    """Progress tracking for index building."""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    current_file: str = ""
    stage: str = "idle"  # idle, parsing, chunking, embedding, storing, complete
    start_time: float = 0.0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CollectionManager:
    """Manages RAG collection directories and indexing."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.allowlist = get_path_allowlist()
        self.collection_dir = self.allowlist.get_collection_dir(collection_name)
        self.parsed_dir = self.collection_dir / "parsed"
        self.chunks_path = self.collection_dir / "chunks.parquet"
        self.index_dir = self.collection_dir / "index"
        self.eval_dir = self.collection_dir / "eval"
        self.logs_dir = self.collection_dir / "logs"
        self.manifest_path = self.collection_dir / "manifest.json"

        # Create subdirectories
        for d in [self.parsed_dir, self.index_dir, self.eval_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_manifest(self) -> Optional[CollectionManifest]:
        """Load collection manifest if exists."""
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
        return CollectionManifest(**data)

    def save_manifest(self, manifest: CollectionManifest) -> None:
        """Save collection manifest."""
        with open(self.manifest_path, "w") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    def get_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_file_cached(self, file_path: Path, file_hash: str) -> bool:
        """Check if file is already parsed and cached."""
        cache_file = self.parsed_dir / f"{file_hash}.json"
        return cache_file.exists()

    def load_parsed_cache(self, file_hash: str) -> Optional[ParsedDocument]:
        """Load parsed document from cache."""
        cache_file = self.parsed_dir / f"{file_hash}.json"
        if not cache_file.exists():
            return None
        with open(cache_file, "r") as f:
            data = json.load(f)
        return ParsedDocument(**data)

    def save_parsed_cache(self, parsed: ParsedDocument) -> None:
        """Save parsed document to cache."""
        cache_file = self.parsed_dir / f"{parsed.file_hash}.json"
        with open(cache_file, "w") as f:
            # Convert to dict for JSON serialization
            data = {
                "file_path": parsed.file_path,
                "file_hash": parsed.file_hash,
                "content_type": parsed.content_type.value,
                "text": parsed.text,
                "pages": parsed.pages,
                "metadata": parsed.metadata,
                "parser_used": parsed.parser_used,
                "parse_time_ms": parsed.parse_time_ms,
            }
            json.dump(data, f)

    def load_chunks(self) -> list[Chunk]:
        """Load chunks from parquet file."""
        if not self.chunks_path.exists():
            return []
        table = pq.read_table(self.chunks_path)
        chunks = []
        for row in table.to_pylist():
            # Convert content_type string back to enum
            row_data = dict(row)
            if "content_type" in row_data:
                row_data["content_type"] = ContentType(row_data["content_type"])
            chunks.append(Chunk(**row_data))
        return chunks

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save chunks to parquet file."""
        if not chunks:
            return
        # Convert to arrow table
        data = [asdict(c) for c in chunks]
        # Handle enums
        for d in data:
            d["content_type"] = d["content_type"].value if hasattr(d["content_type"], "value") else d["content_type"]
        table = pa.Table.from_pylist(data)
        pq.write_table(table, self.chunks_path)

    def get_or_create_store(self, plan: RagPlan) -> BaseStore:
        """Get or create vector store for this collection."""
        store_registry = get_store_registry()
        
        # Map schema StoreChoice to stores StoreChoice
        store_map = {
            SchemaStoreChoice.LANCEDB: StoreChoice.LANCEDB,
            SchemaStoreChoice.QDRANT: StoreChoice.QDRANT,
            SchemaStoreChoice.PGVECTOR: StoreChoice.PGVECTOR,
        }
        store_choice = store_map.get(plan.config.store, StoreChoice.LANCEDB)
        
        store = store_registry.create_store(store_choice, path=self.index_dir)
        embedder_registry = get_embedder_registry()
        embedder = embedder_registry.get(plan.config.embedder)
        store.initialize("collection", embedder.dimensions)
        return store


class IndexBuilder:
    """Builds RAG index from plan with resumable, batched processing."""

    def __init__(self, collection_manager: CollectionManager, plan: RagPlan):
        self.collection = collection_manager
        self.plan = plan
        self.parser_registry = get_parser_registry()
        self.chunker_registry = get_chunker_registry()
        self.embedder_registry = get_embedder_registry()
        self.progress = IndexProgress()
        self.progress.start_time = time.time()

    def find_files(self, corpus_path: Path) -> list[Path]:
        """Find all files in corpus directory."""
        files = list(corpus_path.rglob("*"))
        return [f for f in files if f.is_file() and not f.name.startswith('.')]

    def should_process_file(self, file_path: Path, file_hash: str, force_reindex: bool = False) -> bool:
        """Check if file needs processing (not cached or changed)."""
        if force_reindex:
            return True
        
        manifest = self.collection.get_manifest()
        if manifest and file_hash in manifest.file_hashes:
            # Check if hash matches (file unchanged)
            if manifest.file_hashes[file_hash] == file_hash:
                return False
        return True

    def parse_file(self, file_path: Path, parser_name: str) -> ParsedDocument:
        """Parse a file with specified parser."""
        if parser_name == "fast":
            return self.parser_registry.parse(file_path, "fast")
        elif parser_name == "docling":
            return self.parser_registry.parse(file_path, "docling")
        else:
            # Auto-detect
            return self.parser_registry.parse(file_path)

    def chunk_document(self, doc: ParsedDocument, plan: RagPlan) -> list[Chunk]:
        """Chunk a parsed document according to plan."""
        # Get chunker for this content type
        chunker_name = plan.config.chunker.get(doc.content_type, ChunkerChoice.HEADING_AWARE)
        chunker = self.chunker_registry.get_chunker(chunker_name.value)
        
        # Get chunk size/overlap from plan
        chunk_size = plan.config.chunk_size
        overlap = plan.config.overlap
        
        chunks = chunker.chunk(doc, chunk_size, overlap)
        return chunks

    def embed_chunks(self, chunks: list[Chunk], plan: RagPlan) -> list[list[float]]:
        """Embed chunks using plan's embedder."""
        embedder = self.embedder_registry.get(plan.config.embedder)
        texts = [c.text for c in chunks]
        result = embedder.embed(texts)
        return result.embeddings

    def build_index(
        self,
        corpus_path: str | Path,
        resume: bool = True,
        force_reindex: bool = False,
        batch_size: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> CollectionManifest:
        """Build or resume index for a corpus."""
        corpus_path = Path(corpus_path)
        
        # Find all files
        files = self.find_files(corpus_path)
        self.progress.total_files = len(files)
        self.progress.stage = "parsing"
        
        # Load existing manifest or create new
        manifest = self.collection.get_manifest()
        if manifest is None or force_reindex:
            manifest = CollectionManifest(
                collection_name=self.collection.collection_name,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                plan=self.plan,
                version=1,
                consent_flags={},
                file_hashes={},
                total_chunks=0,
                index_path=str(self.collection.index_dir),
            )
        
        all_chunks = self.collection.load_chunks()
        total_chunks = len(all_chunks)
        
        # Process files in batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file_path in batch:
                self.progress.current_file = str(file_path)
                self.progress.processed_files += 1
                
                if progress_callback:
                    progress_callback(self.progress)
                
                # Compute file hash
                file_hash = self.collection.get_file_hash(file_path)
                
                # Check if already processed
                if resume and not force_reindex:
                    if file_hash in manifest.file_hashes:
                        if manifest.file_hashes[file_hash] == file_hash:
                            continue  # Skip unchanged file
                
                # Parse
                self.progress.stage = "parsing"
                try:
                    parser_name = self.plan.config.parser.get(
                        ContentType.PDF, ParserChoice.DOCLING
                    ).value
                    # Actually detect content type first
                    fast_parsed = self.parser_registry.parse(file_path, "fast")
                    parser_name = self.plan.config.parser.get(
                        fast_parsed.content_type, ParserChoice.DOCLING
                    ).value
                    
                    parsed = self.parse_file(file_path, parser_name)
                    self.collection.save_parsed_cache(parsed)
                except Exception as e:
                    self.progress.errors.append(f"Parse failed for {file_path}: {e}")
                    continue
                
                # Chunk
                self.progress.stage = "chunking"
                try:
                    chunks = self.chunk_document(parsed, self.plan)
                except Exception as e:
                    self.progress.errors.append(f"Chunk failed for {file_path}: {e}")
                    continue
                
                # Embed
                self.progress.stage = "embedding"
                try:
                    embeddings = self.embed_chunks(chunks, self.plan)
                except Exception as e:
                    self.progress.errors.append(f"Embed failed for {file_path}: {e}")
                    continue
                
                # Store
                self.progress.stage = "storing"
                try:
                    store = self.collection.get_or_create_store(self.plan)
                    store.add_chunks(chunks, embeddings)
                except Exception as e:
                    self.progress.errors.append(f"Store failed for {file_path}: {e}")
                    continue
                
                # Update tracking
                all_chunks.extend(chunks)
                total_chunks += len(chunks)
                manifest.file_hashes[file_hash] = file_hash
            
            # Save progress after each batch
            self.collection.save_chunks(all_chunks)
            manifest.total_chunks = total_chunks
            self.collection.save_manifest(manifest)
        
        self.progress.stage = "complete"
        manifest.total_chunks = total_chunks
        self.collection.save_manifest(manifest)
        
        if progress_callback:
            progress_callback(self.progress)
        
        return manifest


def build_index(
    collection_name: str,
    plan: RagPlan,
    corpus_path: str | Path,
    resume: bool = True,
    force_reindex: bool = False,
    batch_size: int = 10,
    progress_callback: Optional[callable] = None,
) -> CollectionManifest:
    """Main entry point for building index."""
    collection = CollectionManager(collection_name)
    builder = IndexBuilder(collection, plan)
    return builder.build_index(
        corpus_path=corpus_path,
        resume=resume,
        force_reindex=force_reindex,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )