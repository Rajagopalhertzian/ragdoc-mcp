"""Diagnosis module for RAG Doctor MCP - diagnose_failure tool implementation."""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ragdoc_mcp.chunkers import get_chunker_registry, Chunk
from ragdoc_mcp.embedders import get_embedder_registry
from ragdoc_mcp.indexer import CollectionManager
from ragdoc_mcp.parsers import get_parser_registry, ParsedDocument
from ragdoc_mcp.retrieval import RetrievalPipeline
from ragdoc_mcp.schemas import (
    ContentType,
    DiagnosisResult,
    FailureCode,
    FixProposal,
    RerankerChoice,
    RetrieverChoice,
    StageEvidence,
    StoreChoice,
)
from ragdoc_mcp.stores import get_store_registry
from ragdoc_mcp.security import get_path_allowlist


@dataclass
class StageTrace:
    """Trace result for a single stage."""
    stage_num: int
    stage_name: str
    passed: bool
    failure_code: FailureCode | None
    confidence: float
    evidence: dict[str, Any]
    rank_table: list[dict] | None


class FailureDiagnoser:
    """Diagnoses RAG pipeline failures by tracing through each stage."""

    def __init__(self, collection_path: str, plan_config: dict):
        self.collection_path = collection_path
        self.plan_config = plan_config
        self.parser_registry = get_parser_registry()
        self.chunker_registry = get_chunker_registry()
        self.embedder_registry = get_embedder_registry()
        self.store_registry = get_store_registry()
        self.collection = CollectionManager(Path(collection_path).name)
        
        # Initialize retrieval pipeline
        self.retrieval = self._create_retrieval_pipeline()

    def _create_retrieval_pipeline(self) -> RetrievalPipeline:
        """Create retrieval pipeline from plan config."""
        store_choice = self.plan_config.get("store", "lancedb")
        embedder_name = self.plan_config.get("embedder", "bge-small")
        retrieve_top_k = self.plan_config.get("retrieve_top_k", 50)
        rerank_top_k = self.plan_config.get("rerank_top_k", 8)
        rrf_k = self.plan_config.get("rrf_k", 60)
        reranker_model = self.plan_config.get("reranker", "bge-reranker-base")
        max_tokens = self.plan_config.get("max_tokens_context", 1500)

        store = self.store_registry.create_store(StoreChoice(store_choice), path=Path(self.collection_path))
        store.initialize("collection", self.embedder_registry.get(embedder_name).dimensions)

        embedder = self.embedder_registry.get(embedder_name)

        return RetrievalPipeline(
            store=store,
            embedder=embedder,
            retriever_choice=RetrieverChoice.HYBRID_RRF,
            reranker_choice=RerankerChoice.CROSS_ENCODER,
            reranker_model=reranker_model,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            rrf_k=rrf_k,
            max_tokens_context=max_tokens,
        )

    def diagnose(
        self,
        question: str,
        expected_answer: str | None = None,
        expected_source: str | None = None,
    ) -> DiagnosisResult:
        """Run full diagnosis pipeline."""
        traces = []

        # Stage 1: Parse - check if expected answer exists in parsed docs
        trace1 = self._check_parse_stage(question, expected_answer, expected_source)
        traces.append(trace1)
        if not trace1.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 2: Chunk - check if answer span is in a single chunk
        trace2 = self._check_chunk_stage(question, expected_answer, expected_source)
        traces.append(trace2)
        if not trace2.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 3: Index - check if chunk is in index
        trace3 = self._check_index_stage(question, expected_answer, expected_source)
        traces.append(trace3)
        if not trace3.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 4: Retrieve - check rank in dense/BM25/hybrid
        trace4 = self._check_retrieve_stage(question, expected_answer, expected_source)
        traces.append(trace4)
        if not trace4.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 5: Rerank - check if rerank demoted
        trace5 = self._check_rerank_stage(question, expected_answer, expected_source)
        traces.append(trace5)
        if not trace5.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 6: Pack - check if truncated by token budget
        trace6 = self._check_pack_stage(question, expected_answer, expected_source)
        traces.append(trace6)
        if not trace6.passed:
            return self._build_result(traces, question, expected_answer, expected_source)

        # Stage 7: Downstream - answer in context but generation failed
        trace7 = self._check_downstream_stage(question, expected_answer, expected_source)
        traces.append(trace7)

        return self._build_result(traces, question, expected_answer, expected_source)

    def _load_parsed_for_source(self, expected_source: str) -> Optional[ParsedDocument]:
        """Load parsed document for the expected source file."""
        # Find the cached parsed document
        # First try to find by file hash
        source_path = Path(expected_source)
        if not source_path.is_absolute():
            # Try relative to collection
            allowlist = get_path_allowlist()
            try:
                source_path = allowlist.validate_read(Path(expected_source))
            except Exception:
                pass
        
        if not source_path.exists():
            return None
        
        file_hash = self._compute_file_hash(source_path)
        return self.collection.load_parsed_cache(file_hash)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        import hashlib
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy comparison."""
        import re
        # Lowercase, remove extra whitespace, normalize punctuation
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _fuzzy_contains(self, haystack: str, needle: str, threshold: float = 0.8) -> bool:
        """Check if needle is contained in haystack with fuzzy matching."""
        from difflib import SequenceMatcher
        haystack_norm = self._normalize_text(haystack)
        needle_norm = self._normalize_text(needle)
        
        # Exact substring check first
        if needle_norm in haystack_norm:
            return True
        
        # Fuzzy ratio check
        ratio = SequenceMatcher(None, haystack_norm, needle_norm).ratio()
        return ratio >= threshold

    def _check_parse_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 1: Check if expected answer exists in parsed output."""
        evidence = {
            "check": "expected_answer_in_parsed",
            "method": "fuzzy_match_normalized",
        }

        if not expected_answer or not expected_source:
            evidence["note"] = "No expected_answer or expected_source provided; limited check"
            return StageTrace(
                stage_num=1,
                stage_name="Parse",
                passed=True,
                failure_code=None,
                confidence=0.5,
                evidence=evidence,
                rank_table=None,
            )

        parsed = self._load_parsed_for_source(expected_source)
        if not parsed:
            evidence["note"] = f"Parsed cache not found for {expected_source}"
            evidence["source_file"] = expected_source
            return StageTrace(
                stage_num=1,
                stage_name="Parse",
                passed=False,
                failure_code=FailureCode.PARSE_LOSS,
                confidence=0.7,
                evidence=evidence,
                rank_table=None,
            )

        # Check if expected answer exists in parsed text
        found = False
        matched_location = None
        for page in parsed.pages:
            page_text = page.get("text", "")
            if self._fuzzy_contains(page_text, expected_answer):
                found = True
                matched_location = {
                    "page": page.get("page_num"),
                    "text_preview": page_text[:200],
                }
                break

        evidence["source_file"] = expected_source
        evidence["parser_used"] = parsed.parser_used
        evidence["total_pages"] = len(parsed.pages)

        if found:
            evidence["status"] = "found"
            evidence["matched_location"] = matched_location
            return StageTrace(
                stage_num=1,
                stage_name="Parse",
                passed=True,
                failure_code=None,
                confidence=0.9,
                evidence=evidence,
                rank_table=None,
            )

        # Check sub-causes
        failure_code = FailureCode.PARSE_LOSS
        for page in parsed.pages:
            if not page.get("text", "").strip():
                if page.get("tables") or page.get("equations"):
                    failure_code = FailureCode.PARSE_LOSS_TABLE_FLATTENED if page.get("tables") else FailureCode.PARSE_LOSS_EQUATION_GARBLED
                    break
                failure_code = FailureCode.PARSE_LOSS_SCANNED_NO_OCR
                break

        evidence["status"] = "not_found"
        evidence["failure_subtype"] = failure_code.value
        evidence["note"] = f"Expected answer not found in parsed output. Subtype: {failure_code.value}"

        return StageTrace(
            stage_num=1,
            stage_name="Parse",
            passed=False,
            failure_code=failure_code,
            confidence=0.85,
            evidence=evidence,
            rank_table=None,
        )

    def _check_chunk_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 2: Check if answer span survives chunking."""
        evidence = {
            "check": "answer_span_in_single_chunk",
            "method": "check_chunk_boundaries",
        }

        if not expected_answer or not expected_source:
            evidence["note"] = "No expected_answer or expected_source provided"
            return StageTrace(
                stage_num=2,
                stage_name="Chunk",
                passed=True,
                failure_code=None,
                confidence=0.5,
                evidence=evidence,
                rank_table=None,
            )

        parsed = self._load_parsed_for_source(expected_source)
        if not parsed:
            evidence["note"] = "Parsed document not available"
            return StageTrace(
                stage_num=2,
                stage_name="Chunk",
                passed=False,
                failure_code=FailureCode.CHUNK_SPLIT,
                confidence=0.6,
                evidence=evidence,
                rank_table=None,
            )

        # Get chunker config for this content type
        content_type = parsed.content_type
        chunker_name = self.plan_config.get("chunker", {}).get(content_type.value, "heading_aware")
        chunk_size = self.plan_config.get("chunk_size", 512)
        overlap = self.plan_config.get("overlap", 0.1)

        chunker = self.chunker_registry.get_chunker(chunker_name)
        chunks = chunker.chunk(parsed, chunk_size, overlap)

        # Check if expected answer is fully contained in any single chunk
        found_in_chunk = False
        split_across_chunks = False
        containing_chunk = None
        
        for chunk in chunks:
            if self._fuzzy_contains(chunk.text, expected_answer):
                found_in_chunk = True
                containing_chunk = chunk
                break
        
        # Also check if answer is split across adjacent chunks
        if not found_in_chunk and len(chunks) > 1:
            for i in range(len(chunks) - 1):
                combined = chunks[i].text + " " + chunks[i + 1].text
                if self._fuzzy_contains(combined, expected_answer):
                    split_across_chunks = True
                    break

        evidence["total_chunks"] = len(chunks)
        evidence["chunker"] = chunker_name
        evidence["chunk_size"] = chunk_size
        evidence["overlap"] = overlap
        evidence["content_type"] = content_type.value

        if found_in_chunk:
            evidence["status"] = "found_in_single_chunk"
            evidence["chunk_id"] = containing_chunk.chunk_id
            evidence["chunk_tokens"] = containing_chunk.token_count
            return StageTrace(
                stage_num=2,
                stage_name="Chunk",
                passed=True,
                failure_code=None,
                confidence=0.9,
                evidence=evidence,
                rank_table=None,
            )

        if split_across_chunks:
            evidence["status"] = "split_across_chunks"
            return StageTrace(
                stage_num=2,
                stage_name="Chunk",
                passed=False,
                failure_code=FailureCode.CHUNK_SPLIT,
                confidence=0.9,
                evidence=evidence,
                rank_table=None,
            )

        # Check if answer is present but diluted (in chunk but low density)
        # This would require checking token ratio
        if len(expected_answer) > chunk_size * 4 * 0.5:  # Answer longer than half chunk
            evidence["status"] = "answer_larger_than_chunk"
            return StageTrace(
                stage_num=2,
                stage_name="Chunk",
                passed=False,
                failure_code=FailureCode.CHUNK_SPLIT,
                confidence=0.8,
                evidence=evidence,
                rank_table=None,
            )

        evidence["status"] = "not_found_in_chunks"
        evidence["note"] = "Answer not found in any chunk - may be context missing or diluted"
        
        # Determine if it's dilution or context missing
        if len(chunks) > 0:
            max_chunk_tokens = max(c.token_count for c in chunks)
            if max_chunk_tokens > chunk_size * 1.5:
                failure_code = FailureCode.CHUNK_DILUTED
            else:
                failure_code = FailureCode.CONTEXT_MISSING
        else:
            failure_code = FailureCode.CHUNK_SPLIT

        return StageTrace(
            stage_num=2,
            stage_name="Chunk",
            passed=False,
            failure_code=failure_code,
            confidence=0.75,
            evidence=evidence,
            rank_table=None,
        )

    def _check_index_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 3: Check if chunk is in index and current."""
        evidence = {
            "check": "chunk_hash_in_index",
            "method": "verify_chunk_hash_exists",
        }

        if not expected_source:
            evidence["note"] = "No expected_source provided; cannot verify index"
            return StageTrace(
                stage_num=3,
                stage_name="Index",
                passed=True,
                failure_code=None,
                confidence=0.5,
                evidence=evidence,
                rank_table=None,
            )

        # Get the chunk that contains the answer
        parsed = self._load_parsed_for_source(expected_source)
        if not parsed:
            evidence["note"] = "Parsed document not available"
            return StageTrace(
                stage_num=3,
                stage_name="Index",
                passed=False,
                failure_code=FailureCode.INDEX_MISSING,
                confidence=0.6,
                evidence=evidence,
                rank_table=None,
            )

        content_type = parsed.content_type
        chunker_name = self.plan_config.get("chunker", {}).get(content_type.value, "heading_aware")
        chunk_size = self.plan_config.get("chunk_size", 512)
        overlap = self.plan_config.get("overlap", 0.1)

        chunker = self.chunker_registry.get_chunker(chunker_name)
        chunks = chunker.chunk(parsed, chunk_size, overlap)

        # Find chunk containing answer
        target_chunk = None
        for chunk in chunks:
            if self._fuzzy_contains(chunk.text, expected_answer or ""):
                target_chunk = chunk
                break

        if not target_chunk:
            evidence["note"] = "No chunk contains expected answer"
            return StageTrace(
                stage_num=3,
                stage_name="Index",
                passed=False,
                failure_code=FailureCode.INDEX_MISSING,
                confidence=0.7,
                evidence=evidence,
                rank_table=None,
            )

        # Check if chunk exists in vector store
        store = self.store_registry.create_store(
            StoreChoice(self.plan_config.get("store", "lancedb")), 
            path=Path(self.collection_path)
        )
        store.initialize("collection", self.embedder_registry.get(self.plan_config.get("embedder", "bge-small")).dimensions)

        stored_chunk = store.get_chunk(target_chunk.chunk_id)
        
        evidence["target_chunk_id"] = target_chunk.chunk_id
        evidence["target_chunk_tokens"] = target_chunk.token_count

        if stored_chunk:
            evidence["status"] = "found_in_index"
            return StageTrace(
                stage_num=3,
                stage_name="Index",
                passed=True,
                failure_code=None,
                confidence=0.95,
                evidence=evidence,
                rank_table=None,
            )
        
        # Check if it's stale (file hash mismatch)
        manifest = self.collection.get_manifest()
        if manifest:
            source_path = Path(expected_source)
            if not source_path.is_absolute():
                allowlist = get_path_allowlist()
                try:
                    source_path = allowlist.validate_read(source_path)
                except Exception:
                    pass
            
            if source_path.exists():
                current_hash = self._compute_file_hash(source_path)
                if current_hash in manifest.file_hashes:
                    if manifest.file_hashes[current_hash] != current_hash:
                        evidence["status"] = "stale"
                        evidence["note"] = "File hash mismatch - index stale"
                        return StageTrace(
                            stage_num=3,
                            stage_name="Index",
                            passed=False,
                            failure_code=FailureCode.INDEX_STALE,
                            confidence=0.9,
                            evidence=evidence,
                            rank_table=None,
                        )

        evidence["status"] = "missing_from_index"
        evidence["note"] = "Chunk not found in vector store"
        return StageTrace(
            stage_num=3,
            stage_name="Index",
            passed=False,
            failure_code=FailureCode.INDEX_MISSING,
            confidence=0.85,
            evidence=evidence,
            rank_table=None,
        )

    def _check_retrieve_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 4: Check retrieval rank in dense, BM25, hybrid."""
        try:
            results, _ = self.retrieval.search(question, k=100)

            rank_table = []
            for i, r in enumerate(results[:20]):
                rank_table.append({
                    "rank": i + 1,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "file": r.file_path,
                    "page": r.page,
                    "preview": r.text[:100],
                })

            expected_found = False
            expected_rank = None
            if expected_source:
                for i, r in enumerate(results):
                    if expected_source in r.file_path:
                        expected_found = True
                        expected_rank = i + 1
                        break

            evidence = {
                "check": "retrieval_rank",
                "hybrid_rank": expected_rank,
                "in_top_100": expected_found,
                "total_candidates": len(results),
                "rank_table": rank_table,
            }

            if expected_source and not expected_found:
                passed = False
                failure_code = FailureCode.RETRIEVAL_MISS
                confidence = 0.9
            elif expected_source and expected_rank and expected_rank > 8:
                passed = False
                failure_code = FailureCode.RANK_LOW
                confidence = 0.85
            else:
                passed = True
                failure_code = None
                confidence = 0.8

        except Exception as e:
            evidence = {"check": "retrieval_rank", "error": str(e)}
            passed = False
            failure_code = FailureCode.RETRIEVAL_MISS
            confidence = 0.5
            rank_table = None

        return StageTrace(
            stage_num=4,
            stage_name="Retrieve",
            passed=passed,
            failure_code=failure_code,
            confidence=confidence,
            evidence=evidence,
            rank_table=rank_table,
        )

    def _check_rerank_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 5: Check if reranker demoted the gold chunk."""
        evidence = {
            "check": "rerank_position_change",
            "method": "compare_pre_post_rerank_ranks",
        }

        # Run retrieval with and without reranking
        try:
            # Get pre-rerank (hybrid) results
            hybrid_results, _ = self.retrieval.hybrid_retriever.retrieve(question, None)
            
            # Get post-rerank results  
            reranked_results, _ = self.retrieval.search(question, k=20)
            
            # Compare ranks for gold chunk
            pre_rank = None
            post_rank = None
            
            if expected_source:
                for i, r in enumerate(hybrid_results):
                    if expected_source in r.chunk.file_path:
                        pre_rank = i + 1
                        break
                
                for i, r in enumerate(reranked_results):
                    if expected_source in r.file_path:
                        post_rank = i + 1
                        break
            
            evidence["pre_rerank_rank"] = pre_rank
            evidence["post_rerank_rank"] = post_rank
            
            if pre_rank and post_rank and post_rank > pre_rank:
                # Reranker demoted the gold chunk
                passed = False
                failure_code = FailureCode.RERANK_DEMOTED
                confidence = 0.8
                evidence["note"] = f"Gold chunk demoted from rank {pre_rank} to {post_rank}"
            else:
                passed = True
                failure_code = None
                confidence = 0.7
                evidence["note"] = "Reranker did not demote gold chunk" if pre_rank else "No expected source to verify"

        except Exception as e:
            evidence = {"check": "rerank_position_change", "error": str(e)}
            passed = False
            failure_code = FailureCode.RERANK_DEMOTED
            confidence = 0.5
            rank_table = None

        return StageTrace(
            stage_num=5,
            stage_name="Rerank",
            passed=passed,
            failure_code=failure_code,
            confidence=confidence,
            evidence=evidence,
            rank_table=None,
        )

    def _check_pack_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 6: Check if gold chunk survives token budget packing."""
        evidence = {
            "check": "gold_chunk_in_packed_context",
            "method": "simulate_context_packing",
        }

        max_tokens = self.plan_config.get("max_tokens_context", 1500)
        
        try:
            # Get full retrieval results
            results, _ = self.retrieval.search(question, k=20)
            
            # Simulate packing
            total_tokens = 0
            gold_chunk_packed = False
            
            for r in results:
                chunk_tokens = r.token_count if hasattr(r, 'token_count') else len(r.text) // 4
                if total_tokens + chunk_tokens <= max_tokens:
                    total_tokens += chunk_tokens
                    if expected_source and expected_source in r.file_path:
                        gold_chunk_packed = True
                else:
                    # Budget exceeded
                    if expected_source and expected_source in r.file_path:
                        evidence["status"] = "truncated"
                        evidence["total_tokens_before"] = total_tokens
                        evidence["chunk_tokens"] = chunk_tokens
                        evidence["budget"] = max_tokens
                        return StageTrace(
                            stage_num=6,
                            stage_name="Pack",
                            passed=False,
                            failure_code=FailureCode.BUDGET_TRUNCATED,
                            confidence=0.9,
                            evidence=evidence,
                            rank_table=None,
                        )
                    break
            
            if gold_chunk_packed:
                evidence["status"] = "packed"
                evidence["total_tokens_used"] = total_tokens
                evidence["budget"] = max_tokens
                return StageTrace(
                    stage_num=6,
                    stage_name="Pack",
                    passed=True,
                    failure_code=None,
                    confidence=0.9,
                    evidence=evidence,
                    rank_table=None,
                )
            
            # Gold chunk not in results at all (should have been caught earlier)
            evidence["status"] = "gold_chunk_not_in_candidates"
            return StageTrace(
                stage_num=6,
                stage_name="Pack",
                passed=False,
                failure_code=FailureCode.BUDGET_TRUNCATED,
                confidence=0.6,
                evidence=evidence,
                rank_table=None,
            )

        except Exception as e:
            evidence = {"check": "gold_chunk_in_packed_context", "error": str(e)}
            return StageTrace(
                stage_num=6,
                stage_name="Pack",
                passed=False,
                failure_code=FailureCode.BUDGET_TRUNCATED,
                confidence=0.5,
                evidence=evidence,
                rank_table=None,
            )

    def _check_downstream_stage(
        self, question: str, expected_answer: str | None, expected_source: str | None
    ) -> StageTrace:
        """Stage 7: Answer in context but generation failed."""
        evidence = {
            "check": "answer_in_final_context",
            "method": "verify_answer_text_in_packed_chunks",
        }

        # If we reach here, all retrieval stages passed
        # The failure is in the LLM generation
        evidence["status"] = "retrieval_succeeded"
        evidence["note"] = "All retrieval stages passed; failure is likely in LLM generation (prompt, model, temperature)"

        return StageTrace(
            stage_num=7,
            stage_name="Downstream",
            passed=True,
            failure_code=FailureCode.GENERATION_SIDE,
            confidence=0.9,
            evidence=evidence,
            rank_table=None,
        )

    def _build_result(
        self,
        traces: list[StageTrace],
        question: str,
        expected_answer: str | None,
        expected_source: str | None,
    ) -> DiagnosisResult:
        """Build final diagnosis result from traces."""
        first_failing = None
        failure_code = FailureCode.GENERATION_SIDE
        for trace in traces:
            if not trace.passed:
                first_failing = trace.stage_num
                failure_code = trace.failure_code or FailureCode.GENERATION_SIDE
                break

        if first_failing is None:
            first_failing = 7
            failure_code = FailureCode.GENERATION_SIDE

        relevant_traces = traces[:first_failing] if first_failing else traces
        confidence = sum(t.confidence for t in relevant_traces) / len(relevant_traces)

        fixes = self._generate_fixes(failure_code, traces)

        evidence_per_stage = [
            StageEvidence(
                stage=t.stage_num,
                stage_name=t.stage_name,
                passed=t.passed,
                failure_code=t.failure_code,
                confidence=t.confidence,
                evidence=t.evidence,
                rank_table=t.rank_table,
            )
            for t in traces
        ]

        return DiagnosisResult(
            first_failing_stage=first_failing,
            failure_code=failure_code,
            confidence=confidence,
            evidence_per_stage=evidence_per_stage,
            fixes=fixes,
            question=question,
            expected_answer=expected_answer,
            expected_source=expected_source,
        )

    def _generate_fixes(self, failure_code: FailureCode, traces: list[StageTrace]) -> list[FixProposal]:
        """Generate fix proposals based on failure code."""
        fixes = []

        fix_map = {
            FailureCode.PARSE_LOSS: [
                FixProposal(
                    target_stage=1,
                    description="Re-parse with Docling (enable OCR for scanned, formula/table extraction)",
                    config_change={"parser": "docling", "ocr": True, "formula_enrichment": True},
                    expected_impact="Recover text from scanned pages, preserve table structure and equations",
                    confidence=0.85,
                    risk="low",
                ),
                FixProposal(
                    target_stage=1,
                    description="Use cloud parser (LlamaParse) for best OCR quality",
                    config_change={"parser": "llamaparse", "cloud_ok": True},
                    expected_impact="Best-in-class OCR and layout detection",
                    confidence=0.9,
                    risk="medium",
                ),
            ],
            FailureCode.PARSE_LOSS_SCANNED_NO_OCR: [
                FixProposal(
                    target_stage=1,
                    description="Enable OCR in Docling parser",
                    config_change={"parser": "docling", "ocr": True},
                    expected_impact="Extract text from scanned/image-based PDFs",
                    confidence=0.9,
                    risk="low",
                ),
            ],
            FailureCode.PARSE_LOSS_TABLE_FLATTENED: [
                FixProposal(
                    target_stage=1,
                    description="Enable table structure extraction in Docling",
                    config_change={"parser": "docling", "table_structure": True},
                    expected_impact="Preserve table headers, rows, and cell relationships",
                    confidence=0.85,
                    risk="low",
                ),
                FixProposal(
                    target_stage=2,
                    description="Use table-per-chunk chunker to keep tables intact",
                    config_change={"chunker": "table_per_chunk"},
                    expected_impact="Each table becomes a retrievable unit with summary",
                    confidence=0.8,
                    risk="low",
                ),
            ],
            FailureCode.PARSE_LOSS_EQUATION_GARBLED: [
                FixProposal(
                    target_stage=1,
                    description="Enable formula enrichment in Docling",
                    config_change={"parser": "docling", "formula_enrichment": True},
                    expected_impact="Extract LaTeX representations of equations",
                    confidence=0.85,
                    risk="low",
                ),
                FixProposal(
                    target_stage=2,
                    description="Use formula-preserving chunker",
                    config_change={"chunker": "formula_preserving"},
                    expected_impact="Never split inside math formulas",
                    confidence=0.8,
                    risk="low",
                ),
            ],
            FailureCode.CHUNK_SPLIT: [
                FixProposal(
                    target_stage=2,
                    description="Increase chunk size to keep answer span intact",
                    config_change={"chunk_size": 800, "overlap": 0.15},
                    expected_impact="Longer chunks reduce split probability",
                    confidence=0.8,
                    risk="low",
                ),
                FixProposal(
                    target_stage=2,
                    description="Increase overlap to bridge chunk boundaries",
                    config_change={"overlap": 0.2},
                    expected_impact="Overlap ensures answer context spans chunks",
                    confidence=0.75,
                    risk="low",
                ),
            ],
            FailureCode.CHUNK_DILUTED: [
                FixProposal(
                    target_stage=2,
                    description="Decrease chunk size for more focused chunks",
                    config_change={"chunk_size": 256},
                    expected_impact="Smaller chunks increase answer density per chunk",
                    confidence=0.75,
                    risk="low",
                ),
            ],
            FailureCode.CONTEXT_MISSING: [
                FixProposal(
                    target_stage=2,
                    description="Enable heading-aware chunking to preserve document structure",
                    config_change={"chunker": "heading_aware", "prepend_heading": True},
                    expected_impact="Heading path prepended to each chunk for context",
                    confidence=0.8,
                    risk="low",
                ),
            ],
            FailureCode.INDEX_STALE: [
                FixProposal(
                    target_stage=3,
                    description="Re-index changed files (incremental)",
                    config_change={"build_index": {"resume": True, "force_reindex_changed": True}},
                    expected_impact="Update index to match current document state",
                    confidence=0.95,
                    risk="low",
                ),
            ],
            FailureCode.RETRIEVAL_MISS: [
                FixProposal(
                    target_stage=4,
                    description="Try different embedding model (e.g., bge-base, nomic)",
                    config_change={"embedder": "bge-base"},
                    expected_impact="Better semantic matching for this domain",
                    confidence=0.7,
                    risk="low",
                ),
                FixProposal(
                    target_stage=4,
                    description="Enable hybrid search (dense + BM25) if not already",
                    config_change={"retriever": "hybrid_rrf"},
                    expected_impact="BM25 catches exact keyword matches dense misses",
                    confidence=0.85,
                    risk="low",
                ),
                FixProposal(
                    target_stage=4,
                    description="Increase retrieve top-k before reranking",
                    config_change={"retrieve_top_k": 100},
                    expected_impact="More candidates for reranker to consider",
                    confidence=0.75,
                    risk="low",
                ),
                FixProposal(
                    target_stage=4,
                    description="Add metadata filters (content_type, file_path)",
                    config_change={"filters": {"content_type": "table"}},
                    expected_impact="Focus search on relevant content type",
                    confidence=0.7,
                    risk="low",
                ),
            ],
            FailureCode.RANK_LOW: [
                FixProposal(
                    target_stage=4,
                    description="Increase retrieve top-k to include gold chunk in candidate pool",
                    config_change={"retrieve_top_k": 100},
                    expected_impact="Gold chunk enters reranker candidate set",
                    confidence=0.8,
                    risk="low",
                ),
                FixProposal(
                    target_stage=5,
                    description="Swap reranker model (e.g., bge-reranker-large)",
                    config_change={"reranker": "bge-reranker-large"},
                    expected_impact="Better reranking precision",
                    confidence=0.7,
                    risk="low",
                ),
            ],
            FailureCode.RERANK_DEMOTED: [
                FixProposal(
                    target_stage=5,
                    description="Skip reranker for this content type",
                    config_change={"reranker": "none", "rerank_skip_types": ["code"]},
                    expected_impact="Avoid reranker bias against certain content",
                    confidence=0.7,
                    risk="low",
                ),
                FixProposal(
                    target_stage=5,
                    description="Rerank more candidates (increase rerank_top_k)",
                    config_change={"rerank_top_k": 20},
                    expected_impact="More candidates survive reranking",
                    confidence=0.65,
                    risk="low",
                ),
            ],
            FailureCode.BUDGET_TRUNCATED: [
                FixProposal(
                    target_stage=6,
                    description="Increase max token budget for context",
                    config_change={"max_tokens_context": 3000},
                    expected_impact="More chunks fit in context window",
                    confidence=0.9,
                    risk="low",
                ),
                FixProposal(
                    target_stage=2,
                    description="Decrease chunk size to fit more chunks in budget",
                    config_change={"chunk_size": 256},
                    expected_impact="More diverse chunks in same token budget",
                    confidence=0.75,
                    risk="low",
                ),
                FixProposal(
                    target_stage=6,
                    description="Deduplicate similar chunks before packing",
                    config_change={"deduplicate": True},
                    expected_impact="Remove redundant chunks, fit more unique content",
                    confidence=0.7,
                    risk="low",
                ),
            ],
            FailureCode.GENERATION_SIDE: [
                FixProposal(
                    target_stage=7,
                    description="Issue is in LLM generation, not retrieval. Check prompt, model, temperature.",
                    config_change={},
                    expected_impact="N/A - retrieval is working correctly",
                    confidence=0.9,
                    risk="none",
                ),
            ],
        }

        fixes = fix_map.get(failure_code, [])

        if failure_code != FailureCode.GENERATION_SIDE:
            fixes.append(FixProposal(
                target_stage=0,
                description="Run domain_eval to measure quality per content type and identify systematic issues",
                config_change={"run_eval": True},
                expected_impact="Identify if failure is systematic or one-off",
                confidence=0.6,
                risk="low",
            ))

        return fixes


def diagnose_failure(
    collection_path: str,
    question: str,
    expected_answer: str | None = None,
    expected_source: str | None = None,
    plan_config: dict | None = None,
) -> DiagnosisResult:
    """Main entry point for diagnose_failure tool."""
    if plan_config is None:
        plan_config = {
            "store": "lancedb",
            "embedder": "bge-small",
            "retrieve_top_k": 50,
            "rerank_top_k": 8,
            "rrf_k": 60,
            "reranker": "bge-reranker-base",
            "max_tokens_context": 1500,
        }

    diagnoser = FailureDiagnoser(collection_path, plan_config)
    return diagnoser.diagnose(question, expected_answer, expected_source)