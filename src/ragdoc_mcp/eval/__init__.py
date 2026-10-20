"""Evaluation module for RAG Doctor MCP - domain_eval and verify_fix tools."""

import random
import uuid
import time
from typing import Any, Optional
from pathlib import Path

from ragdoc_mcp.chunkers import get_chunker_registry, Chunk
from ragdoc_mcp.embedders import get_embedder_registry
from ragdoc_mcp.indexer import CollectionManager
from ragdoc_mcp.parsers import get_parser_registry, ParsedDocument
from ragdoc_mcp.retrieval import RetrievalPipeline
from ragdoc_mcp.schemas import (
    ContentType,
    DomainEvalResult,
    FixProposal,
    RerankerChoice,
    RetrieverChoice,
    SearchResult,
    StoreChoice,
    VerifyFixResult,
    FailureCode,
)
from ragdoc_mcp.stores import get_store_registry
from ragdoc_mcp.security import get_path_allowlist


class DomainEvaluator:
    """Evaluates RAG quality per content type."""

    def __init__(self, collection_path: str, plan_config: dict):
        self.collection_path = Path(collection_path)
        self.plan_config = plan_config
        self.collection = CollectionManager(self.collection_path.name)
        self.retrieval = self._create_retrieval_pipeline()

    def _create_retrieval_pipeline(self) -> RetrievalPipeline:
        store_registry = get_store_registry()
        embedder_registry = get_embedder_registry()

        store_choice = self.plan_config.get("store", "lancedb")
        embedder_name = self.plan_config.get("embedder", "bge-small")

        store = store_registry.create_store(StoreChoice(store_choice), path=self.collection_path)
        store.initialize("collection", embedder_registry.get(embedder_name).dimensions)

        embedder = embedder_registry.get(embedder_name)

        return RetrievalPipeline(
            store=store,
            embedder=embedder,
            retriever_choice=RetrieverChoice.HYBRID_RRF,
            reranker_choice=RerankerChoice.CROSS_ENCODER,
            reranker_model=self.plan_config.get("reranker", "bge-reranker-base"),
            retrieve_top_k=self.plan_config.get("retrieve_top_k", 50),
            rerank_top_k=self.plan_config.get("rerank_top_k", 8),
            rrf_k=self.plan_config.get("rrf_k", 60),
            max_tokens_context=self.plan_config.get("max_tokens_context", 1500),
        )

    def evaluate(
        self,
        content_type: ContentType,
        n_per_type: int = 30,
        question_generator: str = "extractive",
    ) -> DomainEvalResult:
        """Evaluate retrieval quality for a specific content type."""
        # Generate test questions from the corpus
        questions = self._generate_questions(content_type, n_per_type, question_generator)

        if not questions:
            return DomainEvalResult(
                content_type=content_type,
                hit_at_k={1: 0.0, 3: 0.0, 5: 0.0, 8: 0.0},
                mrr=0.0,
                recall=0.0,
                n_questions=0,
                weakest_documents=[],
            )

        # Run retrieval for each question
        hits_at_k = {1: 0, 3: 0, 5: 0, 8: 0}
        reciprocal_ranks = []
        recalls = []
        doc_scores = {}

        for q_data in questions:
            question = q_data["question"]
            expected_chunk_id = q_data.get("expected_chunk_id")
            expected_file = q_data.get("expected_file")

            results, _ = self.retrieval.search(question, k=8)

            # Check hits
            found_rank = None
            for i, r in enumerate(results):
                if expected_chunk_id and r.chunk_id == expected_chunk_id:
                    found_rank = i + 1
                    break
                if expected_file and expected_file in r.file_path:
                    found_rank = i + 1
                    break

            if found_rank:
                for k in hits_at_k:
                    if found_rank <= k:
                        hits_at_k[k] += 1
                reciprocal_ranks.append(1.0 / found_rank)
                recalls.append(1.0)

                # Track per-document performance
                doc_key = expected_file or "unknown"
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"hits": 0, "total": 0}
                doc_scores[doc_key]["hits"] += 1
                doc_scores[doc_key]["total"] += 1
            else:
                reciprocal_ranks.append(0.0)
                recalls.append(0.0)
                doc_key = expected_file or "unknown"
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"hits": 0, "total": 0}
                doc_scores[doc_key]["total"] += 1

        n = len(questions)
        hit_at_k = {k: v / n for k, v in hits_at_k.items()}
        mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0
        recall = sum(recalls) / n if n > 0 else 0.0

        # Find weakest documents
        weakest = []
        for doc, scores in doc_scores.items():
            if scores["total"] > 0:
                hit_rate = scores["hits"] / scores["total"]
                if hit_rate < 0.5:  # Less than 50% hit rate
                    weakest.append({
                        "document": doc,
                        "hit_rate": hit_rate,
                        "questions_tested": scores["total"],
                    })

        weakest.sort(key=lambda x: x["hit_rate"])
        weakest = weakest[:5]  # Top 5 weakest

        return DomainEvalResult(
            eval_id=str(uuid.uuid4())[:8],
            content_type=content_type,
            hit_at_k=hit_at_k,
            mrr=mrr,
            recall=recall,
            n_questions=n,
            weakest_documents=weakest,
        )

    def _generate_questions(
        self, content_type: ContentType, n: int, method: str
    ) -> list[dict]:
        """Generate test questions for a content type by extracting from stored chunks."""
        # Load chunks from collection
        chunks = self.collection.load_chunks()
        
        # Filter chunks by content type
        type_chunks = [c for c in chunks if c.content_type == content_type]
        
        if not type_chunks:
            return []
        
        questions = []
        
        if method == "extractive":
            # Extract key phrases as questions
            questions = self._extractive_question_generation(type_chunks, n)
        elif method == "llm":
            # Would use MCP sampling or BYOK LLM
            questions = self._llm_question_generation(type_chunks, n)
        else:
            questions = self._extractive_question_generation(type_chunks, n)
        
        return questions[:n]

    def _extractive_question_generation(self, chunks: list[Chunk], n: int) -> list[dict]:
        """Generate questions by extracting key phrases from chunks."""
        import re
        
        questions = []
        
        for chunk in chunks:
            if len(questions) >= n:
                break
            
            text = chunk.text
            if not text or len(text) < 50:
                continue
            
            # Extract sentences with key indicators
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 200:
                    continue
                
                # Look for factual statements (has numbers, specific terms, etc.)
                has_fact = any([
                    re.search(r'\d+', sentence),  # Numbers
                    re.search(r'\b(is|was|has|have|equals|defined as|configured as)\b', sentence, re.I),
                    re.search(r'\b(timeout|limit|threshold|rate|size|version)\b', sentence, re.I),
                ])
                
                if has_fact:
                    # Convert statement to question
                    question = self._statement_to_question(sentence)
                    if question:
                        questions.append({
                            "question": question,
                            "expected_chunk_id": chunk.chunk_id,
                            "expected_file": chunk.file_path,
                            "content_type": chunk.content_type.value,
                        })
                        if len(questions) >= n:
                            break
        
        # If not enough extractive questions, create simple ones
        while len(questions) < n and len(chunks) > 0:
            chunk = random.choice(chunks)
            questions.append({
                "question": f"What does the document say about {chunk.file_path}?",
                "expected_chunk_id": chunk.chunk_id,
                "expected_file": chunk.file_path,
                "content_type": chunk.content_type.value,
            })
        
        return questions

    def _llm_question_generation(self, chunks: list[Chunk], n: int) -> list[dict]:
        """Generate questions using LLM (placeholder for MCP sampling)."""
        # Would use MCP sampling here
        # For now, fall back to extractive
        return self._extractive_question_generation(chunks, n)

    def _statement_to_question(self, statement: str) -> str | None:
        """Convert a factual statement to a question."""
        import re
        
        # Simple patterns
        patterns = [
            (r'^(\w+) is (\d+)', r'What is the \1?'),
            (r'^(\w+) was (\d+)', r'What was the \1?'),
            (r'^(\w+) has (\d+)', r'How many \1 does it have?'),
            (r'^(\w+) equals (\d+)', r'What does \1 equal?'),
            (r'^(\w+) configured as (\d+)', r'What is the \1 configured as?'),
            (r'^The (\w+) is (\d+)', r'What is the \1?'),
            (r'^(\w+) = (\d+)', r'What is \1?'),
        ]
        
        for pattern, replacement in patterns:
            match = re.match(pattern, statement, re.I)
            if match:
                return re.sub(pattern, replacement, statement, flags=re.I)
        
        # Generic fallback
        if len(statement) > 10:
            return f"What does the document say about {statement[:50]}...?"
        
        return None


class FixVerifier:
    """Verifies proposed fixes by running on shadow config/index."""

    def __init__(self, collection_path: str, plan_config: dict):
        self.collection_path = Path(collection_path)
        self.plan_config = plan_config.copy()
        self.collection = CollectionManager(self.collection_path.name)

    def verify(
        self,
        fix: FixProposal,
        test_cases: list[dict],
        regression_cases: list[dict] | None = None,
        apply: bool = False,
    ) -> VerifyFixResult:
        """Verify a fix by applying to shadow config and re-running test cases."""
        # Create shadow config with fix applied
        shadow_config = self._apply_fix(self.plan_config, fix)

        # Run before/after evaluation
        before_metrics = self._run_evaluation(self.plan_config, test_cases)
        after_metrics = self._run_evaluation(shadow_config, test_cases)

        regression_metrics = {}
        if regression_cases:
            before_reg = self._run_evaluation(self.plan_config, regression_cases)
            after_reg = self._run_evaluation(shadow_config, regression_cases)
            regression_metrics = {
                "before_hit_rate": before_reg.get("hit_rate", 0),
                "after_hit_rate": after_reg.get("hit_rate", 0),
                "regression": after_reg.get("hit_rate", 0) < before_reg.get("hit_rate", 0),
            }

        # Determine if fix should be promoted
        improved = after_metrics.get("hit_rate", 0) > before_metrics.get("hit_rate", 0)
        no_regression = not regression_metrics.get("regression", False)
        promoted = improved and no_regression and apply

        return VerifyFixResult(
            fix_id=fix.fix_id,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            regression_check=regression_metrics,
            promoted=promoted,
            apply_confirmed=apply,
        )

    def _apply_fix(self, config: dict, fix: FixProposal) -> dict:
        """Apply fix config changes to create shadow config."""
        import copy
        shadow = copy.deepcopy(config)
        
        def deep_update(d, updates):
            for k, v in updates.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        
        deep_update(shadow, fix.config_change)
        return shadow

    def _run_evaluation(self, config: dict, test_cases: list[dict]) -> dict[str, float]:
        """Run evaluation on test cases with given config."""
        if not test_cases:
            return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "avg_latency_ms": 0.0}

        # Create retrieval pipeline with config
        store_registry = get_store_registry()
        embedder_registry = get_embedder_registry()

        store_choice = config.get("store", "lancedb")
        embedder_name = config.get("embedder", "bge-small")

        store = store_registry.create_store(StoreChoice(store_choice), path=self.collection_path)
        store.initialize("collection", embedder_registry.get(embedder_name).dimensions)

        embedder = embedder_registry.get(embedder_name)

        retrieval = RetrievalPipeline(
            store=store,
            embedder=embedder,
            retriever_choice=RetrieverChoice.HYBRID_RRF,
            reranker_choice=RerankerChoice.CROSS_ENCODER,
            reranker_model=config.get("reranker", "bge-reranker-base"),
            retrieve_top_k=config.get("retrieve_top_k", 50),
            rerank_top_k=config.get("rerank_top_k", 8),
            rrf_k=config.get("rrf_k", 60),
            max_tokens_context=config.get("max_tokens_context", 1500),
        )

        # Run test cases
        hits_at_k = {1: 0, 3: 0, 5: 0, 8: 0}
        reciprocal_ranks = []
        recalls = []
        latencies = []

        for tc in test_cases:
            question = tc.get("question", "")
            expected_chunk_id = tc.get("expected_chunk_id")
            expected_file = tc.get("expected_file")

            if not question:
                continue

            start = time.perf_counter()
            results, _ = retrieval.search(question, k=8)
            latencies.append((time.perf_counter() - start) * 1000)

            found_rank = None
            for i, r in enumerate(results):
                if expected_chunk_id and r.chunk_id == expected_chunk_id:
                    found_rank = i + 1
                    break
                if expected_file and expected_file in r.file_path:
                    found_rank = i + 1
                    break

            if found_rank:
                for k in hits_at_k:
                    if found_rank <= k:
                        hits_at_k[k] += 1
                reciprocal_ranks.append(1.0 / found_rank)
                recalls.append(1.0)
            else:
                reciprocal_ranks.append(0.0)
                recalls.append(0.0)

        n = len([tc for tc in test_cases if tc.get("question")])
        if n == 0:
            return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "avg_latency_ms": 0.0}

        hit_rate = sum(1 for r in recalls if r > 0) / n
        mrr = sum(reciprocal_ranks) / n
        recall = sum(recalls) / n
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "recall": recall,
            "avg_latency_ms": avg_latency,
            "hit_at_1": hits_at_k[1] / n,
            "hit_at_3": hits_at_k[3] / n,
            "hit_at_5": hits_at_k[5] / n,
            "hit_at_8": hits_at_k[8] / n,
        }


def domain_eval(
    collection_path: str,
    content_types: list[ContentType],
    n_per_type: int = 30,
    plan_config: dict | None = None,
) -> list[DomainEvalResult]:
    """Run domain evaluation for multiple content types."""
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

    evaluator = DomainEvaluator(collection_path, plan_config)
    results = []

    for ctype in content_types:
        result = evaluator.evaluate(ctype, n_per_type)
        results.append(result)

    return results


def verify_fix(
    collection_path: str,
    fix: FixProposal,
    test_cases: list[dict],
    regression_cases: list[dict] | None = None,
    plan_config: dict | None = None,
    apply: bool = False,
) -> VerifyFixResult:
    """Verify a proposed fix."""
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

    verifier = FixVerifier(collection_path, plan_config)
    return verifier.verify(fix, test_cases, regression_cases, apply)