"""Planner module for RAG Doctor MCP - profile_corpus and plan_rag tools."""

import os
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ragdoc_mcp.chunkers import get_chunker_registry
from ragdoc_mcp.config import get_defaults, get_scale_tier, load_models_config
from ragdoc_mcp.embedders import get_embedder_registry
from ragdoc_mcp.parsers import get_parser_registry
from ragdoc_mcp.schemas import (
    ChunkerChoice,
    ContentType,
    CorpusProfile,
    CostEstimate,
    DecisionLogEntry,
    HardwareTarget,
    ParserChoice,
    PrivacyMode,
    RagConstraints,
    RagPlan,
    RerankerChoice,
    RetrieverChoice,
    StageConfig,
    StoreChoice,
)
from ragdoc_mcp.stores import StoreChoice, get_store_registry


class CorpusProfiler:
    """Profiles a document corpus to extract characteristics."""

    def __init__(self, sample_size: int = 200):
        self.sample_size = sample_size
        self.parser_registry = get_parser_registry()

    def profile(self, path: str | Path) -> CorpusProfile:
        """Profile a corpus directory."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Find all files
        files = list(path.rglob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith('.')]

        if not files:
            raise ValueError(f"No files found in {path}")

        # Sample files
        import random
        random.shuffle(files)
        sample_files = files[:self.sample_size]

        # Analyze sample
        type_counts = dict.fromkeys(ContentType, 0)
        total_pages = 0
        total_size = 0
        scanned_pages = 0
        equation_pages = 0
        table_pages = 0
        code_pages = 0
        languages = set()

        for file_path in sample_files:
            try:
                parsed = self.parser_registry.parse(file_path)
                type_counts[parsed.content_type] = type_counts.get(parsed.content_type, 0) + 1
                total_pages += len(parsed.pages)
                total_size += file_path.stat().st_size

                # Check for scanned content (low text extraction)
                for page in parsed.pages:
                    text = page.get("text", "")
                    if len(text.strip()) < 50:  # Likely scanned
                        scanned_pages += 1
                    # Check for equations
                    if page.get("equations") or '\\begin{' in text or '$' in text:
                        equation_pages += 1
                    # Check for tables
                    if page.get("tables") or '|' in text and text.count('|') > 4:
                        table_pages += 1
                    # Check for code
                    if parsed.content_type == ContentType.CODE:
                        code_pages += 1
                        lang = page.get("metadata", {}).get("language")
                        if lang:
                            languages.add(lang)

            except Exception:
                # Skip files that can't be parsed
                continue

        sampled = len(sample_files)
        total_files = len(files)

        # Extrapolate to full corpus
        scale_factor = total_files / sampled if sampled > 0 else 1
        est_total_pages = int(total_pages * scale_factor)

        return CorpusProfile(
            profile_id=str(uuid.uuid4())[:8],
            path=str(path),
            total_files=total_files,
            total_pages=est_total_pages,
            total_size_bytes=int(total_size * scale_factor),
            type_mix=type_counts,
            scanned_share=scanned_pages / max(total_pages, 1),
            equation_density=equation_pages / max(total_pages, 1),
            table_density=table_pages / max(total_pages, 1),
            code_density=code_pages / max(total_pages, 1),
            languages=sorted(languages),
            avg_doc_length_pages=est_total_pages / max(total_files, 1),
            sample_size=sampled,
        )


class RagPlanner:
    """Plans RAG pipeline based on corpus profile and constraints."""

    def __init__(self):
        self.config = load_models_config()
        self.defaults = get_defaults()

    def plan(self, profile: CorpusProfile, constraints: RagConstraints | None = None) -> RagPlan:
        """Create a RAG pipeline plan."""
        if constraints is None:
            constraints = RagConstraints()

        decision_log = []

        # 1. Choose parsers per content type
        parser_choices = self._choose_parsers(profile, constraints, decision_log)

        # 2. Choose chunkers per content type
        chunker_choices = self._choose_chunkers(profile, decision_log)

        # 3. Choose chunk size and overlap
        chunk_size, overlap = self._choose_chunk_params(profile, decision_log)

        # 4. Choose embedder
        embedder = self._choose_embedder(profile, constraints, decision_log)

        # 5. Choose store
        store = self._choose_store(profile, constraints, decision_log)

        # 6. Choose retriever and reranker
        retriever, reranker = self._choose_retrieval(profile, constraints, decision_log)

        # 7. Set retrieval params
        retrieve_top_k = self.defaults.retrieve_top_k
        rerank_top_k = self.defaults.rerank_top_k
        rrf_k = self.defaults.rrf_k

        # 8. Estimate cost and time
        cost_estimate = self._estimate_cost(profile, parser_choices, embedder, constraints)

        config = StageConfig(
            parser=parser_choices,
            chunker=chunker_choices,
            chunk_size=chunk_size,
            overlap=overlap,
            embedder=embedder,
            store=store,
            retriever=retriever,
            reranker=reranker,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            rrf_k=rrf_k,
        )

        return RagPlan(
            plan_id=str(uuid.uuid4())[:8],
            profile_id=profile.profile_id,
            constraints=constraints,
            config=config,
            cost_estimate=cost_estimate,
            decision_log=decision_log,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def _choose_parsers(
        self, profile: CorpusProfile, constraints: RagConstraints, log: list
    ) -> dict[ContentType, ParserChoice]:
        choices = {}

        for ctype in ContentType:
            count = profile.type_mix.get(ctype, 0)
            if count == 0:
                continue

            if ctype in (ContentType.PDF, ContentType.EQUATION, ContentType.TABLE):
                if profile.scanned_share > 0.3 and constraints.privacy == PrivacyMode.LOCAL_ONLY:
                    choice = ParserChoice.DOCLING  # Docling with OCR
                    reason = "High scanned share, local-only: Docling with OCR"
                elif profile.scanned_share > 0.3 and constraints.privacy == PrivacyMode.CLOUD_OK:
                    choice = ParserChoice.LLAMAPARSE  # Cloud parser for scanned
                    reason = "High scanned share, cloud allowed: LlamaParse for best OCR"
                else:
                    choice = ParserChoice.DOCLING
                    reason = "PDF with tables/equations: Docling for layout/formula extraction"
            elif ctype == ContentType.CODE:
                choice = ParserChoice.FAST
                reason = "Source code: direct read, no parsing needed"
            else:
                choice = ParserChoice.FAST
                reason = "Clean text/markdown: fast parser sufficient"

            choices[ctype] = choice
            log.append(DecisionLogEntry(
                stage="parser",
                choice=choice.value,
                reason=reason,
                alternatives_considered=[p.value for p in ParserChoice if p != choice]
            ))

        return choices

    def _choose_chunkers(
        self, profile: CorpusProfile, log: list
    ) -> dict[ContentType, ChunkerChoice]:
        choices = {}

        for ctype in ContentType:
            count = profile.type_mix.get(ctype, 0)
            if count == 0:
                continue

            if ctype == ContentType.TABLE:
                choice = ChunkerChoice.TABLE_PER_CHUNK
                reason = "Tables: keep intact with summary for retrieval"
            elif ctype == ContentType.EQUATION:
                choice = ChunkerChoice.FORMULA_PRESERVING
                reason = "Equations: never split inside formula, keep context"
            elif ctype == ContentType.CODE:
                choice = ChunkerChoice.AST_CODE
                reason = "Code: AST-based chunking by function/class with imports"
            elif ctype == ContentType.MARKDOWN:
                choice = ChunkerChoice.HEADING_AWARE
                reason = "Markdown: heading-aware preserves document structure"
            else:
                choice = ChunkerChoice.HEADING_AWARE
                reason = "Prose/text: heading-aware with overlap"

            choices[ctype] = choice
            log.append(DecisionLogEntry(
                stage="chunker",
                choice=choice.value,
                reason=reason,
                alternatives_considered=[c.value for c in ChunkerChoice if c != choice]
            ))

        return choices

    def _choose_chunk_params(self, profile: CorpusProfile, log: list) -> tuple[int, float]:
        # Base on content types present
        has_code = profile.type_mix.get(ContentType.CODE, 0) > 0
        has_equations = profile.type_mix.get(ContentType.EQUATION, 0) > 0
        has_tables = profile.type_mix.get(ContentType.TABLE, 0) > 0

        if has_code:
            chunk_size = 800  # Code needs more context
            overlap = 0.15
            reason = "Code present: larger chunks (800) with 15% overlap for function context"
        elif has_equations:
            chunk_size = 800
            overlap = 0.15
            reason = "Equations present: larger chunks (800) with 15% overlap for formula context"
        elif has_tables:
            chunk_size = 512
            overlap = 0.10
            reason = "Tables present: medium chunks (512) with 10% overlap"
        else:
            chunk_size = 512
            overlap = 0.10
            reason = "General prose: default 512 tokens with 10% overlap"

        log.append(DecisionLogEntry(
            stage="chunk_params",
            choice=f"size={chunk_size}, overlap={overlap}",
            reason=reason,
            alternatives_considered=[f"size={s}, overlap={o}" for s in [256, 512, 800, 1200] for o in [0, 0.1, 0.15]]
        ))

        return chunk_size, overlap

    def _choose_embedder(
        self, profile: CorpusProfile, constraints: RagConstraints, log: list
    ) -> str:
        has_code = profile.type_mix.get(ContentType.CODE, 0) > 0
        has_equations = profile.type_mix.get(ContentType.EQUATION, 0) > 0

        if has_code:
            embedder = "jina-code"
            reason = "Code-heavy corpus: jina-embeddings-v2-base-code for code retrieval"
        elif has_equations:
            embedder = "bge-base"
            reason = "Equation-heavy: bge-base-en-v1.5 for better semantic understanding"
        elif constraints.hardware == HardwareTarget.CPU:
            embedder = "bge-small"
            reason = "CPU only: bge-small-en-v1.5 for speed"
        else:
            embedder = "bge-base"
            reason = "Balanced: bge-base-en-v1.5 for quality/speed tradeoff"

        log.append(DecisionLogEntry(
            stage="embedder",
            choice=embedder,
            reason=reason,
            alternatives_considered=["bge-small", "bge-base", "bge-large", "nomic", "e5-small", "jina-code"]
        ))

        return embedder

    def _choose_store(
        self, profile: CorpusProfile, constraints: RagConstraints, log: list
    ) -> StoreChoice:
        est_chunks = profile.total_pages * 10  # rough estimate

        tier = get_scale_tier(est_chunks)

        if tier.name == "small":
            store = StoreChoice.LANCEDB
            reason = f"Small scale (~{est_chunks} chunks): LanceDB local, zero-config"
        elif tier.name == "medium":
            store = StoreChoice.QDRANT
            reason = f"Medium scale (~{est_chunks} chunks): Qdrant for performance"
        else:
            store = StoreChoice.QDRANT
            reason = f"Large scale (~{est_chunks} chunks): Qdrant cluster recommended"

        log.append(DecisionLogEntry(
            stage="store",
            choice=store.value,
            reason=reason,
            alternatives_considered=[s.value for s in StoreChoice if s != store]
        ))

        return store

    def _choose_retrieval(
        self, profile: CorpusProfile, constraints: RagConstraints, log: list
    ) -> tuple[RetrieverChoice, str | None]:
        # Always use hybrid RRF for best results
        retriever = RetrieverChoice.HYBRID_RRF
        reranker = "bge-reranker-base"

        reason = "Hybrid RRF (dense+BM25) with cross-encoder reranker for best recall/precision"

        log.append(DecisionLogEntry(
            stage="retrieval",
            choice=f"{retriever.value} + {reranker}",
            reason=reason,
            alternatives_considered=["dense only", "bm25 only", "hybrid no rerank"]
        ))

        return retriever, reranker

    def _estimate_cost(
        self, profile: CorpusProfile, parser_choices: dict, embedder: str, constraints: RagConstraints
    ) -> CostEstimate:
        # Sample measurement (simulated - in reality would run on sample)
        # For now, use heuristic estimates

        pages = profile.total_pages
        est_chunks = pages * 10

        # Parse rate estimates (pages/sec)
        parse_rates = {
            ParserChoice.FAST: 50,
            ParserChoice.DOCLING: 5,
            ParserChoice.LLAMAPARSE: 2,  # network limited
        }

        # Weighted average parse rate
        total_weight = sum(profile.type_mix.values())
        weighted_rate = 0
        for ctype, count in profile.type_mix.items():
            if count == 0:
                continue
            parser = parser_choices.get(ctype, ParserChoice.FAST)
            rate = parse_rates.get(parser, 10)
            weighted_rate += rate * (count / total_weight)

        parse_time_min = pages / max(weighted_rate, 1) / 60

        # Embedding rate (chunks/sec)
        embed_rates = {
            "bge-small": 200,
            "bge-base": 100,
            "bge-large": 50,
            "nomic": 80,
            "e5-small": 150,
            "jina-code": 100,
        }
        embed_rate = embed_rates.get(embedder, 100)
        embed_time_min = est_chunks / embed_rate / 60

        # Cloud parser cost
        cloud_parser_cost = 0.0
        if constraints.privacy == PrivacyMode.CLOUD_OK:
            # Estimate LlamaParse cost
            scanned_pages = int(pages * profile.scanned_share)
            cloud_parser_cost = scanned_pages * 0.003  # ~$0.003/page

        total_time = parse_time_min + embed_time_min + 5  # 5 min overhead

        return CostEstimate(
            parse_time_minutes=round(parse_time_min, 1),
            embed_time_minutes=round(embed_time_min, 1),
            total_time_minutes=round(total_time, 1),
            estimated_cost_usd=round(cloud_parser_cost, 2),
            cloud_parser_cost_usd=round(cloud_parser_cost, 2),
            sample_size_pages=min(50, pages),
            measured_parse_rate_pages_per_sec=round(weighted_rate, 1),
            measured_embed_rate_chunks_per_sec=embed_rate,
        )


def create_plan_from_path(
    path: str,
    privacy: str = "local_only",
    budget_usd: float | None = None,
    hardware: str = "cpu",
    latency_target_ms: int | None = None,
    sample_size: int = 200,
) -> RagPlan:
    """Convenience function to create plan from path."""
    profiler = CorpusProfiler(sample_size=sample_size)
    profile = profiler.profile(path)

    constraints = RagConstraints(
        privacy=PrivacyMode(privacy),
        budget_usd=budget_usd,
        hardware=HardwareTarget(hardware),
        latency_target_ms=latency_target_ms,
    )

    planner = RagPlanner()
    return planner.plan(profile, constraints)
