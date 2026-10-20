"""RAG Doctor MCP Server - Main entry point with all MCP tools."""

import json
from pathlib import Path

from fastmcp import FastMCP

from ragdoc_mcp.config import settings
from ragdoc_mcp.diagnose import diagnose_failure
from ragdoc_mcp.eval import domain_eval, verify_fix
from ragdoc_mcp.indexer import CollectionManager, build_index as indexer_build_index
from ragdoc_mcp.planner import CorpusProfiler, create_plan_from_path
from ragdoc_mcp.retrieval import RetrievalPipeline, create_retrieval_pipeline
from ragdoc_mcp.schemas import (
    ContentType,
    DetailLevel,
    FixProposal,
    RagPlan,
    ToolResult,
    ToolStatus,
)
from ragdoc_mcp.security import get_path_allowlist, init_path_allowlist, wrap_untrusted_text
from ragdoc_mcp.stores import get_store_registry, StoreChoice
from ragdoc_mcp.embedders import get_embedder_registry

# Initialize FastMCP server
mcp = FastMCP("ragdoc-mcp")

# In-memory plan store (in production, would persist to disk)
_plan_store: dict[str, RagPlan] = {}


# Tool: profile_corpus
@mcp.tool(
    name="profile_corpus",
    description="Profile a document corpus: sample files and report type mix, scanned share, equation/table/code density, languages, total pages, and average document length. Read-only.",
)
async def profile_corpus(
    path: str,
    sample_size: int = 200,
    detail_level: DetailLevel = DetailLevel.BRIEF,
) -> ToolResult:
    """Profile a document corpus."""
    try:
        # Validate path
        allowlist = get_path_allowlist()
        validated_path = allowlist.validate_read(Path(path))

        profiler = CorpusProfiler(sample_size=sample_size)
        profile = profiler.profile(validated_path)

        data = profile.model_dump()
        if detail_level == DetailLevel.BRIEF:
            # Reduce output size
            data.pop("type_mix", None)

        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Profiled {profile.total_files} files (~{profile.total_pages} pages) at {path}. "
                    f"Type mix: {', '.join(f'{k.value}:{v}' for k,v in profile.type_mix.items() if v>0)}. "
                    f"Scanned: {profile.scanned_share:.0%}, Equations: {profile.equation_density:.0%}, "
                    f"Tables: {profile.table_density:.0%}, Code: {profile.code_density:.0%}.",
            data=data,
            next_step="plan_rag",
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Failed to profile corpus: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: plan_rag
@mcp.tool(
    name="plan_rag",
    description="Create a RAG pipeline plan from a corpus profile or path. Constraints: privacy (local_only/cloud_ok), budget_usd, hardware (cpu/gpu), latency_target_ms. Returns joint plan with parser/chunker/embedder/store/retriever choices, measured time/cost estimates, and decision log with reasons.",
)
async def plan_rag(
    profile_id: str | None = None,
    path: str | None = None,
    privacy: str = "local_only",
    budget_usd: float | None = None,
    hardware: str = "cpu",
    latency_target_ms: int | None = None,
    detail_level: DetailLevel = DetailLevel.BRIEF,
) -> ToolResult:
    """Create a RAG pipeline plan."""
    try:
        if not profile_id and not path:
            return ToolResult(
                status=ToolStatus.ERROR,
                summary="Must provide either profile_id or path",
                data={"error": "Missing required parameter: profile_id or path"},
                detail_level=detail_level,
            )

        if path:
            allowlist = get_path_allowlist()
            validated_path = allowlist.validate_read(Path(path))
            plan = create_plan_from_path(
                str(validated_path),
                privacy=privacy,
                budget_usd=budget_usd,
                hardware=hardware,
                latency_target_ms=latency_target_ms,
            )
            
            # Store plan with its profile_id for later lookup
            _plan_store[plan.plan_id] = plan
        else:
            # Would load profile by ID - for now require path
            return ToolResult(
                status=ToolStatus.ERROR,
                summary="profile_id lookup not yet implemented; provide path instead",
                data={"error": "profile_id not supported yet"},
                detail_level=detail_level,
            )

        data = plan.model_dump()
        if detail_level == DetailLevel.BRIEF:
            # Reduce decision log verbosity
            data["decision_log"] = [
                {"stage": d.stage, "choice": d.choice, "reason": d.reason}
                for d in plan.decision_log
            ]

        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Created plan {plan.plan_id} for {plan.profile_id}. "
                    f"Parser: {', '.join(f'{k.value}:{v.value}' for k,v in plan.config.parser.items())}. "
                    f"Chunker: {', '.join(f'{k.value}:{v.value}' for k,v in plan.config.chunker.items())}. "
                    f"Embedder: {plan.config.embedder}, Store: {plan.config.store.value}. "
                    f"Est. time: {plan.cost_estimate.total_time_minutes:.0f} min, "
                    f"Cost: ${plan.cost_estimate.estimated_cost_usd:.2f}.",
            data=data,
            next_step="build_index",
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Failed to create plan: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: build_index
@mcp.tool(
    name="build_index",
    description="Execute a RAG plan to build the index. Resumable, batched, cached by file hash. Reports progress. Writes only inside collection directory.",
)
async def build_index(
    plan_id: str,
    collection: str,
    resume: bool = True,
    detail_level: DetailLevel = DetailLevel.BRIEF,
) -> ToolResult:
    """Build index from plan."""
    try:
        # Load plan from store
        plan = _plan_store.get(plan_id)
        if plan is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                summary=f"Plan {plan_id} not found",
                data={"error": "Plan not found"},
                detail_level=detail_level,
            )
        
        # Validate collection path
        allowlist = get_path_allowlist()
        collection_dir = allowlist.get_collection_dir(collection)
        
        # Get corpus path from plan profile
        corpus_path = plan.profile_id  # This should be the original path
        
        # For now, we need the corpus path - would come from profile store
        # Using a simplified approach
        return ToolResult(
            status=ToolStatus.PARTIAL,
            summary="build_index requires corpus path. Use plan_rag with path then build_index with plan_id and collection name.",
            data={
                "plan_id": plan_id,
                "collection": collection,
                "resume": resume,
                "note": "Implementation needs corpus path from profile store"
            },
            next_step="search",
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Failed to build index: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: search
@mcp.tool(
    name="search",
    description="Hybrid retrieval (dense + BM25 fused with RRF) plus cross-encoder rerank. Returns chunks with citations (file, page, line/heading), per-stage scores, and truncation notes. Results labeled as untrusted document text.",
)
async def search(
    collection: str,
    query: str,
    k: int = 8,
    filters: dict | None = None,
    max_tokens: int = 1500,
    detail_level: DetailLevel = DetailLevel.BRIEF,
) -> ToolResult:
    """Search a collection."""
    try:
        # Validate collection path
        allowlist = get_path_allowlist()
        collection_dir = allowlist.get_collection_dir(collection)
        
        # Load manifest to get plan config
        manifest_path = collection_dir / "manifest.json"
        if not manifest_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                summary=f"Collection {collection} not found or not initialized",
                data={"error": "Collection manifest not found"},
                detail_level=detail_level,
            )
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        plan_config = manifest.get("plan", {}).get("config", {})
        
        # Create retrieval pipeline
        store_choice = plan_config.get("store", "lancedb")
        embedder_name = plan_config.get("embedder", "bge-small")
        retrieve_top_k = plan_config.get("retrieve_top_k", 50)
        rerank_top_k = plan_config.get("rerank_top_k", 8)
        rrf_k = plan_config.get("rrf_k", 60)
        reranker_model = plan_config.get("reranker", "bge-reranker-base")
        
        store = get_store_registry().create_store(StoreChoice(store_choice), path=collection_dir)
        store.initialize("collection", get_embedder_registry().get(embedder_name).dimensions)
        
        retrieval = create_retrieval_pipeline(
            store=store,
            embedder_name=embedder_name,
            collection_path=str(collection_dir),
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            rrf_k=rrf_k,
            reranker_model=reranker_model,
            max_tokens_context=max_tokens,
        )
        
        results, timings = retrieval.search(query, k=k, filters=filters)
        
        data = {
            "results": [r.model_dump() for r in results],
            "query": query,
            "total_candidates": len(results),
            "retrieval_time_ms": timings.get("retrieval_time_ms", 0),
            "rerank_time_ms": timings.get("rerank_time_ms", 0),
        }
        
        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Found {len(results)} results for query: {query}",
            data=data,
            next_step="diagnose_failure",
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Search failed: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: diagnose_failure
@mcp.tool(
    name="diagnose_failure",
    description="Use when a RAG answer is wrong, missing, or low quality. Traces the question through every stage (parse, chunk, index, retrieve, rerank, pack, downstream) and returns the first failing stage, failure code, confidence, evidence (snippets, ranks per stage), and fixes ordered by expected impact.",
)
async def diagnose_failure_tool(
    collection: str,
    question: str,
    expected_answer: str | None = None,
    expected_source: str | None = None,
    detail_level: DetailLevel = DetailLevel.FULL,
) -> ToolResult:
    """Diagnose a RAG failure."""
    try:
        allowlist = get_path_allowlist()
        collection_dir = allowlist.get_collection_dir(collection)

        # Default plan config (would load from collection manifest)
        plan_config = {
            "store": "lancedb",
            "embedder": "bge-small",
            "retrieve_top_k": 50,
            "rerank_top_k": 8,
            "rrf_k": 60,
            "reranker": "bge-reranker-base",
            "max_tokens_context": 1500,
        }

        result = diagnose_failure(
            collection_path=str(collection_dir),
            question=question,
            expected_answer=expected_answer,
            expected_source=expected_source,
            plan_config=plan_config,
        )

        data = result.model_dump()
        if detail_level == DetailLevel.BRIEF:
            data.pop("evidence_per_stage", None)
            data.pop("fixes", None)

        # Wrap any document text in results as untrusted
        for evidence in data.get("evidence_per_stage", []):
            if "evidence" in evidence and isinstance(evidence["evidence"], dict):
                for key, value in evidence["evidence"].items():
                    if isinstance(value, str) and len(value) > 100:
                        evidence["evidence"][key] = wrap_untrusted_text(value, "diagnosis_evidence")

        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Diagnosis complete. First failing stage: {result.first_failing_stage} ({result.failure_code.value}). "
                    f"Confidence: {result.confidence:.0%}. {len(result.fixes)} fix(es) proposed.",
            data=data,
            next_step="verify_fix" if result.fixes else None,
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Diagnosis failed: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: verify_fix
@mcp.tool(
    name="verify_fix",
    description="Apply a proposed fix to a shadow copy of the config/index, re-run failing cases plus a regression sample, return before/after metrics. Promotes the change only if caller confirms (apply=true).",
)
async def verify_fix_tool(
    collection: str,
    fix: dict,  # FixProposal as dict
    cases: list[dict],
    regression_cases: list[dict] | None = None,
    apply: bool = False,
    detail_level: DetailLevel = DetailLevel.FULL,
) -> ToolResult:
    """Verify a proposed fix."""
    try:
        allowlist = get_path_allowlist()
        collection_dir = allowlist.get_collection_dir(collection)

        plan_config = {
            "store": "lancedb",
            "embedder": "bge-small",
            "retrieve_top_k": 50,
            "rerank_top_k": 8,
            "rrf_k": 60,
            "reranker": "bge-reranker-base",
            "max_tokens_context": 1500,
        }

        fix_proposal = FixProposal(**fix)
        result = verify_fix(
            collection_path=str(collection_dir),
            fix=fix_proposal,
            test_cases=cases,
            regression_cases=regression_cases,
            plan_config=plan_config,
            apply=apply,
        )

        data = result.model_dump()

        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Fix verification complete. Before hit_rate: {result.before_metrics.get('hit_rate', 0):.2f}, "
                    f"After: {result.after_metrics.get('hit_rate', 0):.2f}. "
                    f"Promoted: {result.promoted}. Apply confirmed: {result.apply_confirmed}.",
            data=data,
            next_step=None,
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Fix verification failed: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Tool: domain_eval
@mcp.tool(
    name="domain_eval",
    description="Measure retrieval quality per content type (prose, table, equation, code). Builds test questions from corpus, scores hit@k, MRR, recall per type, lists weakest documents.",
)
async def domain_eval_tool(
    collection: str,
    types: list[ContentType],
    n_per_type: int = 30,
    detail_level: DetailLevel = DetailLevel.FULL,
) -> ToolResult:
    """Run domain evaluation."""
    try:
        allowlist = get_path_allowlist()
        collection_dir = allowlist.get_collection_dir(collection)

        plan_config = {
            "store": "lancedb",
            "embedder": "bge-small",
            "retrieve_top_k": 50,
            "rerank_top_k": 8,
            "rrf_k": 60,
            "reranker": "bge-reranker-base",
            "max_tokens_context": 1500,
        }

        results = domain_eval(
            collection_path=str(collection_dir),
            content_types=types,
            n_per_type=n_per_type,
            plan_config=plan_config,
        )

        data = {"results": [r.model_dump() for r in results]}

        if detail_level == DetailLevel.BRIEF:
            for r in data["results"]:
                r.pop("weakest_documents", None)

        # Summary
        summaries = []
        for r in results:
            summaries.append(f"{r.content_type.value}: hit@5={r.hit_at_k.get(5,0):.2f}, MRR={r.mrr:.2f}, recall={r.recall:.2f}")

        return ToolResult(
            status=ToolStatus.OK,
            summary=f"Domain evaluation complete for {len(results)} content type(s). " + "; ".join(summaries),
            data=data,
            next_step=None,
            detail_level=detail_level,
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            summary=f"Domain evaluation failed: {e}",
            data={"error": str(e)},
            detail_level=detail_level,
        )


# Resources
@mcp.resource("ragdoc://collections")
async def list_collections() -> str:
    """List all collections."""
    allowlist = get_path_allowlist()
    collections_dir = allowlist.base_dir / "collections"
    if not collections_dir.exists():
        return json.dumps({"collections": []})

    collections = []
    for d in collections_dir.iterdir():
        if d.is_dir():
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                collections.append({
                    "name": d.name,
                    "created_at": manifest.get("created_at"),
                    "total_chunks": manifest.get("total_chunks", 0),
                    "plan_id": manifest.get("plan", {}).get("plan_id"),
                })

    return json.dumps({"collections": collections})


@mcp.resource("ragdoc://plan/{plan_id}")
async def get_plan(plan_id: str) -> str:
    """Get plan by ID."""
    # Would load from plan store
    return json.dumps({"error": "Plan lookup not implemented", "plan_id": plan_id})


@mcp.resource("ragdoc://report/{report_id}")
async def get_report(report_id: str) -> str:
    """Get evaluation/diagnosis report by ID."""
    return json.dumps({"error": "Report lookup not implemented", "report_id": report_id})


# Prompts
@mcp.prompt("build-rag")
async def build_rag_prompt(corpus_path: str) -> str:
    """Prompt for building a RAG system."""
    return f"""You are helping build a RAG system for the corpus at {corpus_path}.

Follow this workflow:
1. Call profile_corpus(path="{corpus_path}") to understand the document types and characteristics
2. Call plan_rag(path="{corpus_path}", privacy="local_only") to get a tailored pipeline plan
3. Review the plan's decision log - each choice has a reason
4. Call build_index(plan_id="<returned_plan_id>") to execute the plan
5. Test with search(collection="<collection_name>", query="your test question")

The plan will recommend parsers, chunkers, embedder, store, and retrieval strategy based on your corpus profile and constraints."""


@mcp.prompt("debug-rag")
async def debug_rag_prompt(collection: str, failing_question: str) -> str:
    """Prompt for debugging a RAG failure."""
    return f"""You are debugging a RAG failure in collection "{collection}".

The question that failed: "{failing_question}"

Follow this workflow:
1. Call diagnose_failure(collection="{collection}", question="{failing_question}", expected_answer="...", expected_source="...")
   - Provide expected_answer if you know what the answer should be
   - Provide expected_source (file path) if you know where the answer should come from
2. Review the diagnosis: first_failing_stage, failure_code, confidence, and evidence per stage
3. Pick a fix from the proposed fixes (ordered by expected impact)
4. Call verify_fix(collection="{collection}", fix=<chosen_fix>, cases=[{{"question": "{failing_question}", "expected_source": "..."}}], regression_cases=[...], apply=false)
5. If verification shows improvement without regression, re-run with apply=true

The diagnosis will tell you exactly which stage lost the answer and why."""


def main():
    """Main entry point."""
    # Initialize path allowlist
    init_path_allowlist(settings.base_dir)

    # Run MCP server
    mcp.run()


if __name__ == "__main__":
    main()
