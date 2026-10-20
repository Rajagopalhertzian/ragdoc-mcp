"""Core data models and schemas for RAG Doctor MCP tools."""

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"


class ContentType(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    UNKNOWN = "unknown"


class DetailLevel(str, Enum):
    BRIEF = "brief"
    FULL = "full"


class ToolResult(BaseModel):
    """Standard response format for all MCP tools."""
    status: ToolStatus
    summary: str = Field(description="One or two sentence summary")
    data: dict[str, Any] = Field(default_factory=dict)
    next_step: str | None = Field(default=None, description="Suggested next tool call")
    detail_level: DetailLevel = Field(default=DetailLevel.BRIEF)


class CorpusProfile(BaseModel):
    """Profile of a document corpus."""
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    path: str
    total_files: int
    total_pages: int
    total_size_bytes: int
    type_mix: dict[ContentType, int]
    scanned_share: float = Field(ge=0.0, le=1.0)
    equation_density: float = Field(ge=0.0, le=1.0)
    table_density: float = Field(ge=0.0, le=1.0)
    code_density: float = Field(ge=0.0, le=1.0)
    languages: list[str]
    avg_doc_length_pages: float
    sample_size: int


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_OK = "cloud_ok"


class HardwareTarget(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class RagConstraints(BaseModel):
    """Constraints for pipeline planning."""
    privacy: PrivacyMode = PrivacyMode.LOCAL_ONLY
    budget_usd: float | None = None
    hardware: HardwareTarget = HardwareTarget.CPU
    latency_target_ms: int | None = None


class ParserChoice(str, Enum):
    FAST = "fast"
    DOCLING = "docling"
    LLAMAPARSE = "llamaparse"


class ChunkerChoice(str, Enum):
    HEADING_AWARE = "heading_aware"
    TOKEN_WINDOW = "token_window"
    TABLE_PER_CHUNK = "table_per_chunk"
    FORMULA_PRESERVING = "formula_preserving"
    AST_CODE = "ast_code"


class StoreChoice(str, Enum):
    LANCEDB = "lancedb"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class RetrieverChoice(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"


class RerankerChoice(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    NONE = "none"


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""
    parser: dict[ContentType, ParserChoice]
    chunker: dict[ContentType, ChunkerChoice]
    chunk_size: int
    overlap: float
    embedder: str
    store: StoreChoice
    retriever: RetrieverChoice
    reranker: str | None = None
    retrieve_top_k: int = 50
    rerank_top_k: int = 8
    rrf_k: int = 60


class CostEstimate(BaseModel):
    """Time and cost estimate for full-scale indexing."""
    parse_time_minutes: float
    embed_time_minutes: float
    total_time_minutes: float
    estimated_cost_usd: float
    cloud_parser_cost_usd: float = 0.0
    sample_size_pages: int
    measured_parse_rate_pages_per_sec: float
    measured_embed_rate_chunks_per_sec: float


class DecisionLogEntry(BaseModel):
    """Single entry in the decision log."""
    stage: str
    choice: str
    reason: str
    alternatives_considered: list[str]


class RagPlan(BaseModel):
    """Complete RAG pipeline plan."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    profile_id: str
    constraints: RagConstraints
    config: StageConfig
    cost_estimate: CostEstimate
    decision_log: list[DecisionLogEntry]
    created_at: str  # ISO format


class SearchResult(BaseModel):
    """Single search result with citations."""
    chunk_id: str
    text: str
    score: float
    file_path: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    heading_path: str | None = None
    content_type: ContentType
    stage_scores: dict[str, float] = Field(default_factory=dict)
    truncated: bool = False
    untrusted: bool = True


class SearchResponse(BaseModel):
    """Response from search tool."""
    results: list[SearchResult]
    query: str
    total_candidates: int
    retrieval_time_ms: float
    rerank_time_ms: float


class FailureCode(str, Enum):
    PARSE_LOSS = "PARSE_LOSS"
    PARSE_LOSS_SCANNED_NO_OCR = "PARSE_LOSS_SCANNED_NO_OCR"
    PARSE_LOSS_TABLE_FLATTENED = "PARSE_LOSS_TABLE_FLATTENED"
    PARSE_LOSS_EQUATION_GARBLED = "PARSE_LOSS_EQUATION_GARBLED"
    PARSE_LOSS_COLUMN_ORDER = "PARSE_LOSS_COLUMN_ORDER"
    CHUNK_SPLIT = "CHUNK_SPLIT"
    CHUNK_DILUTED = "CHUNK_DILUTED"
    CONTEXT_MISSING = "CONTEXT_MISSING"
    INDEX_STALE = "INDEX_STALE"
    INDEX_MISSING = "INDEX_MISSING"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    RANK_LOW = "RANK_LOW"
    RERANK_DEMOTED = "RERANK_DEMOTED"
    BUDGET_TRUNCATED = "BUDGET_TRUNCATED"
    GENERATION_SIDE = "GENERATION_SIDE"


class StageEvidence(BaseModel):
    """Evidence for a single stage in diagnosis."""
    stage: int
    stage_name: str
    passed: bool
    failure_code: FailureCode | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    rank_table: list[dict] | None = None


class FixProposal(BaseModel):
    """Proposed fix for a diagnosed failure."""
    fix_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_stage: int
    description: str
    config_change: dict[str, Any]
    expected_impact: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk: str = "low"


class DiagnosisResult(BaseModel):
    """Result from diagnose_failure tool."""
    first_failing_stage: int
    failure_code: FailureCode
    confidence: float
    evidence_per_stage: list[StageEvidence]
    fixes: list[FixProposal]
    question: str
    expected_answer: str | None = None
    expected_source: str | None = None


class VerifyFixResult(BaseModel):
    """Result from verify_fix tool."""
    fix_id: str
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    regression_check: dict[str, float]
    promoted: bool
    apply_confirmed: bool


class DomainEvalResult(BaseModel):
    """Result from domain_eval tool."""
    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content_type: ContentType
    hit_at_k: dict[int, float]
    mrr: float
    recall: float
    n_questions: int
    weakest_documents: list[dict[str, Any]]


class CollectionManifest(BaseModel):
    """Manifest for a RAG collection."""
    collection_name: str
    created_at: str
    plan: RagPlan
    version: int = 1
    consent_flags: dict[str, bool] = Field(default_factory=dict)
    file_hashes: dict[str, str] = Field(default_factory=dict)
    total_chunks: int = 0
    index_path: str
