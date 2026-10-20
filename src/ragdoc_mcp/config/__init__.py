"""Configuration loader for RAG Doctor MCP."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EmbeddingModel(BaseModel):
    name: str
    dimensions: int
    max_tokens: int
    speed: str
    quality: str
    license: str
    recommended_for: list[str]
    fastembed_id: str


class RerankerModel(BaseModel):
    name: str
    max_tokens: int
    speed: str
    quality: str
    license: str
    recommended_for: list[str]
    fastembed_id: str


class ScaleTier(BaseModel):
    name: str
    max_chunks: int
    recommended_store: str
    description: str


class Defaults(BaseModel):
    chunk_sizes: list[int]
    overlaps: list[float]
    retrieve_top_k: int
    rerank_top_k: int
    rrf_k: int
    max_tokens_context: int


class ModelsConfig(BaseModel):
    embedding_models: list[EmbeddingModel]
    reranker_models: list[RerankerModel]
    scale_tiers: list[ScaleTier]
    defaults: Defaults


_config: ModelsConfig | None = None


def load_models_config(path: str | Path | None = None) -> ModelsConfig:
    """Load models configuration from YAML file."""
    global _config
    if _config is not None:
        return _config

    if path is None:
        path = Path(__file__).parent / "models.yaml"

    with open(path) as f:
        data = yaml.safe_load(f)

    _config = ModelsConfig(**data)
    return _config


def get_embedding_model(name: str) -> EmbeddingModel | None:
    """Get embedding model by name or fastembed_id."""
    config = load_models_config()
    for model in config.embedding_models:
        if model.name == name or model.fastembed_id == name:
            return model
    return None


def get_reranker_model(name: str) -> RerankerModel | None:
    """Get reranker model by name or fastembed_id."""
    config = load_models_config()
    for model in config.reranker_models:
        if model.name == name or model.fastembed_id == name:
            return model
    return None


def get_scale_tier(max_chunks: int) -> ScaleTier:
    """Get recommended scale tier for chunk count."""
    config = load_models_config()
    for tier in config.scale_tiers:
        if max_chunks <= tier.max_chunks:
            return tier
    return config.scale_tiers[-1]


def get_defaults() -> Defaults:
    """Get default configuration values."""
    return load_models_config().defaults


# Environment-based settings
class Settings(BaseModel):
    """Runtime settings from environment variables."""

    base_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("RAGDOC_BASE_DIR", Path.home() / ".ragdoc"))
    )
    log_level: str = Field(default=os.getenv("RAGDOC_LOG_LEVEL", "INFO"))
    enable_telemetry: bool = Field(default=os.getenv("RAGDOC_TELEMETRY", "false").lower() == "true")
    default_embedding_model: str = Field(default=os.getenv("RAGDOC_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    default_reranker_model: str = Field(default=os.getenv("RAGDOC_RERANKER_MODEL", "BAAI/bge-reranker-base"))
    max_file_size_mb: int = Field(default=int(os.getenv("RAGDOC_MAX_FILE_SIZE_MB", "100")))
    max_concurrent_parses: int = Field(default=int(os.getenv("RAGDOC_MAX_CONCURRENT_PARSES", "4")))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
