# RAG Doctor MCP

**Local-first MCP server for diagnosing and fixing retrieval pipelines over technical documents.**

[![PyPI](https://img.shields.io/pypi/v/ragdoc-mcp)](https://pypi.org/project/ragdoc-mcp/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

RAG Doctor helps AI agents and developers build, tune, and debug RAG pipelines over technical documents (PDFs with equations and tables, source code, manuals). Its distinctive feature is **cross-stage diagnosis**: when a RAG answer is wrong, it identifies which stage failed (parsing, chunking, indexing, retrieval, reranking, context packing) and proposes a fix that it can verify.

## Why RAG Doctor?

Existing MCP tools each cover one stage: chunking tuners, retrieval evaluators, parser routers. Agents still lack a tool that answers: **why did this question fail, and what exactly should I change?**

RAG Doctor finds the broken stage and proves the fix.

## Features

| Tool | Purpose |
|------|---------|
| `profile_corpus` | Sample files → type mix, scanned %, equation/table/code density, languages, pages |
| `plan_rag` | Profile + constraints → joint plan (parser/chunker/embedder/store/retriever) with measured time/cost estimates and decision log |
| `build_index` | Execute plan: resumable, batched, cached by file hash |
| `search` | Hybrid (dense+BM25 RRF) + cross-encoder rerank → chunks with citations |
| `diagnose_failure` | Trace question through all 7 stages → first failing stage, failure code, evidence, ranked fixes |
| `verify_fix` | Apply fix to shadow config/index → re-run failing + regression cases → before/after metrics |
| `domain_eval` | Measure hit@k, MRR, recall per content type (prose, table, equation, code) |

## Quick Start

```bash
# Install (requires Python 3.10+)
pip install ragdoc-mcp
# or with uv
uvx ragdoc-mcp

# Profile a corpus
ragdoc profile_corpus --path ./my-docs --sample-size 200

# Create a plan
ragdoc plan_rag --path ./my-docs --privacy local_only --hardware cpu

# Build index (after reviewing plan)
ragdoc build_index --plan-id <plan_id>

# Search
ragdoc search --collection my-docs --query "How does the retry logic work?" --k 8

# Diagnose a failure
ragdoc diagnose_failure --collection my-docs --question "What is the timeout?" --expected-source "config/retry.py"
```

## MCP Client Configuration

### Claude Code / Claude Desktop

Add to your MCP config:

```json
{
  "mcpServers": {
    "ragdoc": {
      "command": "uvx",
      "args": ["ragdoc-mcp"],
      "env": {
        "RAGDOC_BASE_DIR": "~/.ragdoc"
      }
    }
  }
}
```

### Hermes Agent

```bash
hermes mcp add ragdoc --command uvx --args ragdoc-mcp --env RAGDOC_BASE_DIR=~/.ragdoc
```

## Architecture

```
src/ragdoc/
├── server.py          # FastMCP server with 7 tools + resources + prompts
├── parsers/           # Fast (pypdf), Docling (layout/tables/formulas/OCR), LlamaParse (cloud)
├── chunkers/          # Heading-aware, token window, table-per-chunk, formula-preserving, AST code
├── embedders/         # fastembed (bge, nomic, e5, jina-code) + sentence-transformers
├── stores/            # LanceDB (default), Qdrant, pgvector adapters
├── retrieval/         # Hybrid RRF + cross-encoder rerank
├── diagnose/          # 7-stage tracer with evidence and fix proposals
├── planner/           # profile_corpus + plan_rag with decision log
├── eval/              # domain_eval + verify_fix
├── security/          # Path allowlist, untrusted text wrapper, instruction detection
├── config/            # Models YAML, settings, scale tiers
└── schemas/           # Pydantic models for all tool I/O
```

## Security

- **Local-first**: No network calls by default. Cloud parsers/LLMs need explicit consent.
- **Path allowlist**: Only folders under `RAGDOC_BASE_DIR` are readable; writes only inside collection dirs.
- **Untrusted text**: All document text labeled as untrusted; instruction-like patterns flagged.
- **No shell/eval/arbitrary URLs**: Secrets only from environment, never logged.
- **Dependency hygiene**: Pinned versions, pip-audit in CI, SBOM at release.

## Development

```bash
# Clone and setup
git clone https://github.com/your-org/ragdoc-mcp
cd ragdoc-mcp

# Install dev dependencies
uv sync --dev

# Run tests
pytest

# Lint
ruff check .
mypy src

# Type check
mypy src/ragdoc
```

## Project Structure

```
ragdoc-mcp/
├── src/ragdoc/           # Main package
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test documents and failure fixtures
├── skills/rag-doctor/    # Claude Code skill
├── docs/                 # Documentation
├── .github/workflows/    # CI/CD
├── pyproject.toml        # Package config
├── server.json           # MCP registry metadata
├── Dockerfile            # Multi-stage Docker image
├── LICENSE               # MIT
└── README.md             # This file
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap

- **v1.1**: `scaffold_project` (tested templates for LlamaIndex, LangChain, plain Python + Dockerfile/FastAPI), remote HTTP mode
- **v2**: Large-scale resumable ingestion with Qdrant/pgvector, drift monitoring, multi-user

## Citation

If you use RAG Doctor in research, please cite:

```bibtex
@software{ragdoc-mcp,
  title = {RAG Doctor MCP: Cross-Stage Diagnosis for Retrieval Pipelines},
  year = {2026},
  url = {https://github.com/your-org/ragdoc-mcp}
}
```