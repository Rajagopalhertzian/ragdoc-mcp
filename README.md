<!-- mcp-name: io.github.YOUR-USER/ragdoc-mcp -->
# ragdoc-mcp — RAG Doctor

A **local-first MCP server that finds which stage of a RAG pipeline failed** (parse → chunk → index → retrieve → rerank → pack), proves it with evidence, and **verifies the fix on a shadow copy before touching anything**.

Most RAG tooling gives you scores. When an answer is wrong, you still have to guess *why*. RAG Doctor reads the parsed text, the stored chunks, the vector index and a second parser backend, and tells your agent things like:

> `CONTEXT_MISSING` at the chunk stage — the question's key terms (`KQ-401`) appear only in the heading path, which is not part of the indexed chunk text. Fix: prepend the heading path. Verified on a shadow copy: 5/5 target questions fixed, 0/60 regressions.

It also builds the index (parse, chunk, embed, LanceDB + BM25, hybrid retrieval, rerank, token-budget packing), so you can start from a folder of documents.

> **Status: v0.1, alpha.** Read [What is and isn't verified](#what-is-and-isnt-verified) before relying on it.

## Install

```bash
pip install ragdoc-mcp                  # offline defaults, no model downloads
pip install "ragdoc-mcp[neural]"        # optional: fastembed neural embedder (opt-in, downloads a model once)
pip install "ragdoc-mcp[docling]"       # optional: Docling parser for scans, tables, equations
ragdoc-mcp --check                      # prints environment info
```

Requires Python 3.10+ and the MCP Python SDK 2.x (`mcp>=2.3,<3`). Until it is on PyPI: `pip install git+https://github.com/YOUR-USER/ragdoc-mcp`.

### Connect a client

`RAGDOC_BASE_DIR` is the only folder the server may read, and where it keeps its indexes (`<base>/.ragdoc/`). It defaults to the server's working directory.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ragdoc": {
      "command": "uvx",
      "args": ["ragdoc-mcp"],
      "env": { "RAGDOC_BASE_DIR": "/absolute/path/to/your/project" }
    }
  }
}
```

**Claude Code:** `claude mcp add ragdoc -e RAGDOC_BASE_DIR=/absolute/path/to/project -- uvx ragdoc-mcp` (check `claude mcp add --help` for your version).

**HTTP** (for a gateway you control): `ragdoc-mcp --transport streamable-http --host 127.0.0.1 --port 8000`. It binds to loopback by default and has **no authentication**; put auth in front before exposing it.

## Tools

| Tool | Use it to | Changes data? |
|---|---|---|
| `profile_corpus` | Sample a folder: type mix, scanned share, table/equation/code density, language, pages | No |
| `plan_rag` | Choose parser/chunker/embedder/retriever, each with a reason, plus a *measured* time and size estimate | Saves the plan only |
| `build_index` | Build the collection. Resumable, batched, cached by file hash, reports progress | Writes inside the collection dir |
| `search` | Hybrid dense + BM25 + rerank. Cited chunks with per-stage scores, token-budget packing | No |
| `diagnose_failure` | **Name the first failing stage** with evidence and fixes ordered by impact | No |
| `verify_fix` | Test a fix on a shadow copy with a regression check. Promotes only if `apply=true` and nothing regressed | Only when `apply=true`; keeps a backup |
| `domain_eval` | hit@k, MRR, context recall per content type, failure-cause breakdown, weakest documents | Writes a report in the collection |

Resources: `ragdoc://collections`, `ragdoc://plan/{id}`, `ragdoc://report/{id}`. Prompts: `build-rag`, `debug-rag`.

Every tool returns `{status, summary, data, next_step}`, capped near 2,000 tokens (pass `detail_level="full"` for more). Errors always say what to do next.

### Typical session (values are illustrative)

```
profile_corpus(path="docs")            → "412 files (~3,100 pages); signals: tables, equations, mixed"
plan_rag(profile_id=...)               → plan + estimate: "~38k chunks, ~9 min" (needs your approval if large)
build_index(plan_id=..., approve=true)
search(collection=..., query=...)      → cited chunks; the answer is wrong?
diagnose_failure(collection, question, expected_answer="max pressure of 42 bar")
verify_fix(collection, fix=<fixes[0]>, cases=[<case>])         # shadow copy, nothing changes
verify_fix(..., apply=true)                                    # only if verdict = improves_no_regression
```

Install the bundled skill ([`skills/rag-doctor/SKILL.md`](skills/rag-doctor/SKILL.md)) so the agent follows this order every time. The core rule: **diagnose before changing any parameter.**

## Failure codes

| Stage | Codes |
|---|---|
| parse | `PARSE_LOSS` (sub-causes: `scanned_no_ocr`, `table_flattened`, `equation_garbled`, `column_order`) |
| chunk | `CHUNK_SPLIT`, `CHUNK_DILUTED`, `CONTEXT_MISSING` |
| index | `INDEX_STALE`, `INDEX_MISSING` |
| retrieve | `RETRIEVAL_MISS`, `RANK_LOW` |
| rerank | `RERANK_DEMOTED` |
| pack | `BUDGET_TRUNCATED` |
| downstream | `GENERATION_SIDE` — the answer *is* in the returned context; look at the prompt or model |

Rules the diagnoser follows: never blame a stage without evidence; compare digits and units exactly; blame a chunk-shape or index-gap cause only if retrieval actually failed; lower the confidence when only `expected_source` is known (capped at 0.5).

## Safety

- **Read access is confined** to `RAGDOC_BASE_DIR` (symlinks resolved; escapes are skipped or rejected). Writes happen only inside `<base>/.ragdoc/`.
- **Document text is untrusted.** Search results are labeled `untrusted`, carry a notice, and instruction-like text ("ignore previous instructions…") is flagged but never stripped or obeyed.
- **No network by default.** No telemetry. Neural models and cloud parsers are opt-in; cloud parsing needs `constraints.privacy="cloud_ok"`, an API key from the environment, and consent recorded in the manifest.
- Strict argument validation, size/rate/time limits, structured logs to **stderr** with a request id.

See [SECURITY.md](SECURITY.md).

## What is and isn't verified

Measured in the build sandbox (Python 3.12, Linux, 1 CPU, offline hashing embedder). Reproduce with `benchmarks/`.

| Claim | Result | Caveat |
|---|---|---|
| Test suite | 96 passing, including real PDFs, a real scanned PDF, an MCP client over in-process and **real stdio** | Linux only |
| Stage attribution on the fixture suite | **84/86 = 97.7%**; 0/8 false blame on healthy controls; parse sub-cause 16/16 | **Synthetic data, and the diagnoser was developed against these same fault injectors.** A development-set score, not an independent test |
| `CHUNK_DILUTED` | **0/2** (both named `RERANK_DEMOTED`; the reranker *did* demote the chunk) | n too small to conclude; label arguable. Not tuned away |
| `verify_fix` end to end | 26/29 improve with no regressions (89.7%) | The verifier was revised after a first run scored 62% (regression set was biased); not an untouched test. The 3 misses are anchorless questions no config change can fix |
| `plan_rag` vs naive baseline | Beats baseline on 3 of 4 types (hit@8 1.00 vs 0.96, MRR 1.00 vs 0.82) | Plan scores 1.0 everywhere: the benchmark is at its ceiling, synthetic, keyword-style questions |
| Confidence scores | **Not calibrated** (0.87 mean when right, 0.90 when wrong) | Treat as an ordering hint, not a probability |
| Docling, LlamaParse, fastembed adapters | **Not exercised** (no model downloads or cloud access in the sandbox) | Written to documented entry points; verify before relying on them |
| macOS / Windows | **Not tested** (CI file included) | Directory promotion in `verify_fix` may behave differently on Windows |

**Not validated on real-world documents at all.** Run `domain_eval` on your own corpus. See [docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && pytest
python benchmarks/run_attribution.py --docs 60 --per-code 8 --verify-per-code 3
python benchmarks/compare_baseline.py
```

Layout: `src/ragdoc/{parsers,chunkers,embedders,stores,retrieval,diagnose,planner,eval}`, plus `server.py`. How fixtures are labeled: [docs/FIXTURES.md](docs/FIXTURES.md). Publishing and directory listings: [docs/LISTING.md](docs/LISTING.md).

## License

MIT
