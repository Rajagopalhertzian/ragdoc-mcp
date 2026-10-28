# Known limits (v0.1)

**Validation**
- Nothing here has been run on real-world documents. All benchmarks are synthetic.
- Confidence scores are heuristic constants, not calibrated probabilities.
- Table/equation detection in PDFs is heuristic (layout sample of the first 5 pages, math-symbol lines).

**Components not exercised in the build sandbox**
- Docling adapter, LlamaParse adapter, fastembed embedder/cross-encoder (no model downloads or cloud access). They follow documented entry points but are untested.
- macOS and Windows. A CI matrix is provided but was never run.
- Docker image (no Docker in the sandbox).

**Design limits**
- The default embedder is an offline hashing embedder: lexical-fuzzy, not semantic. Hybrid BM25 compensates, but recall on paraphrased queries is weak. Opt in to a neural embedder for semantic recall.
- Only LanceDB is implemented. Corpora above ~100k chunks get a warning; Qdrant/pgvector adapters are not included.
- Code chunking uses Python's `ast` for Python and a regex splitter for other languages (not tree-sitter).
- BM25 is rebuilt in memory when a collection is loaded (fast for tens of thousands of chunks, slow for very large ones).
- The build is single-process; estimates assume `workers=1`.
- Diagnosis can only see indexes this server built. Pointing it at an existing LangChain/LlamaIndex pipeline needs an adapter (not included).
- `diagnose_failure` needs a ground-truth answer or source. It cannot find the right answer for you.
- Two simultaneous builds of the same collection are not coordinated.

**Deliberate deviations from the original spec**
- MCP sampling for question generation was **removed**: it is deprecated in the current spec and the 2.x SDK moved server-initiated requests to a different mechanism that does not fit a loop. `domain_eval` instead accepts `questions=[...]` written by the calling LLM, and defaults to deterministic extractive questions. The "bring your own API key" generator was not built.
- `src/ragdoc/models.json` instead of `config/models.yaml` (avoids a YAML dependency; ships inside the package).
- `scaffold_project` is v1.1 in the spec and is not included.
- The server uses the 2.x SDK class `MCPServer` (the spec says `FastMCP`, which 2.x renamed).
