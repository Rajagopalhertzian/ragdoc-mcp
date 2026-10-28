# Publishing and listing checklist

mcpmarket.com details below were read from https://mcpmarket.com/submit on 2026-10-27. Sites change; re-read the page before you pay or submit. The page does **not** state the review criteria.

## 0. Before anything: replace the placeholders
- `YOUR-USER` in `README.md` (the `mcp-name:` line and install URL) and `pyproject.toml` (`Homepage`).
- Confirm the PyPI name is still free: `pip index versions ragdoc-mcp` (it was free on 2026-10-27).
- Run the checks: `ruff check src tests && pytest && python benchmarks/run_attribution.py`.
- Decide whether you are comfortable with the honest "What is and isn't verified" table in the README. Keep it: reviewers and users will test your claims.

## 1. GitHub (mcpmarket.com requires a public GitHub repository URL)
```bash
git init && git add . && git commit -m "ragdoc-mcp v0.1.0"
git branch -M main
git remote add origin https://github.com/<you>/ragdoc-mcp.git
git push -u origin main
git tag v0.1.0 && git push --tags
```
Then create a GitHub Release for `v0.1.0`. Add repo topics: `mcp`, `model-context-protocol`, `rag`, `retrieval`, `debugging`, `lancedb`.

## 2. PyPI
```bash
python -m pip install build twine
python -m build            # check dist/ contains a wheel and an sdist
twine upload dist/*        # or use the included publish.yml with PyPI "trusted publishing"
```
After this, `uvx ragdoc-mcp` in the README works.

## 3. mcpmarket.com (what you asked about)
1. Open **https://mcpmarket.com/submit**.
2. Pick the **MCP Server** tab, then the **GitHub repo** option (the other option on that page is *Remote MCP*, for hosted servers).
3. Paste the full repository URL, e.g. `https://github.com/<you>/ragdoc-mcp`.
4. Enter an email you read: the page says they email you when the listing goes live.
5. Choose how it goes live:

| | Free Queue | Get Listed Now |
|---|---|---|
| Price | $0 | $29 one-time (no subscription) |
| Time | average 4–6 weeks | within 24 hours |
| Badge | none | an "Official" badge on the tile and page (per the page) |
| Placement | standard | not specified beyond the badge |
| "Try Now" link to your site | no | yes (optional) |

   The free queue is a fine default for a v0.1. The paid tier buys speed and a badge, not a guaranteed approval (the page says listings are reviewed).
6. Optional, separate submission: the **Agent Skill** tab lists skills. This repo has one at `skills/rag-doctor/SKILL.md`. The page does not say how a skill repo must be laid out; ask `support@mcpmarket.com` if the submission is rejected.

What reviewers will see first is your GitHub README. Lead with what it does (done), install and config snippets (done), and honest limits (done).

Suggested listing copy:
- **Short (<160 chars):** Local-first MCP server that finds which stage of a RAG pipeline failed (parse, chunk, index, retrieve, rerank, pack), with evidence, and verifies the fix before applying it.
- **Category:** Developer Tools (and/or Data Science & ML: those are the two categories shown on similar tiles).
- **Keywords:** rag, retrieval, debugging, evaluation, lancedb, hybrid search, bm25, mcp.

## 4. Official MCP Registry (optional but recommended)
Sources disagree on `server.json` field names and schema versions, so generate it instead of hand-writing it:
```bash
# install mcp-publisher (see https://modelcontextprotocol.io/registry/quickstart)
mcp-publisher init        # creates server.json from your project; choose the PyPI package
mcp-publisher validate
mcp-publisher login github
mcp-publisher publish
```
For a PyPI package, ownership is proven by the `mcp-name: io.github.<you>/ragdoc-mcp` line already in the README. Publish the matching PyPI version *before* `mcp-publisher publish`. The registry was labeled "preview" when this was written.

## 5. After listing
- Watch issues for: path-confinement surprises (`RAGDOC_BASE_DIR`), Windows/macOS problems (never tested), Docling/fastembed adapter problems (never exercised).
- Collect two or three real-document failure cases and add them as fixtures. That is what turns the synthetic benchmark into evidence.
