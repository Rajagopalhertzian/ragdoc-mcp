# Contributing

```bash
pip install -e ".[dev]"
ruff check src tests && pytest
```
- Every new failure code or diagnosis rule needs a fixture family in `src/ragdoc/eval/fixtures.py` *and* a test. Do not tune a rule to match your own fixture label: if the evidence is right and the label is arguable, fix the label or report the case as ambiguous.
- Do not claim a number you did not measure. Update the "What is and isn't verified" table in the README when results change.
- Tool descriptions are written for an LLM: say when to use the tool, keep annotations honest (read-only / destructive / idempotent).
- Never print to stdout (it is the MCP stdio channel); log to stderr.
