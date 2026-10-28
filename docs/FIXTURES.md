# Failure fixtures: how cases are labeled, and what they do not prove

`ragdoc.eval.fixtures.build_suite()` builds a synthetic technical-manual corpus (`ragdoc.eval.corpus`: prose, a table, an equation and a Python module per document, every fact unique) and a set of collections that each contain one **known, injected** fault.

| Family | How the fault is injected | Label rule |
|---|---|---|
| `parse_*` (4 sub-causes) | Test-only parser hooks: blank pages with a scanned flag, interleaved columns, column-major tables, garbled equation characters | Keep a case only if the intact pipeline answers it and the faulted one does not |
| `chunk_split` | 6-token windows | same |
| `context_missing` | Heading path not embedded | same |
| `chunk_diluted` | One huge chunk per file | same (only 2 cases arose) |
| `rerank_demoted` | A reranker that reverses the order | same |
| `budget_truncated` | A 45-token context budget | same |
| `index_stale` | Files edited after indexing | By construction |
| `index_missing` | New files added after indexing | By construction |
| `retrieval_miss`, `rank_low` | None: upstream is healthy | **Measured** rank of the gold chunk in the healthy pipeline |
| `control` | None | The healthy pipeline already puts the answer in context; any blame is a false blame |

## What this does not prove
- It is **synthetic**. Real documents have messier failures, and several can co-occur.
- The diagnoser was **developed against these same injectors**, so the score is a development-set score. A fresh set of faults written by someone else is the real test.
- `retrieval_miss` and `rank_low` labels come from the same rank logic the diagnoser uses, so they mainly guard against upstream faults being blamed on retrieval.
- `chunk_diluted` produced 2 cases (hybrid retrieval is robust to dilution when queries have rare terms), both diagnosed as `RERANK_DEMOTED`, which is also literally true there (the chunk moved #3 → #14). Treat `CHUNK_DILUTED` as unvalidated.
- The `scanned_no_ocr` *fix* is only simulated: a real scan needs OCR (Docling), which was not available in the sandbox. Detection of a real image-only PDF is tested.

## Reproduce
```bash
python benchmarks/run_attribution.py --docs 60 --per-code 8 --verify-per-code 3
```
