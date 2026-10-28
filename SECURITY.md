# Security

## Reporting
Please report vulnerabilities privately (GitHub Security Advisories on this repository). Do not open public issues for them.

## Threat model and controls

| Threat | Control | Test |
|---|---|---|
| Reading files outside the corpus (`../`, absolute paths, symlinks) | All paths resolved and confined to `RAGDOC_BASE_DIR`; symlinks pointing outside are skipped during indexing | `tests/unit/test_security.py`, `test_symlink_outside_base_is_skipped` |
| Writing outside the collection | Writes only under `<base>/.ragdoc/`; collection names restricted to `[A-Za-z0-9_-]{1,64}`; plan/run ids matched against strict patterns | `test_bad_collection_names`, `test_bad_plan_ids_and_paths` |
| Prompt injection from document text | Results labeled `untrusted` with a notice; instruction-like phrases flagged in `search` and `diagnose_failure` (nested evidence included); text is never stripped or executed | `test_prompt_injection_is_flagged_in_search`, `test_evidence_flagged_when_document_contains_instructions` |
| Malicious tool arguments (SQL-ish filters, giant inputs, bad fix payloads) | Filters are applied in Python, never built into SQL; whitelisted filter/fix keys with type and range checks; limits on query length, `k`, token budget, cases, file count and size | `test_bad_fix_payloads_rejected`, `test_search_argument_limits`, `test_filters` |
| Unwanted data egress | No network by default, no telemetry. Neural models are opt-in; cloud parsers need `privacy="cloud_ok"`, an env-var key and recorded consent | `test_cloud_parser_requires_opt_in_key_and_hard_pdfs`, `test_neural_is_opt_in_and_never_downloads_while_planning` |
| Secrets in logs | API keys come only from the environment and are never logged. Logs go to stderr as JSON lines with a request id | review |
| Destructive changes | `verify_fix` works on a shadow copy; promotion requires `apply=true` **and** no regressions, and keeps a backup | `test_harmful_fix_is_flagged_and_refused_even_with_apply` |
| Resource exhaustion | Per-tool rate limit, soft deadlines (resumable builds), file-size and PDF-page caps | `test_rate_limit` |

## Known gaps (be aware)
- **HTTP transport has no authentication.** It binds to loopback by default; put a gateway with auth in front of it.
- Parsing happens in-process (pypdf). A hostile PDF is parsed with the server's privileges; there is no parser sandbox yet.
- Injection detection is pattern-based and will miss novel phrasings. It is a warning, not a guarantee.
- The cloud parser and Docling adapters are unexercised in CI.
- Dependency audit: run `pip-audit` (included in the CI workflow).
