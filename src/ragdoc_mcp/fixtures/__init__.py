"""Failure fixtures for RAG Doctor MCP - inject known faults for testing diagnose_failure."""

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ragdoc_mcp.schemas import ContentType, FailureCode


@dataclass
class FailureFixture:
    """A single failure fixture case."""
    fixture_id: str
    failure_code: FailureCode
    description: str
    corpus_path: str
    question: str
    expected_answer: str
    expected_source: str
    injected_fault: dict[str, Any]
    metadata: dict[str, Any]


class FailureFixtureGenerator:
    """Generates failure fixtures by injecting known faults into a clean corpus."""

    def __init__(self, base_corpus_path: Path, output_path: Path):
        self.base_corpus_path = base_corpus_path
        self.output_path = output_path
        self.fixtures: list[FailureFixture] = []

    def generate_all(self, n_per_failure_code: int = 10) -> list[FailureFixture]:
        """Generate fixtures for all failure codes."""
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Copy base corpus to working directory
        working_corpus = self.output_path / "working_corpus"
        if working_corpus.exists():
            shutil.rmtree(working_corpus)
        shutil.copytree(self.base_corpus_path, working_corpus)

        # Generate fixtures for each failure code
        failure_codes = [
            FailureCode.PARSE_LOSS_SCANNED_NO_OCR,
            FailureCode.PARSE_LOSS_TABLE_FLATTENED,
            FailureCode.PARSE_LOSS_EQUATION_GARBLED,
            FailureCode.PARSE_LOSS_COLUMN_ORDER,
            FailureCode.CHUNK_SPLIT,
            FailureCode.CHUNK_DILUTED,
            FailureCode.CONTEXT_MISSING,
            FailureCode.INDEX_STALE,
            FailureCode.INDEX_MISSING,
            FailureCode.RETRIEVAL_MISS,
            FailureCode.RANK_LOW,
            FailureCode.RERANK_DEMOTED,
            FailureCode.BUDGET_TRUNCATED,
        ]

        for failure_code in failure_codes:
            for i in range(n_per_failure_code):
                fixture = self._generate_fixture(failure_code, i, working_corpus)
                if fixture:
                    self.fixtures.append(fixture)

        # Save fixture metadata
        self._save_fixture_metadata()

        return self.fixtures

    def _generate_fixture(
        self, failure_code: FailureCode, index: int, working_corpus: Path
    ) -> Optional[FailureFixture]:
        """Generate a single fixture for a failure code."""
        fixture_id = f"{failure_code.value.lower()}_{index:03d}"

        if failure_code == FailureCode.PARSE_LOSS_SCANNED_NO_OCR:
            return self._inject_no_ocr(fixture_id, working_corpus)
        elif failure_code == FailureCode.PARSE_LOSS_TABLE_FLATTENED:
            return self._inject_table_flattened(fixture_id, working_corpus)
        elif failure_code == FailureCode.PARSE_LOSS_EQUATION_GARBLED:
            return self._inject_equation_garbled(fixture_id, working_corpus)
        elif failure_code == FailureCode.PARSE_LOSS_COLUMN_ORDER:
            return self._inject_column_order(fixture_id, working_corpus)
        elif failure_code == FailureCode.CHUNK_SPLIT:
            return self._inject_chunk_split(fixture_id, working_corpus)
        elif failure_code == FailureCode.CHUNK_DILUTED:
            return self._inject_chunk_diluted(fixture_id, working_corpus)
        elif failure_code == FailureCode.CONTEXT_MISSING:
            return self._inject_context_missing(fixture_id, working_corpus)
        elif failure_code == FailureCode.INDEX_STALE:
            return self._inject_index_stale(fixture_id, working_corpus)
        elif failure_code == FailureCode.INDEX_MISSING:
            return self._inject_index_missing(fixture_id, working_corpus)
        elif failure_code == FailureCode.RETRIEVAL_MISS:
            return self._inject_retrieval_miss(fixture_id, working_corpus)
        elif failure_code == FailureCode.RANK_LOW:
            return self._inject_rank_low(fixture_id, working_corpus)
        elif failure_code == FailureCode.RERANK_DEMOTED:
            return self._inject_rerank_demoted(fixture_id, working_corpus)
        elif failure_code == FailureCode.BUDGET_TRUNCATED:
            return self._inject_budget_truncated(fixture_id, working_corpus)

        return None

    def _inject_no_ocr(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: scanned PDF without OCR - answer in image but not extracted text."""
        # Find a PDF file
        pdf_files = list(corpus_path.rglob("*.pdf"))
        if not pdf_files:
            return None

        target_file = pdf_files[0]
        
        # Create a simple text file with the "answer" that would be in the image
        answer_text = "The quarterly revenue was $42.5 million in Q3 2024."
        question = "What was the quarterly revenue in Q3 2024?"
        
        # Create a sidecar text file with the ground truth
        gt_file = target_file.with_suffix(".ground_truth.txt")
        gt_file.write_text(answer_text)

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.PARSE_LOSS_SCANNED_NO_OCR,
            description="Scanned PDF without OCR - answer only in image layer",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(target_file),
            injected_fault={"type": "no_ocr", "target_file": str(target_file)},
            metadata={"original_file": str(target_file)},
        )

    def _inject_table_flattened(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: table flattened - headers lost, columns merged."""
        # Create a markdown file with a table
        md_file = corpus_path / "table_data.md"
        original_table = """| Quarter | Revenue | Expenses | Profit |
|---------|---------|----------|--------|
| Q1 2024 | $10.2M  | $8.1M    | $2.1M  |
| Q2 2024 | $12.5M  | $9.3M    | $3.2M  |
| Q3 2024 | $15.8M  | $11.2M   | $4.6M  |"""

        # Flatten the table (lose structure)
        flattened = "Quarter Revenue Expenses Profit Q1 2024 $10.2M $8.1M $2.1M Q2 2024 $12.5M $9.3M $3.2M Q3 2024 $15.8M $11.2M $4.6M"
        
        md_file.write_text(f"# Financial Data\n\n{flattened}\n\nNote: Table structure lost during parsing.")

        answer_text = "$4.6M"
        question = "What was the profit in Q3 2024?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.PARSE_LOSS_TABLE_FLATTENED,
            description="Table structure flattened - headers and columns merged",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "table_flattened", "target_file": str(md_file)},
            metadata={"original_table": original_table},
        )

    def _inject_equation_garbled(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: equation garbled - LaTeX not extracted properly."""
        md_file = corpus_path / "physics_formula.md"
        original_eq = "$$E = mc^2$$"
        garbled = "E mc 2"  # Lost formatting
        
        md_file.write_text(f"# Physics Formula\n\nThe famous equation: {garbled}\n\n(Original: {original_eq})")

        answer_text = "E = mc^2"
        question = "What is Einstein's mass-energy equivalence formula?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.PARSE_LOSS_EQUATION_GARBLED,
            description="Equation garbled - LaTeX formatting lost",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "equation_garbled", "target_file": str(md_file)},
            metadata={"original_equation": original_eq},
        )

    def _inject_column_order(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: column order scrambled."""
        md_file = corpus_path / "data_columns.md"
        original = """| Name | Age | City | Score |
|------|-----|------|-------|
| Alice | 30 | NYC | 95 |
| Bob | 25 | LA | 87 |"""

        # Scramble columns
        scrambled = """| Score | City | Age | Name |
|-------|------|-----|------|
| 95 | NYC | 30 | Alice |
| 87 | LA | 25 | Bob |"""
        
        md_file.write_text(f"# Data\n\n{scrambled}\n\n(Columns reordered)")

        answer_text = "Alice is 30 years old and lives in NYC."
        question = "How old is Alice and where does she live?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.PARSE_LOSS_COLUMN_ORDER,
            description="Table column order scrambled",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "column_order", "target_file": str(md_file)},
            metadata={"original_table": original},
        )

    def _inject_chunk_split(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: answer split across chunk boundary."""
        md_file = corpus_path / "long_document.md"
        # Create a long document where the answer spans chunks
        content = "INTRO " * 200  # ~800 tokens
        content += "The critical finding is that the timeout value is 30 seconds. "  # Answer starts here
        content += "This was confirmed by the engineering team. " * 200  # Continues across boundary
        
        md_file.write_text(f"# Long Document\n\n{content}")

        answer_text = "The critical finding is that the timeout value is 30 seconds."
        question = "What is the timeout value?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.CHUNK_SPLIT,
            description="Answer split across chunk boundary",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "chunk_split", "target_file": str(md_file)},
            metadata={"chunk_size": 512, "overlap": 0.1},
        )

    def _inject_chunk_diluted(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: answer diluted in large chunk."""
        md_file = corpus_path / "diluted_answer.md"
        # Large chunk with answer buried
        content = "BACKGROUND " * 500  # ~2000 tokens
        content += "The API rate limit is 100 requests per minute. "
        content += "ADDITIONAL " * 500
        
        md_file.write_text(f"# Large Document\n\n{content}")

        answer_text = "The API rate limit is 100 requests per minute."
        question = "What is the API rate limit?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.CHUNK_DILUTED,
            description="Answer diluted in oversized chunk",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "chunk_diluted", "target_file": str(md_file)},
            metadata={"chunk_size": 2000, "overlap": 0.1},
        )

    def _inject_context_missing(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: missing heading context."""
        md_file = corpus_path / "missing_context.md"
        # Content without heading context
        md_file.write_text("""# Document

The timeout is configured as 30 seconds.

## Configuration

timeout = 30
unit = seconds

## Retry Policy

max_retries = 3
backoff = exponential""")

        answer_text = "The timeout is configured as 30 seconds."
        question = "What is the timeout configuration?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.CONTEXT_MISSING,
            description="Missing heading context for answer",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "context_missing", "target_file": str(md_file)},
            metadata={},
        )

    def _inject_index_stale(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: index stale - file updated but index not rebuilt."""
        # This is a metadata fixture - the fault is in the index state, not the corpus
        md_file = corpus_path / "updated_doc.md"
        md_file.write_text("# Updated Document\n\nThe new timeout is 60 seconds.\n\n(Previously was 30 seconds)")

        answer_text = "The new timeout is 60 seconds."
        question = "What is the current timeout?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.INDEX_STALE,
            description="Index stale - file updated but not re-indexed",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "index_stale", "target_file": str(md_file)},
            metadata={"previous_answer": "The timeout is 30 seconds."},
        )

    def _inject_index_missing(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: chunk missing from index."""
        md_file = corpus_path / "missing_chunk.md"
        md_file.write_text("# Document\n\nThe secret key is ABC-123-XYZ.\n\nDo not share this key.")

        answer_text = "The secret key is ABC-123-XYZ."
        question = "What is the secret key?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.INDEX_MISSING,
            description="Chunk missing from index",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "index_missing", "target_file": str(md_file)},
            metadata={},
        )

    def _inject_retrieval_miss(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: retrieval miss - answer not in top 100."""
        md_file = corpus_path / "retrieval_miss.md"
        md_file.write_text("# Document\n\nThe configuration parameter max_connections is set to 50.\n\nOther settings: pool_size=20, timeout=30")

        answer_text = "The configuration parameter max_connections is set to 50."
        question = "What is the max_connections setting?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.RETRIEVAL_MISS,
            description="Answer not in top 100 retrieved results",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "retrieval_miss", "target_file": str(md_file)},
            metadata={"embedder": "bge-small"},
        )

    def _inject_rank_low(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: answer ranked low (in top 100 but below k)."""
        md_file = corpus_path / "rank_low.md"
        md_file.write_text("# Document\n\nThe default batch size is 32 for training.\n\nRelated: learning_rate=0.001, epochs=100")

        answer_text = "The default batch size is 32 for training."
        question = "What is the default batch size?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.RANK_LOW,
            description="Answer in top 100 but ranked below k=8",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "rank_low", "target_file": str(md_file)},
            metadata={"retrieve_top_k": 50, "rerank_top_k": 8},
        )

    def _inject_rerank_demoted(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: reranker demoted the gold chunk."""
        md_file = corpus_path / "rerank_demoted.md"
        md_file.write_text("# Document\n\nThe model uses AdamW optimizer with weight decay 0.01.\n\nOther optimizers: SGD, RMSprop")

        answer_text = "The model uses AdamW optimizer with weight decay 0.01."
        question = "What optimizer does the model use?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.RERANK_DEMOTED,
            description="Reranker demoted the gold chunk",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "rerank_demoted", "target_file": str(md_file)},
            metadata={"reranker": "bge-reranker-base"},
        )

    def _inject_budget_truncated(self, fixture_id: str, corpus_path: Path) -> Optional[FailureFixture]:
        """Inject: gold chunk truncated by token budget."""
        md_file = corpus_path / "budget_truncated.md"
        # Create many chunks, gold chunk is last
        content = "FILLER " * 1000  # ~4000 tokens
        content += "The critical parameter is learning_rate=0.0001.\n"
        content += "END " * 100
        
        md_file.write_text(f"# Large Document\n\n{content}")

        answer_text = "The critical parameter is learning_rate=0.0001."
        question = "What is the learning rate?"

        return FailureFixture(
            fixture_id=fixture_id,
            failure_code=FailureCode.BUDGET_TRUNCATED,
            description="Gold chunk truncated by token budget",
            corpus_path=str(corpus_path),
            question=question,
            expected_answer=answer_text,
            expected_source=str(md_file),
            injected_fault={"type": "budget_truncated", "target_file": str(md_file)},
            metadata={"max_tokens": 1500, "chunk_size": 512},
        )

    def _save_fixture_metadata(self) -> None:
        """Save fixture metadata to JSON."""
        metadata = []
        for f in self.fixtures:
            metadata.append({
                "fixture_id": f.fixture_id,
                "failure_code": f.failure_code.value,
                "description": f.description,
                "corpus_path": f.corpus_path,
                "question": f.question,
                "expected_answer": f.expected_answer,
                "expected_source": f.expected_source,
                "injected_fault": f.injected_fault,
                "metadata": f.metadata,
            })
        
        output_file = self.output_path / "fixtures_metadata.json"
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)


def create_failure_fixtures(
    base_corpus_path: str | Path,
    output_path: str | Path,
    n_per_failure_code: int = 10,
) -> list[FailureFixture]:
    """Main entry point to create failure fixtures."""
    generator = FailureFixtureGenerator(Path(base_corpus_path), Path(output_path))
    return generator.generate_all(n_per_failure_code)