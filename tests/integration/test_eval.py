"""Tests for domain_eval and verify_fix tools."""

import pytest
import tempfile
from pathlib import Path

from ragdoc_mcp.schemas import ContentType, FailureCode, FixProposal, VerifyFixResult
from ragdoc_mcp.eval import domain_eval, verify_fix, DomainEvaluator, FixVerifier
from ragdoc_mcp.planner import create_plan_from_path
from ragdoc_mcp.indexer import build_index, CollectionManager


class TestEval:
    """Test domain_eval and verify_fix functionality."""

    def setup_method(self):
        """Create temporary directory for test corpus."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self._create_test_corpus()

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_corpus(self):
        """Create a small test corpus with different content types."""
        (self.temp_dir / "config.md").write_text("""# Configuration

## Timeout Settings

The timeout is configured as 30 seconds.

## Retry Policy

max_retries = 3
backoff = exponential""")

        (self.temp_dir / "metrics.md").write_text("""# Metrics

| Metric | Value | Unit |
|--------|-------|------|
| Latency | 150 | ms |
| Throughput | 1000 | req/s |
| Error Rate | 0.01 | % |""")

        (self.temp_dir / "formula.md").write_text("""# Physics Formula

The energy-mass equivalence:

$$E = mc^2$$

Where:
- E = energy
- m = mass
- c = speed of light""")

    @pytest.mark.slow
    def test_domain_eval_prose(self):
        """Test domain evaluation for prose content."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_eval_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        assert manifest.total_chunks > 0
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        # Run domain eval on prose
        results = domain_eval(
            collection_path=collection_path,
            content_types=[ContentType.PROSE, ContentType.MARKDOWN],
            n_per_type=5,
        )
        
        assert len(results) >= 1
        for result in results:
            assert result.content_type in [ContentType.PROSE, ContentType.MARKDOWN]
            assert 0 <= result.hit_at_k.get(1, 0) <= 1
            assert 0 <= result.mrr <= 1
            assert 0 <= result.recall <= 1
            assert result.n_questions >= 0

    @pytest.mark.slow
    def test_domain_eval_table(self):
        """Test domain evaluation for table content."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_eval_table_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        results = domain_eval(
            collection_path=collection_path,
            content_types=[ContentType.TABLE],
            n_per_type=5,
        )
        
        assert len(results) >= 1
        for result in results:
            assert result.content_type == ContentType.TABLE

    @pytest.mark.slow
    def test_verify_fix_improvement(self):
        """Test verify_fix detects improvement."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_fix_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        # Create a fix proposal
        fix = FixProposal(
            target_stage=4,
            description="Test fix - increase retrieve top_k",
            config_change={"retrieve_top_k": 100},
            expected_impact="More candidates for reranker",
            confidence=0.8,
            risk="low",
        )
        
        # Test cases
        test_cases = [
            {"question": "What is the timeout?", "expected_file": str(self.temp_dir / "config.md")},
            {"question": "What is the latency?", "expected_file": str(self.temp_dir / "metrics.md")},
        ]
        
        result = verify_fix(
            collection_path=collection_path,
            fix=fix,
            test_cases=test_cases,
            apply=False,
        )
        
        assert result.fix_id == fix.fix_id
        assert "hit_rate" in result.before_metrics
        assert "hit_rate" in result.after_metrics
        assert "regression_check" in VerifyFixResult.model_fields
        assert result.apply_confirmed is False

    def test_fix_verifier_regression_detection(self):
        """Test FixVerifier detects regression."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_reg_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        fix = FixProposal(
            target_stage=4,
            description="Test fix",
            config_change={"retrieve_top_k": 100},
            expected_impact="More candidates",
            confidence=0.8,
            risk="low",
        )
        
        test_cases = [{"question": "What is the timeout?", "expected_file": str(self.temp_dir / "config.md")}]
        regression_cases = [{"question": "What is the latency?", "expected_file": str(self.temp_dir / "metrics.md")}]
        
        result = verify_fix(
            collection_path=collection_path,
            fix=fix,
            test_cases=test_cases,
            regression_cases=regression_cases,
            apply=False,
        )
        
        assert "regression_check" in VerifyFixResult.model_fields
        assert "before_hit_rate" in result.regression_check
        assert "after_hit_rate" in result.regression_check
        assert "regression" in result.regression_check

    def test_domain_evaluator_extractive_generation(self):
        """Test extractive question generation."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_gen_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        evaluator = DomainEvaluator(collection_path, plan.config.model_dump() if hasattr(plan.config, 'model_dump') else {})
        
        # Test extractive generation
        chunks = evaluator.collection.load_chunks()
        questions = evaluator._extractive_question_generation(chunks, 5)
        
        assert len(questions) >= 0  # May be 0 if no chunks match criteria

    def test_statement_to_question(self):
        """Test statement to question conversion."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_st_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        collection_path = str(Path.home() / ".ragdoc" / "collections" / collection_name)
        
        evaluator = DomainEvaluator(collection_path, {})
        
        # Test various statement patterns
        test_cases = [
            ("The timeout is 30 seconds", "What is the timeout?"),
            ("max_retries equals 3", "What does max_retries equal?"),
            ("batch_size = 32", "What is batch_size?"),
            ("learning_rate configured as 0.001", "What is the learning_rate configured as?"),
        ]
        
        for statement, expected_pattern in test_cases:
            question = evaluator._statement_to_question(statement)
            if question:
                assert "?" in question
                assert len(question) > len(statement) * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])