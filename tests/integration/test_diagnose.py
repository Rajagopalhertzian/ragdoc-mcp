"""Tests for diagnose_failure tool."""

import pytest
import tempfile
from pathlib import Path

from ragdoc_mcp.schemas import FailureCode
from ragdoc_mcp.diagnose import diagnose_failure, FailureDiagnoser
from ragdoc_mcp.embedders import get_embedder_registry
from ragdoc_mcp.planner import create_plan_from_path
from ragdoc_mcp.indexer import build_index, CollectionManager


class TestDiagnoseFailure:
    """Test diagnose_failure functionality."""

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
        """Create a small test corpus."""
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

    @pytest.mark.slow
    def test_diagnose_with_expected_source(self):
        """Test diagnose_failure with expected source file."""
        # Create plan and build index
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_diag_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
            batch_size=10,
        )
        
        assert manifest.total_chunks > 0
        
        # Run diagnosis
        result = diagnose_failure(
            collection_path=str(Path.home() / ".ragdoc" / "collections" / collection_name),
            question="What is the timeout setting?",
            expected_answer="The timeout is configured as 30 seconds.",
            expected_source=str(self.temp_dir / "config.md"),
        )
        
        assert result.first_failing_stage is not None
        assert result.failure_code in FailureCode
        assert result.confidence > 0
        assert len(result.evidence_per_stage) == 7
        assert len(result.fixes) > 0
        
        # Check that we have evidence for each stage
        for evidence in result.evidence_per_stage:
            assert evidence.stage in range(1, 8)
            assert evidence.stage_name
            assert isinstance(evidence.passed, bool)
            assert 0 <= evidence.confidence <= 1

    @pytest.mark.slow
    def test_diagnose_without_expected_source(self):
        """Test diagnose_failure without expected source (limited check)."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_diag2_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        result = diagnose_failure(
            collection_path=str(Path.home() / ".ragdoc" / "collections" / collection_name),
            question="What is the timeout?",
            expected_answer="30 seconds",
            expected_source=None,
        )
        
        assert result.first_failing_stage is not None
        assert result.failure_code in FailureCode

    @pytest.mark.slow
    def test_diagnose_retrieval_miss(self):
        """Test diagnose when answer not retrieved."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_diag3_{plan.plan_id}"
        
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        # Ask about something not in corpus
        result = diagnose_failure(
            collection_path=str(Path.home() / ".ragdoc" / "collections" / collection_name),
            question="What is the quantum entanglement threshold?",
            expected_answer="The threshold is 0.5",
            expected_source=str(self.temp_dir / "config.md"),
        )
        
        assert result.failure_code in FailureCode
        # Should detect retrieval miss or rank low

    def test_fix_proposals_generated(self):
        """Test that fix proposals are generated for each failure code."""
        from ragdoc_mcp.diagnose import FailureDiagnoser
        from ragdoc_mcp.schemas import FailureCode, FixProposal
        
        # Create a mock diagnoser to test fix generation
        plan_config = {
            "store": "lancedb",
            "embedder": "bge-small",
            "retrieve_top_k": 50,
            "rerank_top_k": 8,
            "rrf_k": 60,
            "reranker": "bge-reranker-base",
            "max_tokens_context": 1500,
        }
        
        # Test each failure code produces fixes
        for failure_code in FailureCode:
            if failure_code == FailureCode.GENERATION_SIDE:
                continue
                
            # Create dummy traces
            from ragdoc_mcp.diagnose import StageTrace
            traces = [StageTrace(
                stage_num=1, stage_name="Test", passed=False,
                failure_code=failure_code, confidence=0.8,
                evidence={}, rank_table=None
            )]
            
            diagnoser = FailureDiagnoser("", plan_config)
            fixes = diagnoser._generate_fixes(failure_code, traces)
            
            assert len(fixes) > 0, f"No fixes generated for {failure_code}"
            for fix in fixes:
                assert isinstance(fix, FixProposal)
                assert fix.description
                assert fix.expected_impact
                assert 0 <= fix.confidence <= 1
                assert fix.target_stage >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])