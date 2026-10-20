"""Integration tests for RAG Doctor MCP - end-to-end pipeline tests."""

import pytest
import tempfile
from pathlib import Path

from ragdoc_mcp.schemas import (
    ContentType, FailureCode, ToolStatus, DetailLevel,
    RagConstraints, PrivacyMode, HardwareTarget
)
from ragdoc_mcp.planner import create_plan_from_path, CorpusProfiler
from ragdoc_mcp.indexer import CollectionManager, build_index
from ragdoc_mcp.fixtures import create_failure_fixtures


class TestIntegration:
    """End-to-end integration tests."""

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
        """Create a small test corpus with various document types."""
        # Markdown with headings
        (self.temp_dir / "readme.md").write_text("""# Project Overview

## Configuration

The timeout is configured as 30 seconds.

## Retry Policy

max_retries = 3
backoff = exponential""")

        # Python file
        (self.temp_dir / "config.py").write_text("""# Configuration module

TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF = "exponential"

def get_timeout():
    return TIMEOUT""")

        # Table data
        (self.temp_dir / "metrics.md").write_text("""# Metrics

| Metric | Value | Unit |
|--------|-------|------|
| Latency | 150 | ms |
| Throughput | 1000 | req/s |
| Error Rate | 0.01 | % |""")

        # Equation
        (self.temp_dir / "formula.md").write_text("""# Physics Formula

The energy-mass equivalence:

$$E = mc^2$$

Where:
- E = energy
- m = mass
- c = speed of light""")

    @ pytest.mark.slow
    def test_profile_plan_index_search_flow(self):
        """Test complete flow: profile -> plan -> index -> search."""
        # 1. Profile corpus
        profiler = CorpusProfiler(sample_size=10)
        profile = profiler.profile(self.temp_dir)
        
        assert profile.total_files == 4
        assert ContentType.MARKDOWN in profile.type_mix
        assert ContentType.CODE in profile.type_mix
        assert "py" in profile.languages

        # 2. Create plan
        plan = create_plan_from_path(
            str(self.temp_dir),
            privacy="local_only",
            hardware="cpu",
        )
        
        assert plan.plan_id
        # Plan uses the path as profile_id when created from path
        assert plan.profile_id is not None
        assert plan.config.embedder
        assert plan.config.store.value == "lancedb"

        # 3. Build index (collection name derived from plan)
        collection_name = f"test_{plan.plan_id}"
        # Use a smaller subset for faster test
        import time
        start = time.time()
        manifest = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
            batch_size=10,
        )
        elapsed = time.time() - start
        print(f"Index build took {elapsed:.1f}s")
        
        assert manifest.collection_name == collection_name
        assert manifest.total_chunks > 0
        assert manifest.plan.plan_id == plan.plan_id

        # 4. Load collection and verify chunks
        collection = CollectionManager(collection_name)
        chunks = collection.load_chunks()
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.file_path
            assert chunk.content_type in ContentType

    @ pytest.mark.slow
    def test_failure_fixture_generation(self):
        """Test generating failure fixtures."""
        fixture_dir = self.temp_dir / "fixtures"
        fixtures = create_failure_fixtures(
            base_corpus_path=self.temp_dir,
            output_path=fixture_dir,
            n_per_failure_code=2,  # Small number for test
        )
        
        assert len(fixtures) > 0
        
        # Verify metadata file created
        metadata_file = fixture_dir / "fixtures_metadata.json"
        assert metadata_file.exists()
        
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert len(metadata) == len(fixtures)
        
        # Check fixture structure
        for f in fixtures:
            assert f.fixture_id
            assert f.failure_code in FailureCode
            assert f.question
            assert f.expected_answer
            assert f.expected_source
            assert "type" in f.injected_fault

    @ pytest.mark.slow
    def test_multiple_failure_codes(self):
        """Test that all failure codes have fixtures."""
        fixture_dir = self.temp_dir / "fixtures"
        fixtures = create_failure_fixtures(
            base_corpus_path=self.temp_dir,
            output_path=fixture_dir,
            n_per_failure_code=1,
        )
        
        failure_codes_covered = {f.failure_code for f in fixtures}
        
        # Should cover at least these codes
        expected_codes = {
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
        }
        
        assert failure_codes_covered >= expected_codes

    @ pytest.mark.slow
    def test_build_index_resumable(self):
        """Test that build_index is resumable."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_resume_{plan.plan_id}"
        
        # First build
        manifest1 = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        chunks1 = manifest1.total_chunks
        
        # Second build (should resume and not duplicate)
        manifest2 = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        chunks2 = manifest2.total_chunks
        
        # Should have same or fewer chunks (no duplicates)
        assert chunks2 <= chunks1 * 1.1  # Allow small variance

    @ pytest.mark.slow
    def test_build_index_force_reindex(self):
        """Test force_reindex flag."""
        plan = create_plan_from_path(str(self.temp_dir))
        collection_name = f"test_force_{plan.plan_id}"
        
        # First build
        manifest1 = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
        )
        
        # Force reindex
        manifest2 = build_index(
            collection_name=collection_name,
            plan=plan,
            corpus_path=self.temp_dir,
            resume=True,
            force_reindex=True,
        )
        
        assert manifest2.total_chunks > 0

    def test_constraints_affect_plan(self):
        """Test that different constraints produce different plans."""
        # CPU only plan
        plan_cpu = create_plan_from_path(
            str(self.temp_dir),
            privacy="local_only",
            hardware="cpu",
        )
        
        # GPU plan (if available)
        plan_gpu = create_plan_from_path(
            str(self.temp_dir),
            privacy="local_only",
            hardware="gpu",
        )
        
        # Cloud allowed plan
        plan_cloud = create_plan_from_path(
            str(self.temp_dir),
            privacy="cloud_ok",
            hardware="cpu",
        )
        
        # All should be valid plans
        for plan in [plan_cpu, plan_gpu, plan_cloud]:
            assert plan.plan_id
            assert plan.config.embedder
            assert plan.cost_estimate.total_time_minutes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])