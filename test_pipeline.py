"""
Test Suite for Question Generation Pipeline

Tests for all pipeline stages and components.
"""

import pytest
import asyncio

from src.pipeline import QuestionGenerationPipeline
from src.model_adapter import MockAdapter
from src.types import GenerationConfig, ArticleInput, Chunk, QuestionCandidate, QuestionType

from src.pre_generation import chunk_article, normalize_text, allocate_candidates
from src.post_generation import validate_candidates, deduplicate_candidates, rank_candidates, select_final_questions

from src.errors import ChunkingError, ValidationError



# ============================================================================
# Pre-Generation Tests
# ============================================================================

class TestNormalization:
    """Test text normalization"""
    
    def test_collapse_whitespace(self):
        text = "Hello    world   test"
        result = normalize_text(text)
        assert result == "Hello world test"
    
    def test_collapse_newlines(self):
        text = "Line 1\n\n\n\nLine 2"
        result = normalize_text(text)
        assert result == "Line 1\n\nLine 2"
    
    def test_strip_whitespace(self):
        text = "   Hello world   "
        result = normalize_text(text)
        assert result == "Hello world"


class TestChunking:
    """Test article chunking logic"""
    
    def test_short_article_single_chunk(self):
        # 600 words -> 1 chunk
        text = " ".join(["word"] * 600)
        article = ArticleInput(text=text)
        chunks = chunk_article(article)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
    
    def test_medium_article_multiple_chunks(self):
        # 1200 words -> 2 chunks
        text = " ".join(["word"] * 1200)
        article = ArticleInput(text=text)
        chunks = chunk_article(article)
        
        assert len(chunks) == 2
        assert all(len(c) <= 800 for c in chunks)
    
    def test_empty_article_raises_error(self):
        article = ArticleInput(text="")
        with pytest.raises(ChunkingError):
            chunk_article(article)
    
    def test_too_short_article_raises_error(self):
        article = ArticleInput(text="too short")
        with pytest.raises(ChunkingError):
            chunk_article(article)
    
    def test_chunk_metadata(self):
        text = " ".join(["word"] * 600)
        article = ArticleInput(text=text)
        chunks = chunk_article(article)
        
        chunk = chunks[0]
        assert chunk.start_offset >= 0
        assert chunk.end_offset > chunk.start_offset
        assert chunk.text in text


class TestCandidateAllocation:
    """Test candidate allocation logic"""
    
    def test_single_chunk_allocation(self):
        allocations = allocate_candidates(num_chunks=1, desired_questions=4)
        assert len(allocations) == 1
        assert allocations[0] >= 1
    
    def test_multiple_chunks_allocation(self):
        allocations = allocate_candidates(num_chunks=3, desired_questions=4)
        assert len(allocations) == 3
        assert sum(allocations) >= 4  # At least desired questions
        assert all(1 <= a <= 3 for a in allocations)  # Clamped to [1, 3]
    
    def test_allocation_sum_approximately_2k(self):
        k = 4
        allocations = allocate_candidates(num_chunks=3, desired_questions=k)
        total = sum(allocations)
        # Should be around 2*k (8), but clamped per chunk
        assert 6 <= total <= 10


# ============================================================================
# Post-Generation Tests
# ============================================================================

class TestValidation:
    """Test candidate validation"""
    
    def setup_method(self):
        """Create test fixtures"""
        self.chunk = Chunk(
            chunk_id=0,
            text="To enable conversion tracking, navigate to Tools menu and select Conversions. Click the + button.",
            start_offset=0,
            end_offset=100
        )
    
    def test_valid_candidate_passes(self):
        candidate = QuestionCandidate(
            question="How do I enable conversion tracking in the platform?",
            type=QuestionType.PROCEDURAL,
            evidence="navigate to Tools menu and select Conversions",
            chunk_id=0
        )
        
        result = validate_candidates([candidate], [self.chunk])
        assert len(result.valid) == 1
        assert len(result.rejected) == 0
    
    def test_question_without_how_rejected(self):
        candidate = QuestionCandidate(
            question="What is conversion tracking?",
            type=QuestionType.PROCEDURAL,
            evidence="navigate to Tools menu",
            chunk_id=0
        )
        
        result = validate_candidates([candidate], [self.chunk])
        assert len(result.valid) == 0
        assert len(result.rejected) == 1
        assert "does not start with 'How'" in result.rejected[0].reason
    
    def test_question_without_question_mark_rejected(self):
        candidate = QuestionCandidate(
            question="How do I enable conversion tracking",
            type=QuestionType.PROCEDURAL,
            evidence="navigate to Tools menu",
            chunk_id=0
        )
        
        result = validate_candidates([candidate], [self.chunk])
        assert len(result.rejected) == 1
        assert "does not end with '?'" in result.rejected[0].reason
    
    def test_too_short_question_rejected(self):
        candidate = QuestionCandidate(
            question="How to do it?",
            type=QuestionType.PROCEDURAL,
            evidence="navigate to Tools menu",
            chunk_id=0
        )
        
        result = validate_candidates([candidate], [self.chunk])
        assert len(result.rejected) == 1
        assert "too short" in result.rejected[0].reason.lower()
    
    def test_evidence_not_in_chunk_rejected(self):
        candidate = QuestionCandidate(
            question="How do I configure advanced settings?",
            type=QuestionType.PROCEDURAL,
            evidence="this text is not in the chunk at all",
            chunk_id=0
        )
        
        result = validate_candidates([candidate], [self.chunk])
        assert len(result.rejected) == 1
        assert "not found in chunk" in result.rejected[0].reason.lower()


class TestDeduplication:
    """Test candidate deduplication"""
    
    def test_no_duplicates(self):
        candidates = [
            QuestionCandidate(
                question="How do I enable conversion tracking?",
                type=QuestionType.PROCEDURAL,
                evidence="evidence 1",
                chunk_id=0
            ),
            QuestionCandidate(
                question="How do I set up audience targeting?",
                type=QuestionType.PROCEDURAL,
                evidence="evidence 2",
                chunk_id=1
            ),
        ]
        
        result = deduplicate_candidates(candidates)
        assert len(result) == 2
    
    def test_removes_duplicates(self):
        candidates = [
            QuestionCandidate(
                question="How do I enable conversion tracking?",
                type=QuestionType.PROCEDURAL,
                evidence="evidence 1",
                chunk_id=0
            ),
            QuestionCandidate(
                question="How do I enable conversion tracking in the system?",
                type=QuestionType.PROCEDURAL,
                evidence="evidence 2",
                chunk_id=1
            ),
        ]
        
        result = deduplicate_candidates(candidates)
        assert len(result) == 1  # Should keep only one
    
    def test_keeps_longer_question(self):
        # These questions are very similar (>70% similarity)
        short = QuestionCandidate(
            question="How do I enable conversion tracking?",
            type=QuestionType.PROCEDURAL,
            evidence="evidence 1",
            chunk_id=0
        )
        long = QuestionCandidate(
            question="How do I enable conversion tracking properly?",
            type=QuestionType.PROCEDURAL,
            evidence="evidence 2",
            chunk_id=1
        )
        
        result = deduplicate_candidates([short, long])
        assert len(result) == 1
        assert result[0].question == long.question


class TestRanking:
    """Test candidate ranking"""
    
    def test_ranking_produces_scores(self):
        candidates = [
            QuestionCandidate(
                question="How do I enable conversion tracking?",
                type=QuestionType.PROCEDURAL,
                evidence="navigate to Tools > Conversions. Click the + button",
                chunk_id=0
            ),
        ]
        chunks = [Chunk(chunk_id=0, text="test", start_offset=0, end_offset=10)]
        
        ranked = rank_candidates(candidates, chunks)
        assert len(ranked) == 1
        assert 0.0 <= ranked[0].score <= 1.0
    
    def test_ranking_sorts_descending(self):
        candidates = [
            QuestionCandidate(
                question="How?",  # Low specificity
                type=QuestionType.PROCEDURAL,
                evidence="text",
                chunk_id=0
            ),
            QuestionCandidate(
                question="How do I configure advanced conversion tracking settings for campaign optimization?",
                type=QuestionType.PROCEDURAL,
                evidence="navigate to Tools menu, select Settings, click Advanced button with ID 123",
                chunk_id=1
            ),
        ]
        chunks = [
            Chunk(chunk_id=0, text="text", start_offset=0, end_offset=10),
            Chunk(chunk_id=1, text="test", start_offset=0, end_offset=10)
        ]
        
        ranked = rank_candidates(candidates, chunks)
        # Ranking should sort descending - first element should have highest score
        assert ranked[0].score >= ranked[1].score
        # The more specific question should rank higher
        assert "advanced conversion tracking" in ranked[0].candidate.question.lower()


class TestSelection:
    """Test final question selection"""
    
    def test_selects_top_k(self):
        from src.types import RankedCandidate
        
        ranked = [
            RankedCandidate(
                candidate=QuestionCandidate(
                    question=f"How do I do task {i}?",
                    type=QuestionType.PROCEDURAL,
                    evidence="evidence",
                    chunk_id=i
                ),
                score=1.0 - (i * 0.1)
            )
            for i in range(10)
        ]
        
        selected = select_final_questions(ranked, k=4)
        assert len(selected) == 4
    
    def test_returns_all_if_less_than_k(self):
        from src.types import RankedCandidate
        
        ranked = [
            RankedCandidate(
                candidate=QuestionCandidate(
                    question="How do I do this?",
                    type=QuestionType.PROCEDURAL,
                    evidence="evidence",
                    chunk_id=0
                ),
                score=0.9
            ),
        ]
        
        selected = select_final_questions(ranked, k=4)
        assert len(selected) == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Test end-to-end pipeline"""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_mock_adapter(self):
        adapter = MockAdapter(latency_ms=10)
        pipeline = QuestionGenerationPipeline(adapter)
        
        article = ArticleInput(
            text=" ".join(["word"] * 1000),
            desired_questions=4
        )
        
        result = await pipeline.generate_with_metrics(article)
        
        assert len(result.questions) > 0
        assert result.metrics.chunks_created > 0
        assert result.metrics.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics(self):
        adapter = MockAdapter(latency_ms=10)
        pipeline = QuestionGenerationPipeline(adapter)
        
        article = ArticleInput(
            text=" ".join(["word"] * 800),
            desired_questions=3
        )
        
        result = await pipeline.generate_with_metrics(article)
        metrics = result.metrics
        
        assert metrics.chunks_created >= 1
        assert metrics.candidates_generated > 0
        assert 0 <= metrics.validation_pass_rate <= 1.0
        assert metrics.memory_peak_mb > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])