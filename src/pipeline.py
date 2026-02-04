"""
Edge Question Generation Pipeline - Orchestration
End-to-end pipeline with metrics and error handling
"""

import time
import asyncio
from typing import List, Optional
import tracemalloc

from src.pipeline_types import (
    ArticleInput,
    Chunk,
    QuestionCandidate,
    GenerationConfig,
    FinalQuestion,
    PipelineMetrics,
    PipelineResult
)
from src.errors import PipelineError, PipelineStage, ValidationError, ModelInferenceError
from src.model_adapter import ModelAdapter
from src.pre_generation import chunk_article, allocate_candidates
from src.post_generation import (
    validate_candidates,
    deduplicate_candidates,
    rank_candidates,
    select_final_questions
)


class QuestionGenerationPipeline:
    """
    End-to-end pipeline orchestrator.
    
    Handles pre-gen → gen → post-gen with error recovery and metrics.
    """
    
    def __init__(
        self,
        adapter: ModelAdapter,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize pipeline with model adapter and config.
        
        Args:
            adapter: Model adapter for inference
            config: Generation configuration (uses defaults if not provided)
        """
        self.adapter = adapter
        self.config = config or GenerationConfig()
    
    async def generate(self, article: ArticleInput) -> List[FinalQuestion]:
        """
        Generate questions from article.
        
        Returns 3-4 procedural questions optimized for KB retrieval.
        
        Args:
            article: Input article with text
            
        Returns:
            List of final questions
            
        Raises:
            PipelineError: If all candidates fail validation
        """
        result = await self.generate_with_metrics(article)
        return result.questions
    
    async def generate_with_metrics(
        self,
        article: ArticleInput
    ) -> PipelineResult:
        """
        Generate with detailed telemetry for debugging/benchmarking.
        
        Args:
            article: Input article with text
            
        Returns:
            PipelineResult with questions and metrics
            
        Raises:
            PipelineError: If critical errors occur
        """
        # Start timing and memory tracking
        start_time = time.perf_counter()
        tracemalloc.start()
        
        try:
            # ================================================================
            # Stage 1: Pre-Generation
            # ================================================================
            chunks = self._pre_generation(article)
            
            # ================================================================
            # Stage 2: Generation
            # ================================================================
            candidates, chunk_times = await self._generation(chunks, article.desired_questions)
            
            # ================================================================
            # Stage 3: Post-Generation
            # ================================================================
            questions = self._post_generation(candidates, chunks, article.desired_questions)
            
            # ================================================================
            # Metrics Collection
            # ================================================================
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            total_candidates = len(candidates)
            
            # Validation metrics
            validation_result = validate_candidates(candidates, chunks)
            validated_count = len(validation_result.valid)
            
            # Deduplication metrics
            deduplicated = deduplicate_candidates(validation_result.valid)
            deduplicated_count = len(deduplicated)
            
            metrics = PipelineMetrics(
                chunks_created=len(chunks),
                candidates_generated=total_candidates,
                candidates_validated=validated_count,
                candidates_deduplicated=deduplicated_count,
                latency_ms=(end_time - start_time) * 1000,
                memory_peak_mb=peak / (1024 * 1024),
                validation_pass_rate=validation_result.pass_rate,
                deduplication_reduction=(
                    (validated_count - deduplicated_count) / validated_count
                    if validated_count > 0 else 0.0
                ),
                chunk_processing_times=chunk_times
            )
            
            return PipelineResult(
                questions=questions,
                metrics=metrics
            )
            
        except Exception as e:
            tracemalloc.stop()
            
            # Re-raise PipelineError as-is
            if isinstance(e, PipelineError):
                raise
            
            # Wrap other exceptions
            raise PipelineError(
                message=f"Unexpected error in pipeline: {str(e)}",
                stage=PipelineStage.POST_GEN,
                recoverable=False
            ) from e
    
    def _pre_generation(self, article: ArticleInput) -> List[Chunk]:
        """
        Execute pre-generation stage: chunking and planning.
        
        Args:
            article: Input article
            
        Returns:
            List of chunks
            
        Raises:
            PipelineError: If chunking fails
        """
        try:
            chunks = chunk_article(article)
            return chunks
        except Exception as e:
            raise PipelineError(
                message=f"Pre-generation failed: {str(e)}",
                stage=PipelineStage.PRE_GEN,
                recoverable=False
            ) from e
    
    async def _generation(
        self,
        chunks: List[Chunk],
        desired_questions: int
    ) -> tuple[List[QuestionCandidate], List[float]]:
        """
        Execute generation stage: invoke model for each chunk.
        
        Args:
            chunks: Prepared chunks
            desired_questions: Target number of questions
            
        Returns:
            Tuple of (all_candidates, chunk_processing_times)
            
        Raises:
            PipelineError: If generation fails for all chunks
        """
        # Calculate candidate allocation
        allocations = allocate_candidates(len(chunks), desired_questions)
        
        # Check model readiness
        if not await self.adapter.is_ready():
            raise PipelineError(
                message="Model adapter is not ready",
                stage=PipelineStage.GENERATION,
                recoverable=True
            )
        
        # Process chunks in parallel (or sequentially for controlled memory)
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []
        
        # Process chunks sequentially to control memory usage
        for chunk, num_candidates in zip(chunks, allocations):
            chunk_start = time.perf_counter()
            
            try:
                candidates = await self.adapter.generate_questions(
                    chunk=chunk,
                    num_candidates=num_candidates,
                    generation_config=self.config
                )
                all_candidates.extend(candidates)
                
            except Exception as e:
                # Log error but continue with other chunks
                errors.append(e)
            
            finally:
                chunk_end = time.perf_counter()
                chunk_times.append((chunk_end - chunk_start) * 1000)
        
        # Check if we got any candidates
        if not all_candidates:
            raise PipelineError(
                message=f"Generation failed for all chunks. Errors: {errors}",
                stage=PipelineStage.GENERATION,
                recoverable=False
            )
        
        return all_candidates, chunk_times
    
    def _post_generation(
        self,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
        k: int
    ) -> List[FinalQuestion]:
        """
        Execute post-generation stage: validation, dedup, ranking, selection.
        
        Args:
            candidates: Raw candidates from generation
            chunks: Source chunks
            k: Desired number of final questions
            
        Returns:
            Final selected questions
            
        Raises:
            ValidationError: If no candidates pass validation
        """
        try:
            # Validate
            validation_result = validate_candidates(candidates, chunks)
            
            if not validation_result.valid:
                raise ValidationError(
                    message="All candidates failed validation",
                    rejected_count=len(validation_result.rejected),
                    valid_count=0
                )
            
            # Deduplicate
            deduplicated = deduplicate_candidates(validation_result.valid)
            
            if not deduplicated:
                raise ValidationError(
                    message="All candidates removed during deduplication",
                    rejected_count=len(validation_result.valid),
                    valid_count=0
                )
            
            # Rank
            ranked = rank_candidates(deduplicated, chunks)
            
            # Select
            final_questions = select_final_questions(ranked, k)
            
            if not final_questions:
                raise ValidationError(
                    message="Selection produced no questions",
                    rejected_count=len(candidates),
                    valid_count=0
                )
            
            return final_questions
            
        except ValidationError:
            raise
        except Exception as e:
            raise PipelineError(
                message=f"Post-generation failed: {str(e)}",
                stage=PipelineStage.POST_GEN,
                recoverable=False
            ) from e


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'QuestionGenerationPipeline',
]