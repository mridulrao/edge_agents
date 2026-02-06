"""
Edge Question Generation Pipeline - Orchestration
End-to-end pipeline with parallel processing support
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
from src.errors import PipelineError, PipelineStage, ValidationError
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
    
    Handles pre-gen → gen → post-gen with parallel processing support.
    """
    
    def __init__(
        self,
        adapter: ModelAdapter,
        config: Optional[GenerationConfig] = None,
        parallel_processing: bool = False,
        max_concurrent_chunks: int = 3
    ):
        """
        Initialize pipeline with model adapter and config.
        
        Args:
            adapter: Model adapter for inference
            config: Generation configuration (uses defaults if not provided)
            parallel_processing: Enable parallel chunk processing for speed
            max_concurrent_chunks: Maximum concurrent chunk requests (if parallel=True)
        """
        self.adapter = adapter
        self.config = config or GenerationConfig()
        self.parallel_processing = parallel_processing
        self.max_concurrent_chunks = max_concurrent_chunks
    
    async def generate(self, article: ArticleInput) -> List[FinalQuestion]:
        """
        Generate questions from article.
        
        Returns N questions as requested in article.desired_questions.
        
        Args:
            article: Input article with text
            
        Returns:
            List of final questions
            
        Raises:
            PipelineError: If generation fails
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
        Execute pre-generation stage: chunking with buffer strategy.
        
        Args:
            article: Input article
            
        Returns:
            List of chunks (1.5x desired_questions for buffer)
            
        Raises:
            PipelineError: If chunking fails
        """
        try:
            chunks = chunk_article(article)
            
            # Log chunking info
            print(f"\n📦 Chunking complete:")
            print(f"   • Desired questions: {article.desired_questions}")
            print(f"   • Chunks created: {len(chunks)} (1.5x buffer)")
            print(f"   • Chunk size: ~200 words each")
            
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
        
        Supports both sequential (memory-safe) and parallel (faster) processing.
        
        Args:
            chunks: Prepared chunks
            desired_questions: Target number of questions
            
        Returns:
            Tuple of (all_candidates, chunk_processing_times)
            
        Raises:
            PipelineError: If generation fails for all chunks
        """
        # Calculate candidate allocation (always 1 per chunk)
        allocations = allocate_candidates(len(chunks), desired_questions)
        
        # Check model readiness
        if not await self.adapter.is_ready():
            raise PipelineError(
                message="Model adapter is not ready",
                stage=PipelineStage.GENERATION,
                recoverable=True
            )
        
        # Choose processing mode
        if self.parallel_processing:
            print(f"\n⚡ Parallel processing (max {self.max_concurrent_chunks} concurrent)")
            all_candidates, chunk_times, errors = await self._process_chunks_parallel(
                chunks, allocations
            )
        else:
            print(f"\n🔄 Sequential processing")
            all_candidates, chunk_times, errors = await self._process_chunks_sequential(
                chunks, allocations
            )
        
        # Check if we got any candidates
        if not all_candidates:
            raise PipelineError(
                message=f"Generation failed for all chunks. Errors: {errors}",
                stage=PipelineStage.GENERATION,
                recoverable=False
            )
        
        print(f"   • Generated {len(all_candidates)} candidate questions")
        if errors:
            print(f"   ⚠️  {len(errors)} chunks failed")
        
        return all_candidates, chunk_times
    
    async def _process_chunks_sequential(
        self,
        chunks: List[Chunk],
        allocations: List[int]
    ) -> tuple[List[QuestionCandidate], List[float], List[Exception]]:
        """
        Process chunks sequentially (memory-safe, default).
        
        Returns:
            Tuple of (candidates, times, errors)
        """
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []
        
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
        
        return all_candidates, chunk_times, errors
    
    async def _process_chunks_parallel(
        self,
        chunks: List[Chunk],
        allocations: List[int]
    ) -> tuple[List[QuestionCandidate], List[float], List[Exception]]:
        """
        Process chunks in parallel with concurrency control.
        
        Uses semaphore to limit concurrent requests.
        
        Returns:
            Tuple of (candidates, times, errors)
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)
        
        async def process_chunk_with_semaphore(
            chunk: Chunk,
            num_candidates: int,
            chunk_idx: int
        ) -> tuple[int, List[QuestionCandidate], float, Optional[Exception]]:
            """Process a single chunk with semaphore for concurrency control"""
            async with semaphore:
                chunk_start = time.perf_counter()
                
                try:
                    candidates = await self.adapter.generate_questions(
                        chunk=chunk,
                        num_candidates=num_candidates,
                        generation_config=self.config
                    )
                    chunk_end = time.perf_counter()
                    elapsed_ms = (chunk_end - chunk_start) * 1000
                    return chunk_idx, candidates, elapsed_ms, None
                
                except Exception as e:
                    chunk_end = time.perf_counter()
                    elapsed_ms = (chunk_end - chunk_start) * 1000
                    return chunk_idx, [], elapsed_ms, e
        
        # Create tasks for all chunks
        tasks = [
            process_chunk_with_semaphore(chunk, num_candidates, idx)
            for idx, (chunk, num_candidates) in enumerate(zip(chunks, allocations))
        ]
        
        # Execute in parallel with gather
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Sort results by chunk index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Extract candidates, times, and errors
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []
        
        for chunk_idx, candidates, elapsed_ms, error in results:
            chunk_times.append(elapsed_ms)
            
            if error:
                errors.append(error)
            else:
                all_candidates.extend(candidates)
        
        return all_candidates, chunk_times, errors
    
    def _post_generation(
        self,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
        k: int
    ) -> List[FinalQuestion]:
        """
        Execute post-generation stage: minimal validation, take top K.
        
        Args:
            candidates: Raw candidates from generation
            chunks: Source chunks
            k: Desired number of final questions
            
        Returns:
            Final selected questions
            
        Raises:
            ValidationError: If not enough valid candidates
        """
        try:
            print(f"\n🔍 Post-processing:")
            print(f"   • Candidates generated: {len(candidates)}")
            print(f"   • Requested questions: {k}")
            
            # Minimal validation (no evidence checking)
            validation_result = validate_candidates(candidates, chunks)
            
            if not validation_result.valid:
                raise ValidationError(
                    message="All candidates failed validation",
                    rejected_count=len(validation_result.rejected),
                    valid_count=0
                )
            
            print(f"   • Valid candidates: {len(validation_result.valid)}")
            
            # No deduplication
            deduplicated = deduplicate_candidates(validation_result.valid)
            
            # Neutral ranking
            ranked = rank_candidates(deduplicated, chunks)
            
            # Select top K
            final_questions = select_final_questions(ranked, k)
            
            print(f"   • Final questions: {len(final_questions)}")
            
            if len(final_questions) < k:
                print(f"   ⚠️  Got {len(final_questions)} questions, wanted {k}")
            
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