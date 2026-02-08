"""
Edge Question Generation Pipeline - Orchestration
End-to-end pipeline with parallel processing support
"""

import time
import asyncio
from typing import List, Optional, Tuple
import tracemalloc

import psutil  # NEW: for server RSS tracking

from local_llm.pipeline_types import (
    ArticleInput,
    Chunk,
    QuestionCandidate,
    GenerationConfig,
    FinalQuestion,
    PipelineMetrics,
    PipelineResult
)
from local_llm.errors import PipelineError, PipelineStage, ValidationError
from local_llm.model_adapter import ModelAdapter
from local_llm.pre_generation import chunk_article, allocate_candidates
from local_llm.post_generation import (
    validate_candidates,
    deduplicate_candidates,
    rank_candidates,
    select_final_questions
)


# ============================================================================
# Memory helpers (server process)
# ============================================================================

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def _rss_mb(pid: int) -> float:
    p = psutil.Process(pid)
    return _bytes_to_mb(p.memory_info().rss)


async def _sample_rss_peak(pid: int, stop: asyncio.Event, interval_s: float = 0.2) -> float:
    """
    Sample RSS periodically and return the peak RSS observed (MB).
    Best effort: if process exits or errors occur, sampling ends.
    """
    peak = 0.0
    while not stop.is_set():
        try:
            peak = max(peak, _rss_mb(pid))
        except psutil.NoSuchProcess:
            break
        except Exception:
            # best effort; ignore transient errors
            pass
        await asyncio.sleep(interval_s)
    return peak


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
        """
        result = await self.generate_with_metrics(article)
        return result.questions

    async def generate_with_metrics(self, article: ArticleInput) -> PipelineResult:
        """
        Generate with detailed telemetry for debugging/benchmarking.

        IMPORTANT:
        - tracemalloc measures Python heap only (NOT llama-server/model memory).
        - We additionally track llama-server RSS (resident memory) if the adapter exposes a PID.
        """
        start_time = time.perf_counter()

        # Python heap tracking (useful, but not "model memory")
        tracemalloc.start()

        # Server memory tracking
        server_pid: Optional[int] = None
        server_rss_before_mb: Optional[float] = None
        server_rss_after_mb: Optional[float] = None
        server_rss_peak_mb: Optional[float] = None

        rss_stop = asyncio.Event()
        rss_task: Optional[asyncio.Task] = None

        try:
            # Try to get llama-server PID (only available for managed local server)
            if hasattr(self.adapter, "get_server_pid"):
                try:
                    server_pid = self.adapter.get_server_pid()
                except Exception:
                    server_pid = None

            # Snapshot RSS before any generation work
            if server_pid is not None:
                try:
                    server_rss_before_mb = _rss_mb(server_pid)
                except Exception:
                    server_rss_before_mb = None

                # Start sampler to capture peak during generation
                rss_task = asyncio.create_task(_sample_rss_peak(server_pid, rss_stop, interval_s=0.2))

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

            # Stop sampler and collect peak
            if rss_task is not None:
                rss_stop.set()
                try:
                    server_rss_peak_mb = await rss_task
                except Exception:
                    server_rss_peak_mb = None

            # Snapshot RSS after generation
            if server_pid is not None:
                try:
                    server_rss_after_mb = _rss_mb(server_pid)
                except Exception:
                    server_rss_after_mb = None

            # ================================================================
            # Metrics Collection
            # ================================================================
            end_time = time.perf_counter()
            _, py_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            total_candidates = len(candidates)

            validation_result = validate_candidates(candidates, chunks)
            validated_count = len(validation_result.valid)

            deduplicated = deduplicate_candidates(validation_result.valid)
            deduplicated_count = len(deduplicated)

            python_heap_peak_mb = py_heap_peak / (1024 * 1024)

            # Prefer server RSS peak as "memory_peak_mb" (what you actually care about)
            if server_rss_peak_mb is not None and server_rss_peak_mb > 0:
                memory_peak_mb = server_rss_peak_mb
            else:
                memory_peak_mb = python_heap_peak_mb

            metrics = PipelineMetrics(
                chunks_created=len(chunks),
                candidates_generated=total_candidates,
                candidates_validated=validated_count,
                candidates_deduplicated=deduplicated_count,
                latency_ms=(end_time - start_time) * 1000.0,
                memory_peak_mb=memory_peak_mb,
                validation_pass_rate=validation_result.pass_rate,
                deduplication_reduction=(
                    (validated_count - deduplicated_count) / validated_count
                    if validated_count > 0 else 0.0
                ),
                chunk_processing_times=chunk_times,

                # -------- NEW OPTIONAL FIELDS (add to PipelineMetrics with defaults=None)
                python_heap_peak_mb=python_heap_peak_mb,
                server_pid=server_pid,
                server_rss_before_mb=server_rss_before_mb,
                server_rss_after_mb=server_rss_after_mb,
                server_rss_peak_mb=server_rss_peak_mb,
                server_rss_delta_mb=(
                    (server_rss_after_mb - server_rss_before_mb)
                    if (server_rss_after_mb is not None and server_rss_before_mb is not None)
                    else None
                ),
            )

            return PipelineResult(questions=questions, metrics=metrics)

        except Exception as e:
            # Ensure tracemalloc stops
            try:
                tracemalloc.stop()
            except Exception:
                pass

            # Ensure sampler stops
            if rss_task is not None:
                rss_stop.set()
                try:
                    rss_task.cancel()
                except Exception:
                    pass

            if isinstance(e, PipelineError):
                raise

            raise PipelineError(
                message=f"Unexpected error in pipeline: {str(e)}",
                stage=PipelineStage.POST_GEN,
                recoverable=False
            ) from e

    def _pre_generation(self, article: ArticleInput) -> List[Chunk]:
        """
        Execute pre-generation stage: chunking with buffer strategy.
        """
        try:
            chunks = chunk_article(article)

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

    async def _generation(self, chunks: List[Chunk], desired_questions: int) -> Tuple[List[QuestionCandidate], List[float]]:
        """
        Execute generation stage: invoke model for each chunk.
        """
        allocations = allocate_candidates(len(chunks), desired_questions)

        if not await self.adapter.is_ready():
            raise PipelineError(
                message="Model adapter is not ready",
                stage=PipelineStage.GENERATION,
                recoverable=True
            )

        if self.parallel_processing:
            print(f"\n⚡ Parallel processing (max {self.max_concurrent_chunks} concurrent)")
            all_candidates, chunk_times, errors = await self._process_chunks_parallel(chunks, allocations)
        else:
            print(f"\n🔄 Sequential processing")
            all_candidates, chunk_times, errors = await self._process_chunks_sequential(chunks, allocations)

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
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception]]:
        """
        Process chunks sequentially (memory-safe, default).
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
                errors.append(e)
            finally:
                chunk_end = time.perf_counter()
                chunk_times.append((chunk_end - chunk_start) * 1000.0)

        return all_candidates, chunk_times, errors

    async def _process_chunks_parallel(
        self,
        chunks: List[Chunk],
        allocations: List[int]
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception]]:
        """
        Process chunks in parallel with concurrency control.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)

        async def process_chunk_with_semaphore(
            chunk: Chunk,
            num_candidates: int,
            chunk_idx: int
        ) -> Tuple[int, List[QuestionCandidate], float, Optional[Exception]]:
            async with semaphore:
                chunk_start = time.perf_counter()
                try:
                    candidates = await self.adapter.generate_questions(
                        chunk=chunk,
                        num_candidates=num_candidates,
                        generation_config=self.config
                    )
                    chunk_end = time.perf_counter()
                    return chunk_idx, candidates, (chunk_end - chunk_start) * 1000.0, None
                except Exception as e:
                    chunk_end = time.perf_counter()
                    return chunk_idx, [], (chunk_end - chunk_start) * 1000.0, e

        tasks = [
            process_chunk_with_semaphore(chunk, num_candidates, idx)
            for idx, (chunk, num_candidates) in enumerate(zip(chunks, allocations))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        results.sort(key=lambda x: x[0])

        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []

        for _, candidates, elapsed_ms, error in results:
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
        """
        try:
            print(f"\n🔍 Post-processing:")
            print(f"   • Candidates generated: {len(candidates)}")
            print(f"   • Requested questions: {k}")

            validation_result = validate_candidates(candidates, chunks)

            if not validation_result.valid:
                raise ValidationError(
                    message="All candidates failed validation",
                    rejected_count=len(validation_result.rejected),
                    valid_count=0
                )

            print(f"   • Valid candidates: {len(validation_result.valid)}")

            deduplicated = deduplicate_candidates(validation_result.valid)
            ranked = rank_candidates(deduplicated, chunks)
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


__all__ = [
    "QuestionGenerationPipeline",
]
