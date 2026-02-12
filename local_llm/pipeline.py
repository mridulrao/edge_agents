"""
Edge Question Generation Pipeline - Orchestration
End-to-end pipeline with parallel processing support

UPDATED (for 4 configurations A/B/C/D):
✅ Adds `mode` support to generate()/generate_with_metrics()
✅ Uses mode-aware pre_gen:
   - chunk_article_for_mode(article, mode)
   - allocate_candidates_for_mode(num_chunks, desired_questions, mode)
✅ Uses adapter.generate_questions_mode(...) when available (for FULL_ARTICLE + N modes)
   - Falls back to adapter.generate_questions(...) for CHUNKED + 1 modes
✅ Supports optional fallback fill for FULL_ARTICLE modes:
   - If B/D returns < desired_questions, fill remaining using CHUNKED mode of same output format
✅ Keeps RSS sampling + tracemalloc metrics unchanged
✅ Never reads GenerationConfig.output_format (mode owns scope/format/per_call)
"""

from __future__ import annotations

import time
import asyncio
from typing import List, Optional, Tuple
import tracemalloc

import psutil  # for server RSS tracking

from local_llm.pipeline_types import (
    ArticleInput,
    Chunk,
    QuestionCandidate,
    GenerationConfig,
    FinalQuestion,
    PipelineMetrics,
    PipelineResult,
)
from local_llm.errors import PipelineError, PipelineStage, ValidationError
from local_llm.model_adapter import (
    ModelAdapter,
    QuestionGenMode,
    GenerationScope,
    OutputFormat,
)
from local_llm.pre_generation import (
    chunk_article_for_mode,
    allocate_candidates_for_mode,
)
from local_llm.post_generation import (
    validate_candidates,
    deduplicate_candidates,
    rank_candidates,
    select_final_questions,
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
    peak = 0.0
    while not stop.is_set():
        try:
            peak = max(peak, _rss_mb(pid))
        except psutil.NoSuchProcess:
            break
        except Exception:
            pass
        await asyncio.sleep(interval_s)
    return peak


# ============================================================================
# Pipeline
# ============================================================================

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
        max_concurrent_chunks: int = 3,
    ):
        self.adapter = adapter
        self.config = config or GenerationConfig()
        self.parallel_processing = parallel_processing
        self.max_concurrent_chunks = max_concurrent_chunks

    async def generate(self, article: ArticleInput, mode: Optional[QuestionGenMode] = None) -> List[FinalQuestion]:
        """
        Generate questions from article.

        mode:
          - If None, defaults to Config A (chunked + JSON + 1)
        """
        result = await self.generate_with_metrics(article, mode=mode)
        return result.questions

    async def generate_with_metrics(self, article: ArticleInput, mode: Optional[QuestionGenMode] = None) -> PipelineResult:
        """
        Generate with detailed telemetry for debugging/benchmarking.

        IMPORTANT:
        - tracemalloc measures Python heap only (NOT llama-server/model memory).
        - We additionally track llama-server RSS if adapter exposes a PID.
        """
        mode = mode or QuestionGenMode.A_chunked_json_1()
        start_time = time.perf_counter()

        tracemalloc.start()

        # Server memory tracking
        server_pid: Optional[int] = None
        server_rss_before_mb: Optional[float] = None
        server_rss_after_mb: Optional[float] = None
        server_rss_peak_mb: Optional[float] = None

        rss_stop = asyncio.Event()
        rss_task: Optional[asyncio.Task] = None

        chunk_times: List[float] = []

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
                rss_task = asyncio.create_task(_sample_rss_peak(server_pid, rss_stop, interval_s=0.2))

            # ================================================================
            # Stage 1: Pre-Generation (mode-aware)
            # ================================================================
            chunks = self._pre_generation(article, mode)

            # ================================================================
            # Stage 2: Generation (mode-aware)
            # ================================================================
            candidates, chunk_times = await self._generation(chunks, article, mode)

            # ================================================================
            # Stage 3: Post-Generation (unchanged)
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

            # Prefer server RSS peak as "memory_peak_mb"
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

                # optional memory details
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

                # mode metadata
                mode_scope=mode.scope.value,
                mode_output_format=mode.output_format.value,
                mode_questions_per_call=mode.questions_per_call,
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
                recoverable=False,
            ) from e

    # ============================================================================
    # Stage 1: Pre-Generation (mode-aware)
    # ============================================================================

    def _pre_generation(self, article: ArticleInput, mode: QuestionGenMode) -> List[Chunk]:
        try:
            chunks = chunk_article_for_mode(article, mode)

            print(f"\n📦 Chunking complete:")
            print(f"   • Mode: scope={mode.scope.value}, format={mode.output_format.value}, per_call={mode.questions_per_call}")
            print(f"   • Desired questions: {article.desired_questions}")
            print(f"   • Chunks created: {len(chunks)}")

            if mode.scope == GenerationScope.CHUNKED:
                # keep your original print (even if the number isn't exact)
                print(f"   • Chunk size: ~200 words each (your current print; update if needed)")
            else:
                print(f"   • Using full article in one chunk")

            return chunks

        except Exception as e:
            raise PipelineError(
                message=f"Pre-generation failed: {str(e)}",
                stage=PipelineStage.PRE_GEN,
                recoverable=False,
            ) from e

    # ============================================================================
    # Stage 2: Generation (mode-aware)
    # ============================================================================

    async def _generation(
        self,
        chunks: List[Chunk],
        article: ArticleInput,
        mode: QuestionGenMode,
    ) -> Tuple[List[QuestionCandidate], List[float]]:
        desired_questions = int(article.desired_questions)
        allocations = allocate_candidates_for_mode(len(chunks), desired_questions, mode)

        if not await self.adapter.is_ready():
            raise PipelineError(
                message="Model adapter is not ready",
                stage=PipelineStage.GENERATION,
                recoverable=True,
            )

        # FULL_ARTICLE modes: single call + optional fallback fill
        if mode.scope == GenerationScope.FULL_ARTICLE:
            print(f"\n🧠 Full-article processing (single call)")
            return await self._process_full_article_single_call(
                full_article_chunk=chunks[0],
                mode=mode,
                desired_questions=desired_questions,
            )

        # CHUNKED modes: use sequential/parallel machinery
        if self.parallel_processing:
            print(f"\n⚡ Parallel processing (max {self.max_concurrent_chunks} concurrent)")
            all_candidates, chunk_times, errors = await self._process_chunks_parallel_mode(chunks, allocations, mode)
        else:
            print(f"\n🔄 Sequential processing")
            all_candidates, chunk_times, errors = await self._process_chunks_sequential_mode(chunks, allocations, mode)

        if not all_candidates:
            raise PipelineError(
                message=f"Generation failed for all chunks. Errors: {errors}",
                stage=PipelineStage.GENERATION,
                recoverable=False,
            )

        print(f"   • Generated {len(all_candidates)} candidate questions")
        if errors:
            print(f"   ⚠️  {len(errors)} chunks failed")

        return all_candidates, chunk_times

    async def _process_full_article_single_call(
        self,
        full_article_chunk: Chunk,
        mode: QuestionGenMode,
        desired_questions: int,
    ) -> Tuple[List[QuestionCandidate], List[float]]:
        """
        B/D path: one chunk, one call, yields N candidates.

        Optional fallback fill:
        - If fewer than desired_questions returned OR the full call fails,
          fill remainder using CHUNKED mode of the same output format (JSON->A, TEXT->C).
        """
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []

        # ---- Full-article single call
        t0 = time.perf_counter()
        try:
            if not hasattr(self.adapter, "generate_questions_mode"):
                raise PipelineError(
                    message="Adapter does not support generate_questions_mode() required for FULL_ARTICLE modes",
                    stage=PipelineStage.GENERATION,
                    recoverable=False,
                )

            cands = await self.adapter.generate_questions_mode(
                text=full_article_chunk.text,
                chunk_id=full_article_chunk.chunk_id,
                mode=mode,
                generation_config=self.config,
            )
            all_candidates.extend(cands)

        except Exception as e:
            errors.append(e)

        finally:
            t1 = time.perf_counter()
            chunk_times.append((t1 - t0) * 1000.0)

        # ---- Fallback fill
        enable_fill = bool(getattr(self.config, "enable_fallback_fill", True))
        if enable_fill and len(all_candidates) < desired_questions:
            missing = desired_questions - len(all_candidates)
            print(f"   ⚠️  Full-article returned {len(all_candidates)}/{desired_questions}. Filling missing {missing} via chunked mode...")

            fill_mode = (
                QuestionGenMode.A_chunked_json_1()
                if mode.output_format == OutputFormat.JSON
                else QuestionGenMode.C_chunked_text_1()
            )

            fill_article = ArticleInput(text=full_article_chunk.text, desired_questions=missing)
            fill_chunks = chunk_article_for_mode(fill_article, fill_mode)
            fill_allocs = allocate_candidates_for_mode(len(fill_chunks), missing, fill_mode)

            fill_candidates, fill_times, fill_errors = await self._process_chunks_sequential_mode(fill_chunks, fill_allocs, fill_mode)
            all_candidates.extend(fill_candidates)
            chunk_times.extend(fill_times)
            errors.extend(fill_errors)

        # If STILL nothing, raise a clear generation error (instead of post-gen “0 candidates”)
        if not all_candidates:
            raise PipelineError(
                message=f"Full-article generation produced 0 candidates. Errors: {errors}",
                stage=PipelineStage.GENERATION,
                recoverable=False,
            )

        return all_candidates, chunk_times

    async def _process_chunks_sequential_mode(
        self,
        chunks: List[Chunk],
        allocations: List[int],
        mode: QuestionGenMode,
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception]]:
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []

        for chunk, num_candidates in zip(chunks, allocations):
            t0 = time.perf_counter()
            try:
                if hasattr(self.adapter, "generate_questions_mode"):
                    candidates = await self.adapter.generate_questions_mode(
                        text=chunk.text,
                        chunk_id=chunk.chunk_id,
                        mode=mode,
                        generation_config=self.config,
                    )
                else:
                    # Back-compat: only correct for CHUNKED + 1
                    candidates = await self.adapter.generate_questions(
                        chunk=chunk,
                        num_candidates=num_candidates,
                        generation_config=self.config,
                    )
                all_candidates.extend(candidates)
            except Exception as e:
                errors.append(e)
            finally:
                t1 = time.perf_counter()
                chunk_times.append((t1 - t0) * 1000.0)

        return all_candidates, chunk_times, errors

    async def _process_chunks_parallel_mode(
        self,
        chunks: List[Chunk],
        allocations: List[int],
        mode: QuestionGenMode,
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception]]:
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)

        async def worker(chunk: Chunk, num_candidates: int, idx: int) -> Tuple[int, List[QuestionCandidate], float, Optional[Exception]]:
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    if hasattr(self.adapter, "generate_questions_mode"):
                        candidates = await self.adapter.generate_questions_mode(
                            text=chunk.text,
                            chunk_id=chunk.chunk_id,
                            mode=mode,
                            generation_config=self.config,
                        )
                    else:
                        candidates = await self.adapter.generate_questions(
                            chunk=chunk,
                            num_candidates=num_candidates,
                            generation_config=self.config,
                        )
                    t1 = time.perf_counter()
                    return idx, candidates, (t1 - t0) * 1000.0, None
                except Exception as e:
                    t1 = time.perf_counter()
                    return idx, [], (t1 - t0) * 1000.0, e

        tasks = [worker(c, n, i) for i, (c, n) in enumerate(zip(chunks, allocations))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        results.sort(key=lambda x: x[0])

        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = []
        errors: List[Exception] = []

        for _, cands, elapsed_ms, err in results:
            chunk_times.append(elapsed_ms)
            if err:
                errors.append(err)
            else:
                all_candidates.extend(cands)

        return all_candidates, chunk_times, errors

    # ============================================================================
    # Stage 3: Post-Generation (unchanged)
    # ============================================================================

    def _post_generation(
        self,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
        k: int,
    ) -> List[FinalQuestion]:
        try:
            print(f"\n🔍 Post-processing:")
            print(f"   • Candidates generated: {len(candidates)}")
            print(f"   • Requested questions: {k}")

            validation_result = validate_candidates(candidates, chunks)

            if not validation_result.valid:
                raise ValidationError(
                    message="All candidates failed validation",
                    rejected_count=len(validation_result.rejected),
                    valid_count=0,
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
                recoverable=False,
            ) from e


__all__ = [
    "QuestionGenerationPipeline",
]
