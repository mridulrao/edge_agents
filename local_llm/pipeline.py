"""
Edge Question Generation Pipeline - Orchestration
End-to-end pipeline with parallel processing support

UPDATED:
✅ Early-stop generation once we have enough valid+deduped candidates (don’t process extra buffered chunks)
✅ Clear latency metrics:
   - time_to_first_question_ms  (TTFQ)
   - time_to_all_questions_ms   (TTK)
   - time_to_each_question_ms   ([t1..tK])
✅ Parallel mode uses as_completed + cancels remaining tasks on early-stop
✅ Keeps existing tracemalloc + llama-server RSS sampling
✅ Safe metrics construction: only passes fields that exist on PipelineMetrics (no TypeError)
"""

import time
import asyncio
import tracemalloc
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import List, Optional, Tuple, Dict

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
from local_llm.model_adapter import ModelAdapter
from local_llm.pre_generation import chunk_article, allocate_candidates
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
            pass
        await asyncio.sleep(interval_s)
    return peak


# ============================================================================
# Progress tracking (latency per question)
# ============================================================================

@dataclass
class GenerationProgress:
    """
    Tracks when the pipeline becomes capable of returning i questions (i=1..K),
    where "ready" = valid + deduped candidates.
    """
    t_start: float
    k: int
    time_to_each_question_ms: List[Optional[float]]  # len K
    time_to_first_question_ms: Optional[float] = None
    time_to_all_questions_ms: Optional[float] = None
    ready_count_history: List[Tuple[int, float]] = field(default_factory=list)  # (ready_count, elapsed_ms)


# ============================================================================
# Safe PipelineMetrics construction (avoid breaking if fields not added yet)
# ============================================================================

def _safe_pipeline_metrics_kwargs(**kwargs) -> dict:
    """
    Only include keys that exist in PipelineMetrics, to avoid TypeError when the
    dataclass hasn't been updated yet.
    """
    try:
        allowed = {f.name for f in dataclass_fields(PipelineMetrics)}
    except Exception:
        # Fallback: best effort if PipelineMetrics isn't a dataclass
        allowed = set(getattr(PipelineMetrics, "__annotations__", {}).keys())

    return {k: v for k, v in kwargs.items() if k in allowed}


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

        # Early-stop controls
        early_stop: bool = True,
        early_stop_margin: int = 2,
        min_chunks_before_stop: int = 1,

        # Verbose progress prints (ready count as chunks complete)
        print_progress: bool = False,
    ):
        """
        Args:
            adapter: Model adapter for inference
            config: Generation configuration (uses defaults if not provided)
            parallel_processing: Enable parallel chunk processing for speed
            max_concurrent_chunks: Maximum concurrent chunk requests (if parallel=True)

            early_stop: If True, stop generating once enough candidates are collected
            early_stop_margin: Extra deduped+valid candidates to buffer for failures/dedup
            min_chunks_before_stop: Don't stop before at least this many chunks completed

            print_progress: If True, prints ready_count + elapsed as generation progresses
        """
        self.adapter = adapter
        self.config = config or GenerationConfig()
        self.parallel_processing = parallel_processing
        self.max_concurrent_chunks = max_concurrent_chunks

        self.early_stop = early_stop
        self.early_stop_margin = max(0, int(early_stop_margin))
        self.min_chunks_before_stop = max(1, int(min_chunks_before_stop))

        self.print_progress = print_progress

    async def generate(self, article: ArticleInput) -> List[FinalQuestion]:
        result = await self.generate_with_metrics(article)
        return result.questions

    async def generate_with_metrics(self, article: ArticleInput) -> PipelineResult:
        """
        Generate with detailed telemetry for debugging/benchmarking.

        IMPORTANT:
        - tracemalloc measures Python heap only (NOT llama-server/model memory).
        - We additionally track llama-server RSS (resident memory) if the adapter exposes a PID.
        """
        pipeline_start = time.perf_counter()

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

                rss_task = asyncio.create_task(_sample_rss_peak(server_pid, rss_stop, interval_s=0.2))

            # ================================================================
            # Stage 1: Pre-Generation
            # ================================================================
            chunks = self._pre_generation(article)

            # ================================================================
            # Stage 2: Generation (with early stop + per-question latency)
            # ================================================================
            candidates, chunk_times, gen_progress = await self._generation(
                chunks, article.desired_questions
            )

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
            # Metrics collection
            # ================================================================
            pipeline_end = time.perf_counter()
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

            # Print latency breakdown clearly
            self._print_latency_breakdown(article.desired_questions, gen_progress)

            # Build metrics (only pass fields that exist on PipelineMetrics)
            metrics_kwargs = _safe_pipeline_metrics_kwargs(
                chunks_created=len(chunks),
                candidates_generated=total_candidates,
                candidates_validated=validated_count,
                candidates_deduplicated=deduplicated_count,
                latency_ms=(pipeline_end - pipeline_start) * 1000.0,
                memory_peak_mb=memory_peak_mb,
                validation_pass_rate=validation_result.pass_rate,
                deduplication_reduction=(
                    (validated_count - deduplicated_count) / validated_count
                    if validated_count > 0 else 0.0
                ),
                chunk_processing_times=chunk_times,

                # Optional / NEW fields (safe-filtered)
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

                # Per-question latency (safe-filtered; add to PipelineMetrics if you want them stored)
                time_to_first_question_ms=gen_progress.time_to_first_question_ms,
                time_to_all_questions_ms=gen_progress.time_to_all_questions_ms,
                time_to_each_question_ms=gen_progress.time_to_each_question_ms,
            )

            metrics = PipelineMetrics(**metrics_kwargs)
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
    # Latency breakdown printing
    # ============================================================================

    def _print_latency_breakdown(self, k: int, prog: GenerationProgress) -> None:
        print("\n📊 Latency breakdown (generation readiness):")
        if k <= 0:
            print("   • desired_questions=0 (skipping)")
            return

        if prog.time_to_first_question_ms is not None:
            print(f"   • time_to_first_question_ms: {prog.time_to_first_question_ms:.1f}")
        else:
            print("   • time_to_first_question_ms: None")

        if prog.time_to_all_questions_ms is not None:
            print(f"   • time_to_all_questions_ms:  {prog.time_to_all_questions_ms:.1f}")
        else:
            print("   • time_to_all_questions_ms:  None")

        steps = []
        for i in range(k):
            t = prog.time_to_each_question_ms[i]
            steps.append(f"{i+1}:{t:.0f}ms" if t is not None else f"{i+1}:None")
        print(f"   • time_to_each_question_ms: [{', '.join(steps)}]")

    # ============================================================================
    # Early-stop readiness check + progress updates
    # ============================================================================

    def _init_progress(self, k: int) -> GenerationProgress:
        k = max(1, int(k))
        return GenerationProgress(
            t_start=time.perf_counter(),
            k=k,
            time_to_each_question_ms=[None] * k,
        )

    def _ready_candidate_count(self, candidates: List[QuestionCandidate], chunks: List[Chunk]) -> int:
        """
        Returns count of candidates that are (valid + deduplicated).
        Best proxy for "can we actually return N questions now?"
        """
        try:
            vr = validate_candidates(candidates, chunks)
            if not vr.valid:
                return 0
            deduped = deduplicate_candidates(vr.valid)
            return len(deduped)
        except Exception:
            return 0

    def _update_progress(
        self,
        prog: GenerationProgress,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
    ) -> int:
        """
        Updates progress timestamps based on current candidates.
        Returns current ready_count (valid+deduped).
        """
        ready = self._ready_candidate_count(candidates, chunks)
        elapsed_ms = (time.perf_counter() - prog.t_start) * 1000.0
        prog.ready_count_history.append((ready, elapsed_ms))

        upto = min(ready, prog.k)
        for i in range(1, upto + 1):
            if prog.time_to_each_question_ms[i - 1] is None:
                prog.time_to_each_question_ms[i - 1] = elapsed_ms

        if ready >= 1 and prog.time_to_first_question_ms is None:
            prog.time_to_first_question_ms = elapsed_ms

        if ready >= prog.k and prog.time_to_all_questions_ms is None:
            prog.time_to_all_questions_ms = elapsed_ms

        if self.print_progress:
            print(f"   ⏱️ ready={ready}/{prog.k} elapsed={elapsed_ms:.1f}ms")

        return ready

    def _should_stop_early(
        self,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
        k: int,
        chunks_completed: int,
    ) -> bool:
        if not self.early_stop:
            return False
        if chunks_completed < self.min_chunks_before_stop:
            return False

        target = k + self.early_stop_margin
        ready = self._ready_candidate_count(candidates, chunks)
        return ready >= target

    # ============================================================================
    # Stages
    # ============================================================================

    def _pre_generation(self, article: ArticleInput) -> List[Chunk]:
        try:
            chunks = chunk_article(article)

            print(f"\n📦 Chunking complete:")
            print(f"   • Desired questions: {article.desired_questions}")
            print(f"   • Chunks created: {len(chunks)} (buffered)")
            print(f"   • Chunk size: ~200 words each")

            return chunks
        except Exception as e:
            raise PipelineError(
                message=f"Pre-generation failed: {str(e)}",
                stage=PipelineStage.PRE_GEN,
                recoverable=False,
            ) from e

    async def _generation(
        self,
        chunks: List[Chunk],
        desired_questions: int,
    ) -> Tuple[List[QuestionCandidate], List[float], GenerationProgress]:
        """
        Generation stage: invoke model for each chunk.

        Returns:
          - candidates
          - chunk_times (ms per chunk index; 0 for unused chunks)
          - GenerationProgress (TTFQ / TTK / per-question curve)
        """
        allocations = allocate_candidates(len(chunks), desired_questions)

        if not await self.adapter.is_ready():
            raise PipelineError(
                message="Model adapter is not ready",
                stage=PipelineStage.GENERATION,
                recoverable=True,
            )

        progress = self._init_progress(desired_questions)

        if self.parallel_processing:
            print(f"\n⚡ Parallel processing (max {self.max_concurrent_chunks} concurrent)")
            all_candidates, chunk_times, errors, progress = await self._process_chunks_parallel_early_stop(
                chunks, allocations, desired_questions, progress
            )
        else:
            print(f"\n🔄 Sequential processing")
            all_candidates, chunk_times, errors, progress = await self._process_chunks_sequential_early_stop(
                chunks, allocations, desired_questions, progress
            )

        if not all_candidates:
            raise PipelineError(
                message=f"Generation failed for all chunks. Errors: {errors}",
                stage=PipelineStage.GENERATION,
                recoverable=False,
            )

        print(f"   • Generated {len(all_candidates)} candidate questions")
        if errors:
            print(f"   ⚠️  {len(errors)} chunks failed")

        return all_candidates, chunk_times, progress

    async def _process_chunks_sequential_early_stop(
        self,
        chunks: List[Chunk],
        allocations: List[int],
        desired_questions: int,
        progress: GenerationProgress,
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception], GenerationProgress]:
        """
        Sequential chunk processing with early-stop + progress updates.
        """
        all_candidates: List[QuestionCandidate] = []
        chunk_times: List[float] = [0.0 for _ in range(len(chunks))]
        errors: List[Exception] = []

        chunks_completed = 0

        for idx, (chunk, num_candidates) in enumerate(zip(chunks, allocations)):
            if num_candidates <= 0:
                continue

            t0 = time.perf_counter()
            try:
                cands = await self.adapter.generate_questions(
                    chunk=chunk,
                    num_candidates=num_candidates,
                    generation_config=self.config,
                )
                all_candidates.extend(cands)
            except Exception as e:
                errors.append(e)
            finally:
                t1 = time.perf_counter()
                chunk_times[idx] = (t1 - t0) * 1000.0
                chunks_completed += 1

            # Update latency progress after each chunk completes
            self._update_progress(progress, all_candidates, chunks)

            # Early stop if we have enough (valid+deduped) candidates
            if self._should_stop_early(all_candidates, chunks, desired_questions, chunks_completed):
                ready = self._ready_candidate_count(all_candidates, chunks)
                target = desired_questions + self.early_stop_margin
                print(
                    f"   ✅ Early stop (sequential): ready={ready} target={target} "
                    f"after {chunks_completed} chunk(s)"
                )
                break

        return all_candidates, chunk_times, errors, progress

    async def _process_chunks_parallel_early_stop(
        self,
        chunks: List[Chunk],
        allocations: List[int],
        desired_questions: int,
        progress: GenerationProgress,
    ) -> Tuple[List[QuestionCandidate], List[float], List[Exception], GenerationProgress]:
        """
        Parallel chunk processing with concurrency control, early-stop, and task cancellation.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_chunks)

        chunk_times: List[float] = [0.0 for _ in range(len(chunks))]
        all_candidates: List[QuestionCandidate] = []
        errors: List[Exception] = []

        scheduled: List[Tuple[int, Chunk, int]] = [
            (idx, chunks[idx], allocations[idx])
            for idx in range(len(chunks))
            if allocations[idx] > 0
        ]

        async def run_one(idx: int, chunk: Chunk, num_candidates: int) -> Tuple[int, List[QuestionCandidate], float, Optional[Exception]]:
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    cands = await self.adapter.generate_questions(
                        chunk=chunk,
                        num_candidates=num_candidates,
                        generation_config=self.config,
                    )
                    t1 = time.perf_counter()
                    return idx, cands, (t1 - t0) * 1000.0, None
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    t1 = time.perf_counter()
                    return idx, [], (t1 - t0) * 1000.0, e

        tasks_by_idx: Dict[int, asyncio.Task] = {
            idx: asyncio.create_task(run_one(idx, ch, n))
            for (idx, ch, n) in scheduled
        }
        pending: set[asyncio.Task] = set(tasks_by_idx.values())

        chunks_completed = 0

        try:
            for fut in asyncio.as_completed(pending):
                idx, cands, elapsed_ms, err = await fut
                chunk_times[idx] = elapsed_ms
                chunks_completed += 1

                if err is not None:
                    errors.append(err)
                else:
                    all_candidates.extend(cands)

                # Update latency progress after each completed task
                self._update_progress(progress, all_candidates, chunks)

                # Early stop → cancel remaining tasks
                if self._should_stop_early(all_candidates, chunks, desired_questions, chunks_completed):
                    ready = self._ready_candidate_count(all_candidates, chunks)
                    target = desired_questions + self.early_stop_margin
                    print(
                        f"   ✅ Early stop (parallel): ready={ready} target={target} "
                        f"after {chunks_completed} chunk(s) (cancelling remaining)"
                    )

                    for t in tasks_by_idx.values():
                        if not t.done():
                            t.cancel()

                    await asyncio.gather(*tasks_by_idx.values(), return_exceptions=True)
                    break

        except Exception:
            for t in tasks_by_idx.values():
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks_by_idx.values(), return_exceptions=True)
            raise

        return all_candidates, chunk_times, errors, progress

    def _post_generation(
        self,
        candidates: List[QuestionCandidate],
        chunks: List[Chunk],
        k: int,
    ) -> List[FinalQuestion]:
        """
        Post-generation: validate → dedup → rank → select top K.
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


__all__ = ["QuestionGenerationPipeline"]