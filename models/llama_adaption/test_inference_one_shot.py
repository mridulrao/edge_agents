#!/usr/bin/env python3
"""
One-shot inference test: generate N questions from an article and report:
  - Time to first token (TTFT)
  - Latency to generate the FIRST complete question
  - Total latency to generate ALL questions

Usage:
  python test_oneshot_inference.py --questions 4
  python test_oneshot_inference.py --questions 4 --article path/to/article.txt
  python test_oneshot_inference.py --questions 4 --text "Your article text here..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Script lives at models/llama_adaption/ — walk up to the project root so that
# imports like `from models.llama_adaption...` resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.llama_adaption.llama_server_adapter import create_adapter_with_server
from local_llm.pipeline import QuestionGenerationPipeline
from local_llm.pipeline_types import ArticleInput, GenerationConfig


# ---------------------------------------------------------------------------
# Timing container
# ---------------------------------------------------------------------------

@dataclass
class InferenceTimings:
    ttft_ms: Optional[float] = None              # ms until first token arrives
    first_question_ms: Optional[float] = None    # ms until first complete question
    total_ms: Optional[float] = None             # ms until all questions complete
    per_question_ms: Optional[List[float]] = None  # ms timestamp for each question
    questions: List[str] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("  ONE-SHOT INFERENCE RESULTS")
        print("=" * 60)
        print(f"  Questions generated : {len(self.questions)}")
        print(f"  TTFT                : {self.ttft_ms:.1f} ms" if self.ttft_ms is not None else "  TTFT                : n/a")
        print(f"  First question      : {self.first_question_ms:.1f} ms" if self.first_question_ms is not None else "  First question      : n/a")
        print(f"  Total latency       : {self.total_ms:.1f} ms" if self.total_ms is not None else "  Total latency       : n/a")
        if self.per_question_ms:
            per_q_str = ", ".join(f"Q{i+1}:{v:.0f}ms" for i, v in enumerate(self.per_question_ms))
            print(f"  Per-question        : [{per_q_str}]")
        if self.first_question_ms and self.total_ms and len(self.questions) > 1:
            avg_subsequent = (self.total_ms - self.first_question_ms) / (len(self.questions) - 1)
            print(f"  Avg subsequent q    : {avg_subsequent:.1f} ms")
        print("-" * 60)
        for i, q in enumerate(self.questions, 1):
            print(f"  Q{i}: {q}")
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        return {
            "ttft_ms": self.ttft_ms,
            "first_question_ms": self.first_question_ms,
            "total_ms": self.total_ms,
            "per_question_ms": self.per_question_ms,
            "questions_count": len(self.questions),
            "questions": self.questions,
        }


# ---------------------------------------------------------------------------
# Streaming wrapper that intercepts tokens to measure TTFT + per-question time
# ---------------------------------------------------------------------------

class TimedStreamAdapter:
    """
    Wraps the raw llama-server /completion streaming endpoint.
    Intercepts the token stream to record timing milestones, then
    passes the full assembled text to the pipeline parser.
    """

    def __init__(self, adapter, config: GenerationConfig):
        self.adapter = adapter
        self.config = config

    async def generate_timed(self, prompt: str, expected_questions: int) -> InferenceTimings:
        timings = InferenceTimings()
        collected_tokens: List[str] = []
        question_timestamps: List[float] = []
        t_start = time.perf_counter()

        # Detect question boundaries heuristically: numbered lines like "1.", "2.", etc.
        assembled = ""
        found_questions = 0

        async for token in self.adapter.stream_completion(
            prompt=prompt,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        ):
            now = time.perf_counter()

            # TTFT: record on very first token
            if timings.ttft_ms is None and token:
                timings.ttft_ms = (now - t_start) * 1000.0

            assembled += token
            collected_tokens.append(token)

            # Heuristic: count completed question lines (ends with "?" or newline after content)
            # Adjust the pattern to match your model's output format.
            current_q_count = _count_completed_questions(assembled)
            if current_q_count > found_questions:
                found_questions = current_q_count
                question_timestamps.append((now - t_start) * 1000.0)
                if found_questions == 1:
                    timings.first_question_ms = question_timestamps[0]
                if found_questions >= expected_questions:
                    break  # stop early once we have what we need

        t_end = time.perf_counter()
        timings.total_ms = (t_end - t_start) * 1000.0
        timings.questions = _parse_questions(assembled)

        return timings


def _count_completed_questions(text: str) -> int:
    """
    Counts lines that look like completed questions.
    Handles common model output formats:
      1. What is skateboarding?
      - What is skateboarding?
      Q1: What is skateboarding?
    """
    import re
    # Match lines ending with '?' that have some content
    pattern = re.compile(r"^\s*(?:\d+[.):-]|\-|Q\d+:)?\s*.+\?\s*$", re.MULTILINE)
    return len(pattern.findall(text))


def _parse_questions(text: str) -> List[str]:
    """Extract question strings from assembled model output."""
    import re
    pattern = re.compile(r"^\s*(?:\d+[.):-]|\-|Q\d+:)?\s*(.+\?)\s*$", re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(text)]


# ---------------------------------------------------------------------------
# Fallback: use pipeline directly (no per-token streaming available)
# ---------------------------------------------------------------------------

async def run_via_pipeline(
    adapter,
    article: ArticleInput,
    config: GenerationConfig,
) -> InferenceTimings:
    """
    Uses pipeline.generate_with_metrics(). Maps the actual metrics fields:
      - latency_ms             → total generation time
      - chunk_processing_times → list of per-chunk durations; cumulative sum
                                 gives per-question timestamps
    TTFT is not exposed by the pipeline and will remain n/a.
    """
    pipeline = QuestionGenerationPipeline(adapter=adapter, config=config)
    result = await pipeline.generate_with_metrics(article)
    m = result.metrics

    # Unwrap FinalQuestion objects → plain strings
    questions = []
    for q in result.questions:
        if isinstance(q, str):
            questions.append(q)
        elif hasattr(q, "question"):
            questions.append(q.question)
        else:
            questions.append(str(q))

    total_ms: Optional[float] = getattr(m, "latency_ms", None)

    # chunk_processing_times is a list of individual chunk durations (ms).
    # A cumulative sum gives the wall-clock timestamp for each question.
    chunk_times = getattr(m, "chunk_processing_times", None) or []
    per_question_ms: Optional[List[float]] = None
    first_question_ms: Optional[float] = None

    if chunk_times:
        cumulative = 0.0
        per_question_ms = []
        for t in chunk_times:
            cumulative += float(t)
            per_question_ms.append(round(cumulative, 1))
        # Trim to the number of final questions (pipeline may generate extra candidates)
        per_question_ms = per_question_ms[: len(questions)]
        if per_question_ms:
            first_question_ms = per_question_ms[0]

    return InferenceTimings(
        ttft_ms=None,          # not exposed by pipeline; needs raw token streaming
        first_question_ms=first_question_ms,
        total_ms=total_ms,
        per_question_ms=per_question_ms,
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_ARTICLE = """
Skateboarding began as a desperate attempt to catch a wave on land. In the late 1950s, surfers in
California sought a way to practice their balance when the ocean was calm. They nailed clay roller
skate wheels to wooden planks, creating a crude vehicle they called sidewalk surfing. These early
riders emulated the fluid, carving movements of the water, unaware they were laying the foundation
for a global cultural phenomenon.

The sport underwent a massive technological leap in the early 1970s with the invention of the
polyurethane wheel. Before this, wheels were made of metal or clay, offering no grip and turning
every pebble into a hazard. The new urethane wheels provided a smooth, grippy ride that allowed
skaters to reach higher speeds and perform tighter turns. This era also saw the legendary Z-Boys of
Santa Monica take advantage of a severe California drought by sneaking into empty swimming pools,
giving birth to aerial maneuvers and transforming the urban landscape into a giant playground.

By the 1980s, skateboarding had moved into the streets. Innovators like Alan Gelfand, who invented
the ollie, and Rodney Mullen, who pioneered flip tricks, turned the board into an extension of the
body. In 2021, skateboarding debuted at the Tokyo Olympics, yet it retains its soul as a creative
art form rooted in freedom and resilience.
""".strip()


async def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot inference latency test for llama-server.")
    ap.add_argument("--questions", type=int, default=4, help="Number of questions to generate")
    ap.add_argument("--article", type=str, default=None, help="Path to a plain-text article file")
    ap.add_argument("--text", type=str, default=None, help="Inline article text")
    ap.add_argument("--max-out", type=int, default=80, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--ctx", type=int, default=1048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to save results as JSON")
    ap.add_argument(
        "--no-stream",
        action="store_true",
        help="Skip streaming path and use pipeline directly (TTFT won't be measured)",
    )
    args = ap.parse_args()

    # Resolve article text
    if args.article:
        article_text = Path(args.article).read_text(encoding="utf-8").strip()
    elif args.text:
        article_text = args.text.strip()
    else:
        article_text = DEFAULT_ARTICLE

    article = ArticleInput(text=article_text, desired_questions=args.questions)
    config = GenerationConfig(
        max_output_tokens=args.max_out,
        temperature=args.temperature,
        top_p=1.0,
    )

    MODEL_PATH = "models/llama_adaption/models/LiquidAI_LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf"
    LLAMA_SERVER_PATH = "models/llama.cpp/build/bin/llama-server"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return 1

    print(f"⏳ Starting llama-server (port {args.port})…")
    adapter = await create_adapter_with_server(
        model_path=MODEL_PATH,
        host="127.0.0.1",
        port=args.port,
        n_gpu_layers=0,
        ctx_size=args.ctx,
        n_threads=args.threads,
        llama_server_path=LLAMA_SERVER_PATH,
    )
    print("✅ Server ready. Running inference…\n")

    try:
        # Try streaming path first for TTFT; fall back to pipeline if unavailable
        has_stream = not args.no_stream and hasattr(adapter, "stream_completion")

        if has_stream:
            timed = TimedStreamAdapter(adapter=adapter, config=config)
            # Build the prompt the same way your pipeline does — adjust if needed
            prompt = _build_prompt(article_text, args.questions)
            timings = await timed.generate_timed(prompt=prompt, expected_questions=args.questions)
        else:
            if args.no_stream:
                print("ℹ️  Streaming skipped (--no-stream). TTFT will not be measured.")
            else:
                print("ℹ️  Adapter has no stream_completion(). Falling back to pipeline.")
            timings = await run_via_pipeline(adapter=adapter, article=article, config=config)

        timings.print_report()

        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(
                json.dumps(timings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"📄 Results saved to {args.json_out}")

    finally:
        await adapter.shutdown()

    return 0


def _build_prompt(article_text: str, n_questions: int) -> str:
    """
    Minimal prompt that mirrors what QuestionGenerationPipeline likely sends.
    Replace with your actual prompt template if it differs.
    """
    return (
        f"Read the following article and generate exactly {n_questions} "
        f"comprehension questions. Output each question on its own line, "
        f"numbered 1 through {n_questions}. Questions only, no explanations.\n\n"
        f"Article:\n{article_text}\n\nQuestions:\n"
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))