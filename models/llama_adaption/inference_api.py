#!/usr/bin/env python3
"""
FastAPI service for one-shot question generation with latency stats.

Start:
  uvicorn models.llama_adaption.inference_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /generate          — generate questions + return latency stats
  GET  /health            — server + model readiness check
  GET  /stats             — cumulative stats across all requests
  DELETE /stats           — reset cumulative stats
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

# Resolve project root from this file's location:
# inference_api.py → llama_adaption/ → models/ → project root
_APP_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_APP_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models.llama_adaption.llama_server_adapter import create_adapter_with_server
from local_llm.pipeline import QuestionGenerationPipeline
from local_llm.pipeline_types import ArticleInput, GenerationConfig


# ---------------------------------------------------------------------------
# Config — absolute paths derived from file location, works in any CWD
# ---------------------------------------------------------------------------

_APP_ROOT         = Path(__file__).resolve().parent.parent.parent
MODEL_PATH        = str(_APP_ROOT / "models/llama_adaption/models/LiquidAI_LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf")
LLAMA_SERVER_PATH = str(_APP_ROOT / "models/llama.cpp/build/bin/llama-server")
LLAMA_HOST        = "127.0.0.1"
LLAMA_PORT        = 8080
CTX_SIZE          = 1048
N_THREADS         = 4
N_GPU_LAYERS      = 0


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.adapter = None
        self.total_requests: int = 0
        self.total_questions_generated: int = 0
        self.total_latency_ms: float = 0.0
        self.total_first_question_ms: float = 0.0
        self.first_question_samples: int = 0
        self.errors: int = 0
        self.server_start_time: Optional[float] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start llama-server on boot, shut it down on exit."""
    print(f"📂 APP_ROOT   : {_APP_ROOT}")
    print(f"📂 MODEL_PATH : {MODEL_PATH}")
    print(f"📂 LLAMA_BIN  : {LLAMA_SERVER_PATH}")

    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    if not Path(LLAMA_SERVER_PATH).exists():
        raise RuntimeError(f"llama-server binary not found: {LLAMA_SERVER_PATH}")

    print(f"⏳ Starting llama-server on port {LLAMA_PORT}…")
    state.adapter = await create_adapter_with_server(
        model_path=MODEL_PATH,
        host=LLAMA_HOST,
        port=LLAMA_PORT,
        n_gpu_layers=N_GPU_LAYERS,
        ctx_size=CTX_SIZE,
        n_threads=N_THREADS,
        llama_server_path=LLAMA_SERVER_PATH,
    )
    state.server_start_time = time.time()
    print("✅ llama-server ready.")

    yield

    print("🛑 Shutting down llama-server…")
    if state.adapter:
        try:
            await state.adapter.shutdown()
        except Exception:
            pass


app = FastAPI(
    title="LLM Question Generation API",
    description="Generate comprehension questions from an article with latency profiling.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: str = Field(..., description="Article text to generate questions from", min_length=50)
    num_questions: int = Field(4, ge=1, le=20, description="Number of questions to generate")
    max_output_tokens: int = Field(300, ge=50, le=2048)
    temperature: float = Field(0.3, ge=0.0, le=2.0)


class LatencyStats(BaseModel):
    ttft_ms: Optional[float] = Field(None, description="Time to first token (ms) — n/a in pipeline mode")
    first_question_ms: Optional[float] = Field(None, description="Time until first complete question (ms)")
    total_ms: Optional[float] = Field(None, description="Total generation time (ms)")
    per_question_ms: Optional[List[float]] = Field(None, description="Cumulative ms timestamp per question")
    avg_subsequent_question_ms: Optional[float] = Field(
        None, description="Average ms between Q1 and final question"
    )


class GenerateResponse(BaseModel):
    questions: List[str]
    questions_count: int
    latency: LatencyStats
    model: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    llama_binary: str
    server_uptime_s: Optional[float]


class StatsResponse(BaseModel):
    total_requests: int
    total_questions_generated: int
    errors: int
    avg_total_latency_ms: Optional[float]
    avg_first_question_ms: Optional[float]
    server_uptime_s: Optional[float]


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _unwrap_questions(raw_questions) -> List[str]:
    result = []
    for q in raw_questions:
        if isinstance(q, str):
            result.append(q)
        elif hasattr(q, "question"):
            result.append(q.question)
        else:
            result.append(str(q))
    return result


async def _run_inference(request: GenerateRequest) -> GenerateResponse:
    article = ArticleInput(text=request.text, desired_questions=request.num_questions)
    config = GenerationConfig(
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=1.0,
    )

    pipeline = QuestionGenerationPipeline(adapter=state.adapter, config=config)
    result = await pipeline.generate_with_metrics(article)
    m = result.metrics

    questions = _unwrap_questions(result.questions)

    total_ms: Optional[float] = getattr(m, "latency_ms", None)
    chunk_times = getattr(m, "chunk_processing_times", None) or []

    per_question_ms: Optional[List[float]] = None
    first_question_ms: Optional[float] = None
    avg_subsequent: Optional[float] = None

    if chunk_times:
        cumulative = 0.0
        per_question_ms = []
        for t in chunk_times:
            cumulative += float(t)
            per_question_ms.append(round(cumulative, 1))
        per_question_ms = per_question_ms[: len(questions)]
        if per_question_ms:
            first_question_ms = per_question_ms[0]
        if first_question_ms and total_ms and len(questions) > 1:
            avg_subsequent = (total_ms - first_question_ms) / (len(questions) - 1)

    state.total_requests += 1
    state.total_questions_generated += len(questions)
    if total_ms is not None:
        state.total_latency_ms += total_ms
    if first_question_ms is not None:
        state.total_first_question_ms += first_question_ms
        state.first_question_samples += 1

    return GenerateResponse(
        questions=questions,
        questions_count=len(questions),
        latency=LatencyStats(
            ttft_ms=None,
            first_question_ms=first_question_ms,
            total_ms=total_ms,
            per_question_ms=per_question_ms,
            avg_subsequent_question_ms=round(avg_subsequent, 1) if avg_subsequent else None,
        ),
        model=Path(MODEL_PATH).name,
        request_id=str(uuid.uuid4())[:8],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if state.adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        return await _run_inference(request)
    except Exception as e:
        state.errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    uptime = (time.time() - state.server_start_time) if state.server_start_time else None
    return HealthResponse(
        status="ok" if state.adapter is not None else "starting",
        model_loaded=state.adapter is not None,
        model_path=MODEL_PATH,
        llama_binary=LLAMA_SERVER_PATH,
        server_uptime_s=round(uptime, 1) if uptime else None,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    uptime = (time.time() - state.server_start_time) if state.server_start_time else None
    return StatsResponse(
        total_requests=state.total_requests,
        total_questions_generated=state.total_questions_generated,
        errors=state.errors,
        avg_total_latency_ms=(
            round(state.total_latency_ms / state.total_requests, 1)
            if state.total_requests > 0 else None
        ),
        avg_first_question_ms=(
            round(state.total_first_question_ms / state.first_question_samples, 1)
            if state.first_question_samples > 0 else None
        ),
        server_uptime_s=round(uptime, 1) if uptime else None,
    )


@app.delete("/stats")
async def reset_stats():
    state.total_requests = 0
    state.total_questions_generated = 0
    state.total_latency_ms = 0.0
    state.total_first_question_ms = 0.0
    state.first_question_samples = 0
    state.errors = 0
    return {"message": "Stats reset."}