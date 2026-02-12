"""
Type definitions for the entire pipeline system (SIMPLIFIED)

UPDATED:
✅ Removes unused/overly-specific fields (evidence checking, preferred words, etc.)
✅ Keeps only what the current pipeline + adapters actually use
✅ Adds optional mode metadata used by 4-mode benchmarking
✅ Keeps enums referenced across files
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class QuestionType(str, Enum):
    PROCEDURAL = "procedural"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    DIAGNOSTIC = "diagnostic"
    DEFINITIONAL = "definitional"


class QuestionStyle(str, Enum):
    INTERROGATIVE = "interrogative"
    IMPERATIVE = "imperative"
    CONTEXTUAL = "contextual"


class PipelineStage(str, Enum):
    PRE_GEN = "pre-gen"
    GENERATION = "generation"
    POST_GEN = "post-gen"


class InferenceErrorCause(str, Enum):
    TIMEOUT = "timeout"
    OOM = "out_of_memory"
    PARSE_ERROR = "parse_error"
    MODEL_NOT_READY = "model_not_ready"
    MODEL_LOAD_ERROR = "model_load_error"
    INSUFFICIENT_OUTPUT = "insufficient_output"
    UNKNOWN = "unknown"


# ============================================================================
# Input Types
# ============================================================================

@dataclass
class ArticleInput:
    """Input article for question generation"""
    text: str
    desired_questions: int = 4


# ============================================================================
# Pre-Generation Types
# ============================================================================

@dataclass
class Chunk:
    """Text chunk with metadata for processing"""
    chunk_id: int
    text: str
    start_offset: int = 0
    end_offset: int = 0

    def __len__(self) -> int:
        return len((self.text or "").split())


# ============================================================================
# Generation Types
# ============================================================================

@dataclass
class GenerationConfig:
    """
    Configuration for model generation.
    Keep only what pipeline/adapter uses.
    """
    max_output_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 0.9

    # IMPORTANT: default empty to avoid brace-truncation failures
    stop_sequences: List[str] = field(default_factory=list)

    # Used by pipeline for B/D fallback fill behavior
    enable_fallback_fill: bool = True

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0, 1], got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")


@dataclass
class QuestionCandidate:
    """Raw question candidate from model generation"""
    question: str
    type: QuestionType = QuestionType.PROCEDURAL
    chunk_id: int = 0
    style: QuestionStyle = QuestionStyle.INTERROGATIVE

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "type": self.type.value,
            "chunk_id": self.chunk_id,
            "style": self.style.value,
        }


# ============================================================================
# Post-Generation Types
# ============================================================================

@dataclass
class RejectedCandidate:
    candidate: QuestionCandidate
    reason: str


@dataclass
class ValidationResult:
    valid: List[QuestionCandidate]
    rejected: List[RejectedCandidate]

    @property
    def pass_rate(self) -> float:
        total = len(self.valid) + len(self.rejected)
        return (len(self.valid) / total) if total > 0 else 0.0


@dataclass
class RankedCandidate:
    candidate: QuestionCandidate
    score: float
    score_breakdown: Dict = field(default_factory=dict)


@dataclass
class FinalQuestion:
    question: str
    type: str
    source_chunk_id: int
    confidence_score: float

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "type": self.type,
            "source_chunk_id": self.source_chunk_id,
            "confidence_score": self.confidence_score,
        }


# ============================================================================
# Pipeline Metrics
# ============================================================================

@dataclass
class PipelineMetrics:
    chunks_created: int
    candidates_generated: int
    candidates_validated: int
    candidates_deduplicated: int
    latency_ms: float
    memory_peak_mb: float

    # Optional memory detail
    python_heap_peak_mb: Optional[float] = None
    server_pid: Optional[int] = None
    server_rss_before_mb: Optional[float] = None
    server_rss_after_mb: Optional[float] = None
    server_rss_peak_mb: Optional[float] = None
    server_rss_delta_mb: Optional[float] = None

    # Optional detailed metrics
    validation_pass_rate: float = 0.0
    deduplication_reduction: float = 0.0
    chunk_processing_times: List[float] = field(default_factory=list)

    # Optional mode metadata (safe defaults)
    mode_scope: Optional[str] = None
    mode_output_format: Optional[str] = None
    mode_questions_per_call: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "chunks_created": self.chunks_created,
            "candidates_generated": self.candidates_generated,
            "candidates_validated": self.candidates_validated,
            "candidates_deduplicated": self.candidates_deduplicated,
            "latency_ms": self.latency_ms,
            "memory_peak_mb": self.memory_peak_mb,

            "python_heap_peak_mb": self.python_heap_peak_mb,
            "server_pid": self.server_pid,
            "server_rss_before_mb": self.server_rss_before_mb,
            "server_rss_after_mb": self.server_rss_after_mb,
            "server_rss_peak_mb": self.server_rss_peak_mb,
            "server_rss_delta_mb": self.server_rss_delta_mb,

            "validation_pass_rate": self.validation_pass_rate,
            "deduplication_reduction": self.deduplication_reduction,
            "chunk_processing_times": self.chunk_processing_times,

            "mode_scope": self.mode_scope,
            "mode_output_format": self.mode_output_format,
            "mode_questions_per_call": self.mode_questions_per_call,
        }


@dataclass
class PipelineResult:
    questions: List[FinalQuestion]
    metrics: PipelineMetrics

    def to_dict(self) -> Dict:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "metrics": self.metrics.to_dict(),
        }


__all__ = [
    "QuestionType",
    "QuestionStyle",
    "PipelineStage",
    "InferenceErrorCause",
    "ArticleInput",
    "Chunk",
    "GenerationConfig",
    "QuestionCandidate",
    "RejectedCandidate",
    "ValidationResult",
    "RankedCandidate",
    "FinalQuestion",
    "PipelineMetrics",
    "PipelineResult",
]
