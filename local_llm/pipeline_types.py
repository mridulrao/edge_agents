"""
Type definitions for the entire pipeline system
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class QuestionType(str, Enum):
    """Types of questions that can be generated"""
    PROCEDURAL = "procedural"              # How to do X
    TROUBLESHOOTING = "troubleshooting"    # How to fix/resolve X
    CONFIGURATION = "configuration"        # How to configure/set up X
    CONCEPTUAL = "conceptual"              # What is X, understanding concepts
    TEMPORAL = "temporal"                  # When to do X, timing-related
    COMPARATIVE = "comparative"            # Which X to choose, comparisons
    DIAGNOSTIC = "diagnostic"              # What causes X, root cause
    DEFINITIONAL = "definitional"          # What does X mean, definitions


class QuestionStyle(str, Enum):
    """Question formulation styles"""
    INTERROGATIVE = "interrogative"        # Traditional question format (How/What/When/Which/Why)
    IMPERATIVE = "imperative"              # Command-like (Explain..., Describe..., List...)
    CONTEXTUAL = "contextual"              # Derived from content without question words


class PipelineStage(str, Enum):
    """Pipeline execution stages"""
    PRE_GEN = "pre-gen"
    GENERATION = "generation"
    POST_GEN = "post-gen"


class InferenceErrorCause(Enum):
    """Categorizes different types of inference failures."""
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
    start_offset: int
    end_offset: int
    
    def __len__(self) -> int:
        """Return word count of chunk"""
        return len(self.text.split())


# ============================================================================
# Generation Types
# ============================================================================

@dataclass
class QuestionCandidate:
    """Raw question candidate from model generation"""
    question: str
    type: QuestionType
    evidence: str
    chunk_id: int
    style: QuestionStyle = QuestionStyle.INTERROGATIVE
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "question": self.question,
            "type": self.type.value,
            "evidence": self.evidence,
            "chunk_id": self.chunk_id,
            "style": self.style.value
        }


@dataclass
class GenerationConfig:
    """Configuration for model generation"""
    max_output_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=lambda: ["}]", "\n\n\n"])
    
    # Question generation preferences
    question_types: List[QuestionType] = field(default_factory=lambda: [
        QuestionType.PROCEDURAL,
        QuestionType.CONFIGURATION,
        QuestionType.CONCEPTUAL
    ])
    question_styles: List[QuestionStyle] = field(default_factory=lambda: [
        QuestionStyle.INTERROGATIVE,
        QuestionStyle.CONTEXTUAL
    ])
    preferred_question_words: List[str] = field(default_factory=lambda: [
        "how", "what", "when", "which", "why", "where"
    ])
    allow_non_question_format: bool = True  # Allow imperative/contextual styles
    
    def __post_init__(self):
        """Validate configuration values"""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0, 1], got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")


# ============================================================================
# Post-Generation Types
# ============================================================================

@dataclass
class RejectedCandidate:
    """Candidate that failed validation with reason"""
    candidate: QuestionCandidate
    reason: str


@dataclass
class ValidationResult:
    """Result of candidate validation"""
    valid: List[QuestionCandidate]
    rejected: List[RejectedCandidate]
    
    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate"""
        total = len(self.valid) + len(self.rejected)
        return len(self.valid) / total if total > 0 else 0.0


@dataclass
class RankedCandidate:
    """Question candidate with ranking score"""
    candidate: QuestionCandidate
    score: float
    score_breakdown: dict = field(default_factory=dict)
    
    def __lt__(self, other: 'RankedCandidate') -> bool:
        """Enable sorting by score (descending)"""
        return self.score > other.score


@dataclass
class FinalQuestion:
    """Final selected question with metadata"""
    question: str
    type: str
    source_chunk_id: int
    confidence_score: float  # 0-1, from ranking
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "question": self.question,
            "type": self.type,
            "source_chunk_id": self.source_chunk_id,
            "confidence_score": self.confidence_score
        }


# ============================================================================
# Pipeline Metrics
# ============================================================================

@dataclass
class PipelineMetrics:
    """Telemetry data from pipeline execution"""
    chunks_created: int
    candidates_generated: int
    candidates_validated: int
    candidates_deduplicated: int
    latency_ms: float
    memory_peak_mb: float
    python_heap_peak_mb: float | None = None
    server_pid: int | None = None
    server_rss_before_mb: float | None = None
    server_rss_after_mb: float | None = None
    server_rss_peak_mb: float | None = None
    server_rss_delta_mb: float | None = None

    
    # Optional detailed metrics
    validation_pass_rate: float = 0.0
    deduplication_reduction: float = 0.0
    chunk_processing_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "chunks_created": self.chunks_created,
            "candidates_generated": self.candidates_generated,
            "candidates_validated": self.candidates_validated,
            "candidates_deduplicated": self.candidates_deduplicated,
            "latency_ms": self.latency_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "validation_pass_rate": self.validation_pass_rate,
            "deduplication_reduction": self.deduplication_reduction
        }


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    questions: List[FinalQuestion]
    metrics: PipelineMetrics
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "questions": [q.to_dict() for q in self.questions],
            "metrics": self.metrics.to_dict()
        }