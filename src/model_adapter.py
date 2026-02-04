"""
Edge Question Generation Pipeline - Model Adapter Interface
Abstract interface for model inference with concrete implementations
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import json

from src.pipeline_types import Chunk, QuestionCandidate, GenerationConfig, QuestionType, QuestionStyle
from src.errors import ModelInferenceError, InferenceErrorCause


# ============================================================================
# Model Adapter Interface
# ============================================================================

class ModelAdapter(ABC):
    """
    Abstract interface for model inference.
    
    Implementations handle runtime-specific details (WebGPU, CPU, etc.)
    while maintaining consistent API contract.
    """
    
    @abstractmethod
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        """
        Generate questions from a single chunk via non-streaming API call.
        
        Args:
            chunk: Text chunk to process
            num_candidates: Number of questions to generate
            generation_config: Generation parameters
            
        Returns:
            List of parsed and typed question candidates
            
        Raises:
            ModelInferenceError: On timeout, OOM, or JSON parse failure
        """
        pass
    
    @abstractmethod
    async def is_ready(self) -> bool:
        """
        Check if model is loaded and ready for inference.
        
        Returns:
            True if model is ready, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        """
        Estimate memory footprint for this model + config.
        
        Args:
            chunk: Chunk to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        pass


# ============================================================================
# Prompt Template
# ============================================================================

def build_prompt(
    chunk_text: str, 
    num_questions: int,
    config: GenerationConfig
) -> str:
    """
    Build a simplified prompt for question generation.
    
    Only generates questions - no metadata like type, style, or evidence.
    This makes it much more reliable for small models.
    
    Args:
        chunk_text: Text chunk to generate questions from
        num_questions: Number of questions to generate
        config: Generation configuration (mostly ignored for simplicity)
    """
    
    return f"""Based on this text, generate {num_questions} questions that can be answered from the content.

Return ONLY a JSON object in this exact format (no other text):
{{
  "questions": [
    "How do I enable conversion tracking?",
    "What types of conversions can be tracked?",
    "When should I use conversion tracking?"
  ]
}}

Text:
{chunk_text}

Return only the JSON object with {num_questions} questions:"""


# ============================================================================
# Response Parsing
# ============================================================================

def parse_model_response(
    response_text: str,
    chunk_id: int
) -> List[QuestionCandidate]:
    """
    Parse simplified model JSON response into QuestionCandidate objects.
    
    Expected format: {"questions": ["question1", "question2", ...]}
    
    Args:
        response_text: Raw model output
        chunk_id: Source chunk identifier
        
    Returns:
        List of parsed question candidates
        
    Raises:
        ModelInferenceError: On JSON parse failure or invalid structure
    """
    # Clean response (remove markdown code fences if present)
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ModelInferenceError(
            message=f"Failed to parse model response as JSON: {e}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )
    
    # Validate structure
    if not isinstance(data, dict):
        raise ModelInferenceError(
            message="Model response is not a JSON object",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )
    
    if "questions" not in data:
        raise ModelInferenceError(
            message="Model response missing 'questions' field",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )
    
    if not isinstance(data["questions"], list):
        raise ModelInferenceError(
            message="'questions' field is not a list",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )
    
    # Parse each question - now they're just strings
    candidates = []
    for i, question in enumerate(data["questions"]):
        try:
            # Handle both string format and dict format (for backward compatibility)
            if isinstance(question, str):
                question_text = question.strip()
                if not question_text:
                    continue
                
                # Default metadata
                candidates.append(QuestionCandidate(
                    question=question_text,
                    type=QuestionType.PROCEDURAL,  # Default type
                    evidence="",  # No evidence in simplified format
                    chunk_id=chunk_id,
                    style=QuestionStyle.INTERROGATIVE  # Default style
                ))
                
            elif isinstance(question, dict):
                # Legacy format support
                question_text = question.get("question", "").strip()
                q_type = question.get("type", "procedural").strip()
                q_style = question.get("style", "interrogative").strip()
                evidence = question.get("evidence", "").strip()
                
                if not question_text:
                    continue
                
                # Parse question type
                try:
                    question_type = QuestionType(q_type)
                except ValueError:
                    question_type = QuestionType.PROCEDURAL
                
                # Parse question style
                try:
                    question_style = QuestionStyle(q_style)
                except ValueError:
                    question_style = QuestionStyle.INTERROGATIVE
                
                candidates.append(QuestionCandidate(
                    question=question_text,
                    type=question_type,
                    evidence=evidence,
                    chunk_id=chunk_id,
                    style=question_style
                ))
            
        except Exception:
            # Skip malformed questions
            continue
    
    if not candidates:
        raise ModelInferenceError(
            message="No valid questions parsed from model response",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )
    
    return candidates


# ============================================================================
# Mock Adapter (for testing)
# ============================================================================

class MockAdapter(ModelAdapter):
    """
    Mock adapter for testing without actual model inference.
    
    Generates synthetic but valid question candidates.
    """
    
    def __init__(self, latency_ms: float = 100):
        self.latency_ms = latency_ms
        self._ready = True
    
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        """Generate mock questions based on chunk content"""
        import asyncio
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        # Generate synthetic questions with variety
        candidates = []
        
        # Define question templates for different types and styles
        templates = [
            ("How do I configure the feature mentioned in chunk {chunk_id}?", 
             QuestionType.PROCEDURAL, QuestionStyle.INTERROGATIVE),
            ("What is the purpose of the feature in chunk {chunk_id}?", 
             QuestionType.CONCEPTUAL, QuestionStyle.INTERROGATIVE),
            ("When should I use this feature from chunk {chunk_id}?", 
             QuestionType.TEMPORAL, QuestionStyle.INTERROGATIVE),
            ("Which option is best for the scenario in chunk {chunk_id}?", 
             QuestionType.COMPARATIVE, QuestionStyle.INTERROGATIVE),
            ("Explain the configuration process described in chunk {chunk_id}", 
             QuestionType.PROCEDURAL, QuestionStyle.IMPERATIVE),
        ]
        
        for i in range(num_candidates):
            template, q_type, q_style = templates[i % len(templates)]
            candidates.append(QuestionCandidate(
                question=template.format(chunk_id=chunk.chunk_id),
                type=q_type,
                evidence="",  # No evidence in simplified format
                chunk_id=chunk.chunk_id,
                style=q_style
            ))
        
        return candidates
    
    async def is_ready(self) -> bool:
        """Always ready for mock adapter"""
        return self._ready
    
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        """Estimate ~100MB for mock model"""
        return 100 * 1024 * 1024


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'ModelAdapter',
    'MockAdapter',
    'build_prompt',
    'parse_model_response',
]