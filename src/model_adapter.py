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
    Build the few-shot prompt for question generation.
    
    This prompt supports multiple question types and styles based on config:
    - Question types: procedural, conceptual, temporal, comparative, etc.
    - Question styles: interrogative, imperative, contextual
    - Flexible question words: how, what, when, which, why, where
    - Evidence grounding in chunk text
    
    Args:
        chunk_text: Text chunk to generate questions from
        num_questions: Number of questions to generate
        config: Generation configuration with preferences
    """
    # Build question type guidance
    type_examples = []
    if QuestionType.PROCEDURAL in config.question_types:
        type_examples.append('- Procedural: "How do I enable conversion tracking?"')
    if QuestionType.CONCEPTUAL in config.question_types:
        type_examples.append('- Conceptual: "What is conversion tracking?"')
    if QuestionType.TEMPORAL in config.question_types:
        type_examples.append('- Temporal: "When should I use conversion tracking?"')
    if QuestionType.COMPARATIVE in config.question_types:
        type_examples.append('- Comparative: "Which conversion type should I choose?"')
    if QuestionType.DIAGNOSTIC in config.question_types:
        type_examples.append('- Diagnostic: "What causes conversion tracking failures?"')
    if QuestionType.DEFINITIONAL in config.question_types:
        type_examples.append('- Definitional: "What does CPA mean in this context?"')
    
    type_guidance = "\n".join(type_examples) if type_examples else "- Generate questions of any relevant type"
    
    # Build style guidance
    style_examples = []
    if QuestionStyle.INTERROGATIVE in config.question_styles:
        style_examples.append(f'- Start with: {", ".join(config.preferred_question_words)}')
    if QuestionStyle.IMPERATIVE in config.question_styles:
        style_examples.append('- Imperative form: "Explain how to...", "Describe the process..."')
    if QuestionStyle.CONTEXTUAL in config.question_styles:
        style_examples.append('- Contextual: "Steps to configure X", "Process for setting up Y"')
    
    style_guidance = "\n".join(style_examples) if style_examples else "- Use standard question format"
    
    # Build allowed question words list
    question_words = ", ".join(f'"{w}"' for w in config.preferred_question_words)
    
    return f"""Given this article chunk from a knowledge base, generate diverse questions that can be answered from this chunk.

Question Types to Generate:
{type_guidance}

Question Styles:
{style_guidance}

Requirements:
- Generate questions answerable from this chunk only
- Vary question types and styles for diversity
- Include evidence snippet from the chunk for each question
- Each question should be clear, specific, and useful for knowledge retrieval
{f"- Prefer starting with: {question_words}" if config.preferred_question_words else ""}
{f"- Non-question formats allowed (e.g., 'Explain...', 'Steps to...')" if config.allow_non_question_format else "- All questions must be in interrogative form"}

Example Output:
{{
  "questions": [
    {{
      "question": "How do I enable conversion tracking in the platform?",
      "type": "procedural",
      "style": "interrogative",
      "evidence": "navigate to Tools > Conversions. Click the + button and select the conversion type"
    }},
    {{
      "question": "What types of conversions can be tracked?",
      "type": "conceptual",
      "style": "interrogative",
      "evidence": "You can track web conversions, app installs, or phone calls"
    }},
    {{
      "question": "When should different conversion types be used?",
      "type": "temporal",
      "style": "interrogative",
      "evidence": "Each type requires different setup steps"
    }},
    {{
      "question": "Which conversion tracking method is best for mobile apps?",
      "type": "comparative",
      "style": "interrogative",
      "evidence": "app installs or phone calls. Each type requires different setup steps"
    }}
  ]
}}

Now generate {num_questions} diverse questions for this chunk:

Chunk: {chunk_text}

Return only valid JSON in the format above. Do not include any other text or markdown formatting."""


# ============================================================================
# Response Parsing
# ============================================================================

def parse_model_response(
    response_text: str,
    chunk_id: int
) -> List[QuestionCandidate]:
    """
    Parse model JSON response into QuestionCandidate objects.
    
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
    
    # Parse each question
    candidates = []
    for i, q_data in enumerate(data["questions"]):
        try:
            # Validate required fields
            if not isinstance(q_data, dict):
                continue
            
            question = q_data.get("question", "").strip()
            q_type = q_data.get("type", "procedural").strip()
            q_style = q_data.get("style", "interrogative").strip()
            evidence = q_data.get("evidence", "").strip()
            
            if not question or not evidence:
                continue
            
            # Parse question type
            try:
                question_type = QuestionType(q_type)
            except ValueError:
                # Default to procedural if type is invalid
                question_type = QuestionType.PROCEDURAL
            
            # Parse question style
            try:
                question_style = QuestionStyle(q_style)
            except ValueError:
                # Default to interrogative if style is invalid
                question_style = QuestionStyle.INTERROGATIVE
            
            candidates.append(QuestionCandidate(
                question=question,
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
        words = chunk.text.split()[:50]  # First 50 words
        evidence_snippet = ' '.join(words[:20])
        
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
                evidence=evidence_snippet,
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