"""
Edge Question Generation Pipeline - Model Adapter Interface
Optimized for instruction-tuned models (SmolLM2, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import json
import re

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
            num_candidates: Number of questions to generate (always 1)
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
# Prompt Template - Chat Messages Format for Instruct Models
# ============================================================================

def build_prompt(chunk_text: str, num_candidates: int = 1, config: GenerationConfig = None):
    return [
        {
            "role": "system",
            "content": (
                "You are a strict JSON generator. Output ONLY one JSON object and nothing else. {\"question\":\"Insert your question here...\"}"
            )
        },
        {
            "role": "user",
            "content": (
                "Write ONE question from the given text.\n\n"
                "Text:\n"
                f"{chunk_text}\n\n"
                "Return JSON now like this - {'question':'Insert your question here...'}"
            )
        }
    ]


def format_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer = None
) -> str:
    """
    Format chat messages into a string prompt.
    
    Tries to use tokenizer's chat template if available,
    otherwise falls back to simple concatenation.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Optional tokenizer with apply_chat_template method
        
    Returns:
        Formatted prompt string
    """
    # Try to use tokenizer's chat template
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")
    
    # Manual formatting fallback (ChatML format for SmolLM2)
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add generation prompt for assistant
    formatted += "<|im_start|>assistant\n"
    
    return formatted


# ============================================================================
# Response Parsing - Robust Version with Multiple Strategies
# ============================================================================
def _extract_first_json_object(s: str) -> str | None:
    """
    Extract the first complete JSON object from a string using balanced-brace scanning.

    Improvements vs previous version:
    - Ignores braces that occur inside JSON strings
    - Handles escaped quotes and escaped backslashes correctly
    - Skips leading whitespace/non-json text robustly
    """
    if not s:
        return None

    # Find the first '{'
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        c = s[i]

        if in_str:
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = False
            continue

        # not in string
        if c == '"':
            in_str = True
            continue

        if c == "{":
            depth += 1
            continue

        if c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
            # If depth goes negative, we had an unmatched '}' (garbage) -> abort
            if depth < 0:
                return None

    # Unterminated object
    return None


def parse_model_response(
    response_text: str,
    chunk_id: int
) -> List[QuestionCandidate]:
    """
    Parse model response with robust error handling.

    Accepts either schema:
      - {"question": "....?"}
      - {"questions": ["....?"]}

    Handles:
    - Extra text before/after JSON
    - Markdown code fences
    - Model returning just an array, or raw question text
    - Basic normalization (ensure question ends with '?')
    """
    original_text = response_text
    cleaned = (response_text or "").strip()

    # Remove markdown code fences (more robust than startswith/endswith)
    # e.g. ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    data: dict | None = None

    # Strategy 1: direct JSON parse (fast path)
    try:
        print("*****************************")
        print(f"Original response text: {original_text}")
        print(f"Cleaned response text: {cleaned}")
        print("*****************************")
        data0 = json.loads(cleaned)
        if isinstance(data0, dict):
            data = data0
    except Exception:
        pass

    # Strategy 2: balanced-brace extraction of first JSON object
    if data is None:
        obj = _extract_first_json_object(cleaned)
        if obj:
            try:
                data0 = json.loads(obj)
                if isinstance(data0, dict):
                    data = data0
            except Exception:
                pass

    # Strategy 3: legacy regex extraction (kept as a backstop)
    if data is None:
        # Try to find {"question": "..."} or {"questions": [...]}
        json_match = re.search(
            r'\{[\s\S]*?"(?:question|questions)"[\s\S]*?\}',
            cleaned,
            flags=re.IGNORECASE
        )
        if json_match:
            try:
                data0 = json.loads(json_match.group(0))
                if isinstance(data0, dict):
                    data = data0
            except Exception:
                pass

    # Strategy 4: array-only output -> wrap
    if data is None:
        array_match = re.search(r'\[[\s\S]*?\]', cleaned)
        if array_match:
            try:
                arr = json.loads(array_match.group(0))
                if isinstance(arr, list):
                    data = {"questions": arr}
            except Exception:
                pass

    # Strategy 5: question-like text -> wrap
    if data is None:
        question_patterns = [
            r'Question:\s*(.+?)(?:\n|$)',
            r'Q:\s*(.+?)(?:\n|$)',
            r'(?:^|\n)([A-Z][^.!?]*\?)',  # sentence ending with ?
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, cleaned, re.MULTILINE | re.IGNORECASE)
            if match:
                q = (match.group(1) or "").strip()
                if q:
                    data = {"question": q}
                    break

    if data is None:
        raise ModelInferenceError(
            message=f"Failed to parse model response as JSON.\nOriginal: {original_text[:500]}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )

    if not isinstance(data, dict):
        raise ModelInferenceError(
            message=f"Model response is not a JSON object: {type(data)}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )

    # Normalize supported schemas into a questions list
    questions: list = []

    if "questions" in data:
        qv = data["questions"]
        if isinstance(qv, list):
            questions = qv
        elif isinstance(qv, str):
            questions = [qv]
        else:
            raise ModelInferenceError(
                message=f"'questions' field is not a list or string: {type(qv)}",
                cause=InferenceErrorCause.PARSE_ERROR,
                chunk_id=chunk_id
            )

    elif "question" in data:
        qv = data["question"]
        if isinstance(qv, str):
            questions = [qv]
        elif isinstance(qv, list):
            questions = qv
        else:
            raise ModelInferenceError(
                message=f"'question' field is not a string or list: {type(qv)}",
                cause=InferenceErrorCause.PARSE_ERROR,
                chunk_id=chunk_id
            )
    else:
        raise ModelInferenceError(
            message=f"Model response missing 'question'/'questions'. Keys: {list(data.keys())}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )

    # Build candidates
    candidates: List[QuestionCandidate] = []
    for i, q in enumerate(questions):
        try:
            if isinstance(q, str):
                question_text = q.strip()
            elif isinstance(q, dict):
                # tolerate dict items in list
                question_text = (
                    q.get("question") or
                    q.get("text") or
                    q.get("query") or
                    ""
                ).strip()
            else:
                continue

            if not question_text:
                continue

            # Normalize: ensure it ends with '?'
            if not question_text.endswith("?"):
                # Avoid "??"
                question_text = question_text.rstrip() + "?"

            candidates.append(
                QuestionCandidate(
                    question=question_text,
                    type=QuestionType.PROCEDURAL,
                    evidence="",
                    chunk_id=chunk_id,
                    style=QuestionStyle.INTERROGATIVE,
                )
            )
        except Exception as e:
            print(f"Warning: Skipped malformed question {i}: {e}")
            continue

    if not candidates:
        raise ModelInferenceError(
            message=f"No valid questions parsed from response.\nResponse: {original_text[:500]}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )

    # Enforce "always 1" question contract at adapter boundary
    # (keeps downstream deterministic even if model returns multiple)
    return candidates[:1]


# ============================================================================
# Mock Adapter (for testing)
# ============================================================================

class MockAdapter(ModelAdapter):
    """
    Mock adapter for testing without actual model inference.
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
        """Generate mock question"""
        import asyncio
        
        await asyncio.sleep(self.latency_ms / 1000.0)
        
        templates = [
            "How do I configure this feature?",
            "What is the purpose of this functionality?",
            "When should I use this approach?",
            "Which option is best for my use case?",
        ]
        
        question = templates[chunk.chunk_id % len(templates)]
        
        return [QuestionCandidate(
            question=question,
            type=QuestionType.PROCEDURAL,
            evidence="",
            chunk_id=chunk.chunk_id,
            style=QuestionStyle.INTERROGATIVE
        )]
    
    async def is_ready(self) -> bool:
        return self._ready
    
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        return 100 * 1024 * 1024


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'ModelAdapter',
    'MockAdapter',
    'build_prompt',
    'format_chat_prompt',
    'parse_model_response',
]