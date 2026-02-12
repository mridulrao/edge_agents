"""
Edge Question Generation Pipeline - Model Adapter Interface
Optimized for instruction-tuned models (SmolLM2, etc.)

UPDATED:
- Adds support for 4 configurations (A/B/C/D) by introducing a Mode object.
- Keeps backward compatibility: existing adapters can still implement generate_questions(...)
  and the pipeline can migrate to generate_questions_mode(...) step-by-step.
- Extends parsing to handle:
  - JSON single: {"question":"..."}
  - JSON multi:  {"questions":[{"question":"..."}, ...]} OR {"questions":["...?", ...]}
  - PLAINTEXT single: "....?"
  - PLAINTEXT multi: numbered list "1. ...?\n2. ...?"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import json
import re

from local_llm.pipeline_types import Chunk, QuestionCandidate, GenerationConfig, QuestionType, QuestionStyle
from local_llm.errors import ModelInferenceError, InferenceErrorCause


# ============================================================================
# Mode / Enums for 4 configurations
# ============================================================================

class GenerationScope(str, Enum):
    CHUNKED = "chunked"          # per chunk calls
    FULL_ARTICLE = "full_article"  # single call with full article


class OutputFormat(str, Enum):
    JSON = "json"               # model returns JSON
    PLAINTEXT = "plaintext"     # model returns plain text


@dataclass(frozen=True)
class QuestionGenMode:
    """
    Describes how the model should be prompted + how output should be parsed.
    """
    scope: GenerationScope
    output_format: OutputFormat
    questions_per_call: int  # 1 or N

    @staticmethod
    def A_chunked_json_1() -> "QuestionGenMode":
        return QuestionGenMode(GenerationScope.CHUNKED, OutputFormat.JSON, 1)

    @staticmethod
    def B_full_json_n(n: int) -> "QuestionGenMode":
        return QuestionGenMode(GenerationScope.FULL_ARTICLE, OutputFormat.JSON, n)

    @staticmethod
    def C_chunked_text_1() -> "QuestionGenMode":
        return QuestionGenMode(GenerationScope.CHUNKED, OutputFormat.PLAINTEXT, 1)

    @staticmethod
    def D_full_text_n(n: int) -> "QuestionGenMode":
        return QuestionGenMode(GenerationScope.FULL_ARTICLE, OutputFormat.PLAINTEXT, n)


# ============================================================================
# Model Adapter Interface
# ============================================================================

class ModelAdapter(ABC):
    """
    Abstract interface for model inference.

    Existing contract: generate_questions(chunk, num_candidates=1, ...) -> [QuestionCandidate]
    New contract: generate_questions_mode(text, mode, ...) -> [QuestionCandidate]
    """

    # ---------------- Existing contract (kept) ----------------
    @abstractmethod
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        raise NotImplementedError

    # ---------------- New contract (recommended for 4 modes) ----------------
    async def generate_questions_mode(
        self,
        text: str,
        chunk_id: int,
        mode: QuestionGenMode,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        # Best-effort compatibility for 1-question-per-call chunked modes
        if mode.scope == GenerationScope.CHUNKED and mode.questions_per_call == 1:
            temp_chunk = Chunk(chunk_id=chunk_id, text=text)
            return await self.generate_questions(temp_chunk, 1, generation_config)

        raise NotImplementedError(
            "Adapter must implement generate_questions_mode() for full-article or N-per-call modes."
        )

    @abstractmethod
    async def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        raise NotImplementedError


# ============================================================================
# Prompt Templates (4 modes)
# ============================================================================

def build_prompt_for_mode(
    text: str,
    desired_questions: int,
    mode: QuestionGenMode,
    config: GenerationConfig | None = None
) -> List[Dict[str, str]]:
    rules = (
        "Rules:\n"
        "- Do NOT mention “the text”, “this passage”, or “given context”.\n"
        "- Do NOT ask about writing style/structure/wording.\n"
        "- Ask natural questions a human would ask.\n"
        "- Avoid duplicates.\n"
    )

    if mode.output_format == OutputFormat.JSON:
        if mode.questions_per_call == 1:
            system = (
                "You are a strict JSON generator.\n"
                "Output ONLY valid JSON. No extra text.\n"
                'Schema: {"question":"...?"}\n'
            )
            user = (
                "Write ONE natural question a human would ask after reading the content below.\n\n"
                f"{rules}\n"
                "Output ONLY JSON matching the schema.\n\n"
                f"Content:\n{text}\n"
            )
        else:
            system = (
                "You are a strict JSON generator.\n"
                "Output ONLY valid JSON. No extra text.\n"
                'Schema: {"questions":[{"question":"...?"}, ...]}\n'
            )
            user = (
                f"Generate EXACTLY {desired_questions} distinct, natural questions from the content below.\n\n"
                f"{rules}\n"
                "Output ONLY JSON matching the schema.\n\n"
                f"Content:\n{text}\n"
            )
    else:
        if mode.questions_per_call == 1:
            system = (
                "You generate questions.\n"
                "Output ONLY the question text. No quotes, no numbering, no JSON."
            )
            user = (
                "Write ONE natural question a human would ask after reading the content below.\n\n"
                f"{rules}\n"
                "Output ONE line ending with a question mark.\n\n"
                f"Content:\n{text}\n"
            )
        else:
            system = (
                "You generate questions.\n"
                "Output ONLY questions as a numbered list.\n"
                "No JSON. No extra text."
            )
            user = (
                f"Generate EXACTLY {desired_questions} distinct, natural questions from the content below.\n\n"
                f"{rules}\n"
                "Output as a numbered list, one question per line, like:\n"
                "1. ...?\n2. ...?\n\n"
                f"Content:\n{text}\n"
            )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(chunk_text: str, num_candidates: int = 1, config: GenerationConfig = None):
    return build_prompt_for_mode(
        text=chunk_text,
        desired_questions=1,
        mode=QuestionGenMode.A_chunked_json_1(),
        config=config,
    )


# ============================================================================
# Chat formatting utility (unchanged)
# ============================================================================

def format_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer=None
) -> str:
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")

    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


# ============================================================================
# Response Parsing - Extended for JSON + PLAINTEXT
# ============================================================================

def _extract_first_json_object(s: str) -> str | None:
    if not s:
        return None
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

        if c == '"':
            in_str = True
            continue

        if c == "{":
            depth += 1
            continue

        if c == "}":
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
            if depth < 0:
                return None
    return None


def _repair_json_like_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return t

    t = t.replace("“", '"').replace("”", '"').replace("’", "'")

    # ✅ FIX: JSON does NOT support \' escape. Replace it.
    t = t.replace("\\'", "'")

    t = re.sub(r'\\+"(\s*})$', r'"\1', t)      # \\"} -> "}
    t = re.sub(r'""(\s*})$', r'"\1', t)        # ""}  -> "}
    t = re.sub(r'"\s*"\s*}', r'"}', t)
    t = re.sub(r'\\("(\s*})$)', r'\1', t)
    return t


def _strip_markdown_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _sanitize_question_text(q: str) -> str | None:
    q = (q or "").strip()
    if not q:
        return None
    q = re.sub(r'^\s*[\-\*\d\.\)\:]+\s*', "", q).strip()
    q = q.strip('"\''"”“‘’ ").strip()
    if "?" in q:
        q = q[: q.rfind("?") + 1].strip()
        return q if len(q) >= 4 else None
    return None


def _split_numbered_questions(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: List[str] = []
    buff: List[str] = []

    def flush():
        if buff:
            out.append(" ".join(buff).strip())
            buff.clear()

    for ln in lines:
        if re.match(r"^\s*(\d+[\.\)]|[-*])\s+", ln):
            flush()
            ln2 = re.sub(r"^\s*(\d+[\.\)]|[-*])\s+", "", ln).strip()
            buff.append(ln2)
        else:
            buff.append(ln)
    flush()
    return out


def parse_model_response_for_mode(
    response_text: str,
    chunk_id: int,
    mode: QuestionGenMode,
    limit: Optional[int] = None,
) -> List[QuestionCandidate]:
    original_text = response_text or ""
    cleaned = _strip_markdown_fences(original_text)
    cleaned = _repair_json_like_text(cleaned)

    questions_out: List[str] = []

    if mode.output_format == OutputFormat.JSON:
        data: dict | None = None

        # 1) direct parse
        try:
            x = json.loads(cleaned)
            if isinstance(x, dict):
                data = x
        except Exception:
            pass

        # 2) balanced extraction
        if data is None:
            obj = _extract_first_json_object(cleaned)
            if obj:
                obj = _repair_json_like_text(obj)
                try:
                    x = json.loads(obj)
                    if isinstance(x, dict):
                        data = x
                except Exception:
                    pass

        # 3) regex extraction
        if data is None:
            m = re.search(r'\{[\s\S]*?"(?:question|questions|output)"[\s\S]*?\}', cleaned, flags=re.IGNORECASE)
            if m:
                candidate = _repair_json_like_text(m.group(0))
                try:
                    x = json.loads(candidate)
                    if isinstance(x, dict):
                        data = x
                except Exception:
                    pass

        # 4) array-only output
        if data is None:
            m = re.search(r'\[[\s\S]*?\]', cleaned)
            if m:
                try:
                    arr = json.loads(m.group(0))
                    if isinstance(arr, list):
                        data = {"questions": arr}
                except Exception:
                    pass

        if data is None or not isinstance(data, dict):
            raise ModelInferenceError(
                message=f"Failed to parse model response as JSON.\nOriginal: {original_text[:500]}",
                cause=InferenceErrorCause.PARSE_ERROR,
                chunk_id=chunk_id
            )

        # normalize
        if "questions" in data:
            qv = data["questions"]
            if isinstance(qv, list):
                for it in qv:
                    if isinstance(it, str):
                        questions_out.append(it)
                    elif isinstance(it, dict):
                        questions_out.append(it.get("question") or it.get("text") or it.get("query") or it.get("output") or "")
            elif isinstance(qv, str):
                questions_out.append(qv)
            else:
                raise ModelInferenceError(
                    message=f"'questions' field not list/string: {type(qv)}",
                    cause=InferenceErrorCause.PARSE_ERROR,
                    chunk_id=chunk_id
                )

        elif "question" in data:
            qv = data["question"]
            if isinstance(qv, str):
                questions_out.append(qv)
            elif isinstance(qv, list):
                for it in qv:
                    if isinstance(it, str):
                        questions_out.append(it)
                    elif isinstance(it, dict):
                        questions_out.append(it.get("question") or it.get("text") or it.get("query") or it.get("output") or "")
            else:
                raise ModelInferenceError(
                    message=f"'question' field not string/list: {type(qv)}",
                    cause=InferenceErrorCause.PARSE_ERROR,
                    chunk_id=chunk_id
                )

        # ✅ NEW: accept {"output": "..."} style
        elif "output" in data:
            qv = data["output"]
            if isinstance(qv, str):
                questions_out.append(qv)
            else:
                raise ModelInferenceError(
                    message=f"'output' field not string: {type(qv)}",
                    cause=InferenceErrorCause.PARSE_ERROR,
                    chunk_id=chunk_id
                )
        else:
            raise ModelInferenceError(
                message=f"Missing 'question'/'questions'/'output'. Keys: {list(data.keys())}",
                cause=InferenceErrorCause.PARSE_ERROR,
                chunk_id=chunk_id
            )

    else:
        # PLAINTEXT
        if mode.questions_per_call == 1:
            questions_out = [cleaned]
        else:
            questions_out = _split_numbered_questions(cleaned)

    # sanitize and build candidates
    candidates: List[QuestionCandidate] = []
    for q in questions_out:
        q2 = _sanitize_question_text(q)
        if not q2:
            continue

        # ✅ FIX: QuestionCandidate in pipeline_types has NO 'evidence' field
        candidates.append(
            QuestionCandidate(
                question=q2,
                type=QuestionType.PROCEDURAL,
                chunk_id=chunk_id,
                style=QuestionStyle.INTERROGATIVE,
            )
        )

    if not candidates:
        raise ModelInferenceError(
            message=f"No valid questions parsed from response.\nResponse: {original_text[:500]}",
            cause=InferenceErrorCause.PARSE_ERROR,
            chunk_id=chunk_id
        )

    if limit is not None:
        return candidates[:limit]
    return candidates


def parse_model_response(response_text: str, chunk_id: int) -> List[QuestionCandidate]:
    return parse_model_response_for_mode(
        response_text=response_text,
        chunk_id=chunk_id,
        mode=QuestionGenMode.A_chunked_json_1(),
        limit=1,
    )


__all__ = [
    'ModelAdapter',
    'GenerationScope',
    'OutputFormat',
    'QuestionGenMode',
    'build_prompt_for_mode',
    'build_prompt',
    'format_chat_prompt',
    'parse_model_response_for_mode',
    'parse_model_response',
]
