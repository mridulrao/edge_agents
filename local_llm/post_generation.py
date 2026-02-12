"""
Edge Question Generation Pipeline - Post-Generation (Simplified)
Minimal validation, no evidence checking

UPDATED (for 4 configurations A/B/C/D):
✅ Works with candidates coming from:
   - 1-per-call outputs (A/C)
   - multi-question outputs from a single call (B/D)
✅ Adds optional lightweight dedup (default ON, but conservative)
✅ Adds optional style/meta filtering (still minimal)
✅ Keeps signatures compatible with your pipeline
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple
import re
import hashlib

from local_llm.pipeline_types import (
    QuestionCandidate,
    ValidationResult,
    RejectedCandidate,
    RankedCandidate,
    FinalQuestion,
    Chunk,
)


# ============================================================================
# Helpers
# ============================================================================

_META_PATTERNS = re.compile(
    r"\b("
    r"this\s+(text|passage|article|context|paragraph)|"
    r"the\s+(text|passage|article|context|paragraph)|"
    r"given\s+(text|passage|context)|"
    r"writing\s+(style|structure|tone)|"
    r"purpose\s+of\s+the\s+(text|passage|article)"
    r")\b",
    re.IGNORECASE,
)


def _normalize_for_dedup(q: str) -> str:
    q = (q or "").lower().strip()
    q = q.replace("\u00a0", " ")
    q = re.sub(r"[^a-z0-9\s\?]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # strip leading numbering/bullets
    q = re.sub(r"^\s*[\-\*\d\.\)\:]+\s*", "", q).strip()
    return q


def _hash_norm(q: str) -> str:
    return hashlib.md5(_normalize_for_dedup(q).encode("utf-8")).hexdigest()


def _is_question_like(q: str) -> bool:
    q = (q or "").strip()
    return ("?" in q) and (len(q) >= 6)


# ============================================================================
# Minimal Validation - No Evidence Checking
# ============================================================================

def validate_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk],
    allowed_styles=None  # Keep signature compatible
) -> ValidationResult:
    """
    Minimal validation - only basic checks, no evidence validation.

    Updated checks (still minimal):
    - Non-empty + minimum length (>=10 chars by default)
    - Contains '?' (prevents non-question lines in plaintext modes)
    - Rejects meta questions about "the text/passage/context" (common small-model failure)
    """
    valid: List[QuestionCandidate] = []
    rejected: List[RejectedCandidate] = []

    for candidate in candidates:
        q = (candidate.question or "").strip()

        if len(q) < 10:
            rejected.append(RejectedCandidate(candidate=candidate, reason="Question too short or empty"))
            continue

        if not _is_question_like(q):
            rejected.append(RejectedCandidate(candidate=candidate, reason="Not a question (missing '?')"))
            continue

        if _META_PATTERNS.search(q):
            rejected.append(RejectedCandidate(candidate=candidate, reason="Meta question about the text/context"))
            continue

        # Passed
        valid.append(candidate)

    return ValidationResult(valid=valid, rejected=rejected)


# ============================================================================
# Lightweight Deduplication (Conservative)
# ============================================================================

def deduplicate_candidates(
    candidates: List[QuestionCandidate],
    similarity_threshold: float = 0.7  # Unused, kept for API compatibility
) -> List[QuestionCandidate]:
    """
    Conservative dedup:
    - exact match after normalization hash

    Why this matters for B/D:
    multi-question outputs often contain near-identical questions.
    """
    seen: Set[str] = set()
    out: List[QuestionCandidate] = []

    for c in candidates:
        h = _hash_norm(c.question)
        if h in seen:
            continue
        seen.add(h)
        out.append(c)

    return out


# ============================================================================
# Neutral Ranking - Keep Original Order (but allow simple preference)
# ============================================================================

def rank_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk]
) -> List[RankedCandidate]:
    """
    Mostly neutral ranking to preserve your behavior,
    but adds a tiny preference for "clean" questions:
    - medium length (not too short, not too long)
    - fewer repeated tokens

    If you want fully neutral, set score=1.0 for all.
    """
    def score(q: str) -> float:
        t = _normalize_for_dedup(q)
        words = [w for w in t.split() if w not in {"the", "a", "an", "and", "or", "to", "of", "in"}]
        if not words:
            return 1.0
        uniq = len(set(words)) / max(1, len(words))
        L = len(q)
        # prefer ~60-120 chars
        length_pen = abs(L - 90) / 90.0
        return max(0.1, (uniq * 1.5) - length_pen + 1.0)

    ranked: List[RankedCandidate] = []
    for c in candidates:
        s = float(score(c.question))
        ranked.append(RankedCandidate(
            candidate=c,
            score=s,
            score_breakdown={"composite": s}
        ))

    # Sort descending score but keep stable order among ties
    ranked.sort(key=lambda rc: rc.score, reverse=True)
    return ranked


# ============================================================================
# Simple Selection - Take First K
# ============================================================================

def select_final_questions(
    ranked: List[RankedCandidate],
    k: int,
    diversity_threshold: float = 0.6  # Unused
) -> List[FinalQuestion]:
    """
    Select top-K after ranking.

    Updated:
    - Still returns up to K
    - Uses ranked order (post score sort)
    """
    k = max(1, int(k))
    selected = ranked[:k]

    return [
        FinalQuestion(
            question=rc.candidate.question,
            type=getattr(rc.candidate.type, "value", str(rc.candidate.type)),
            source_chunk_id=rc.candidate.chunk_id,
            confidence_score=rc.score
        )
        for rc in selected
    ]


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'validate_candidates',
    'deduplicate_candidates',
    'rank_candidates',
    'select_final_questions',
]
