"""
Edge Question Generation Pipeline - Post-Generation (Simplified)
Minimal validation, no evidence checking
"""

from typing import List
from src.pipeline_types import (
    QuestionCandidate,
    ValidationResult,
    RejectedCandidate,
    RankedCandidate,
    FinalQuestion,
    Chunk,
)


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
    
    Checks:
    - Question exists and not empty
    - Evidence exists (but NOT validated against chunk)
    
    Args:
        candidates: Raw question candidates
        chunks: Source chunks (unused - we don't validate evidence content)
        allowed_styles: Unused, kept for API compatibility
        
    Returns:
        ValidationResult with valid and rejected candidates
    """
    valid: List[QuestionCandidate] = []
    rejected: List[RejectedCandidate] = []
    
    for candidate in candidates:
        # Check 1: Question exists and not empty
        if not candidate.question or len(candidate.question.strip()) < 10:
            rejected.append(RejectedCandidate(
                candidate=candidate,
                reason="Question too short or empty"
            ))
            continue
        
        # Passed all checks
        valid.append(candidate)
    
    return ValidationResult(valid=valid, rejected=rejected)


# ============================================================================
# No Deduplication
# ============================================================================

def deduplicate_candidates(
    candidates: List[QuestionCandidate],
    similarity_threshold: float = 0.7  # Unused, kept for API compatibility
) -> List[QuestionCandidate]:
    """
    No deduplication - return all candidates as-is.
    
    Args:
        candidates: Valid question candidates
        similarity_threshold: Unused
        
    Returns:
        Same list of candidates (no deduplication)
    """
    return candidates


# ============================================================================
# Neutral Ranking - Keep Original Order
# ============================================================================

def rank_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk]
) -> List[RankedCandidate]:
    """
    Neutral ranking - wrap candidates with score=1.0, keep original order.
    
    Args:
        candidates: Valid, deduplicated candidates
        chunks: Source chunks (unused)
        
    Returns:
        Candidates wrapped as RankedCandidate with neutral scores
    """
    ranked = []
    for candidate in candidates:
        ranked.append(RankedCandidate(
            candidate=candidate,
            score=1.0,  # Neutral score
            score_breakdown={'composite': 1.0}
        ))
    
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
    Simple selection - take first K questions.
    
    Args:
        ranked: Ranked candidates
        k: Desired number of questions
        diversity_threshold: Unused
        
    Returns:
        First K questions (or all if fewer than K)
    """
    # Take first K (or all if we have fewer)
    selected = ranked[:k]
    
    # Convert to FinalQuestion
    return [
        FinalQuestion(
            question=rc.candidate.question,
            type=rc.candidate.type.value,
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