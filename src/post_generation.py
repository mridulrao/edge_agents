"""
Edge Question Generation Pipeline - Post-Generation (Simplified)
Basic validation and selection logic
"""

from typing import List
from src.pipeline_types import (
    QuestionCandidate,
    ValidationResult,
    RejectedCandidate,
    FinalQuestion,
    Chunk,
)


# ============================================================================
# Simplified Validation - Just count check
# ============================================================================

def validate_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk],
    allowed_styles=None  # Keep signature compatible
) -> ValidationResult:
    """
    Simplified validation - just return all candidates as valid.
    
    Args:
        candidates: Raw question candidates
        chunks: Source chunks (unused in simplified version)
        allowed_styles: Unused, kept for API compatibility
        
    Returns:
        ValidationResult with all candidates marked as valid
    """
    # All candidates are valid in simplified mode
    valid = candidates
    rejected = []
    
    return ValidationResult(valid=valid, rejected=rejected)


# ============================================================================
# No-op Deduplication
# ============================================================================

def deduplicate_candidates(
    candidates: List[QuestionCandidate],
    similarity_threshold: float = 0.7  # Unused, kept for API compatibility
) -> List[QuestionCandidate]:
    """
    Simplified deduplication - just return all candidates.
    
    Args:
        candidates: Valid question candidates
        similarity_threshold: Unused
        
    Returns:
        Same list of candidates (no deduplication)
    """
    return candidates


# ============================================================================
# Simplified Ranking - Keep original order
# ============================================================================

def rank_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk]
) -> List['RankedCandidate']:
    """
    Simplified ranking - just wrap candidates with neutral scores.
    
    Args:
        candidates: Valid, deduplicated candidates
        chunks: Source chunks (unused)
        
    Returns:
        Candidates wrapped as RankedCandidate with score=1.0
    """
    from src.pipeline_types import RankedCandidate
    
    ranked = []
    for candidate in candidates:
        ranked.append(RankedCandidate(
            candidate=candidate,
            score=1.0,  # Neutral score
            score_breakdown={'composite': 1.0}
        ))
    
    return ranked


# ============================================================================
# Simplified Selection - Take first K
# ============================================================================

def select_final_questions(
    ranked: List['RankedCandidate'],
    k: int,
    diversity_threshold: float = 0.6  # Unused
) -> List[FinalQuestion]:
    """
    Simplified selection - take first K questions.
    
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