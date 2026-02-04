"""
Edge Question Generation Pipeline - Post-Generation
Validation, deduplication, ranking, and selection logic
"""

import re
from typing import List, Set, Tuple, Optional
from difflib import SequenceMatcher

from pipeline_types import (
    QuestionCandidate,
    ValidationResult,
    RejectedCandidate,
    RankedCandidate,
    FinalQuestion,
    Chunk,
    QuestionType,
    QuestionStyle
)


# ============================================================================
# Validation
# ============================================================================

# Trivial question patterns to reject
TRIVIAL_PATTERNS = [
    r"^how does this work\?*$",
    r"^how is \w+\?*$",
    r"^what is \w+\?*$",
]


def validate_question_format(
    candidate: QuestionCandidate,
    allowed_styles: Optional[List[QuestionStyle]] = None
) -> Tuple[bool, str]:
    """
    Validate question format and structure based on style.
    
    For INTERROGATIVE style:
    - Must start with question word (How/What/When/Which/Why/Where/Who)
    - Must end with "?"
    - Length: 20-150 characters
    
    For IMPERATIVE style:
    - Starts with action verb (Explain, Describe, List, Identify, etc.)
    - Length: 15-150 characters
    
    For CONTEXTUAL style:
    - No strict format requirements
    - Must be meaningful (3+ words)
    - Length: 15-150 characters
    
    Args:
        candidate: Question candidate to validate
        allowed_styles: List of allowed styles (None = allow all)
    
    Returns:
        (is_valid, rejection_reason)
    """
    question = candidate.question.strip()
    style = candidate.style
    
    # Check if style is allowed
    if allowed_styles and style not in allowed_styles:
        return False, f"Question style '{style.value}' not allowed"
    
    # Style-specific validation
    if style == QuestionStyle.INTERROGATIVE:
        # Must start with question word
        question_words = ['how', 'what', 'when', 'which', 'why', 'where', 'who', 'whose', 'whom']
        starts_with_question = any(question.lower().startswith(word) for word in question_words)
        
        if not starts_with_question:
            return False, f"Interrogative question must start with: {', '.join(question_words)}"
        
        # Must end with "?"
        if not question.endswith("?"):
            return False, "Interrogative question must end with '?'"
        
        # Check length
        if len(question) < 20:
            return False, f"Question too short ({len(question)} chars, min 20)"
        
    elif style == QuestionStyle.IMPERATIVE:
        # Must start with action verb
        action_verbs = [
            'explain', 'describe', 'list', 'identify', 'compare', 'contrast',
            'summarize', 'outline', 'detail', 'illustrate', 'demonstrate',
            'clarify', 'define', 'analyze', 'discuss', 'show'
        ]
        starts_with_action = any(question.lower().startswith(verb) for verb in action_verbs)
        
        if not starts_with_action:
            return False, f"Imperative must start with action verb like: {', '.join(action_verbs[:5])}"
        
        # Check length
        if len(question) < 15:
            return False, f"Question too short ({len(question)} chars, min 15)"
    
    elif style == QuestionStyle.CONTEXTUAL:
        # Minimal requirements for contextual
        words = question.split()
        if len(words) < 3:
            return False, f"Contextual question too short ({len(words)} words, min 3)"
        
        # Check length
        if len(question) < 15:
            return False, f"Question too short ({len(question)} chars, min 15)"
    
    # Common validation for all styles
    if len(question) > 150:
        return False, f"Question too long ({len(question)} chars, max 150)"
    
    # Check for trivial patterns (relaxed for non-interrogative)
    if style == QuestionStyle.INTERROGATIVE:
        normalized = question.lower().strip()
        for pattern in TRIVIAL_PATTERNS:
            if re.match(pattern, normalized):
                return False, f"Matches trivial pattern: {pattern}"
    
    # Check meaningful word count
    words = question.lower().split()
    stop_words = {'how', 'do', 'i', 'the', 'a', 'an', 'is', 'are', 'in', 'to', 'for', 'of', 'and'}
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    min_meaningful = 2 if style == QuestionStyle.CONTEXTUAL else 3
    if len(meaningful_words) < min_meaningful:
        return False, f"Too few meaningful words ({len(meaningful_words)}, min {min_meaningful})"
    
    return True, ""


def validate_evidence(
    candidate: QuestionCandidate,
    chunk: Chunk,
    similarity_threshold: float = 0.9
) -> Tuple[bool, str]:
    """
    Validate evidence grounding in source chunk.
    
    Checks:
    - Evidence length > 10 characters
    - Evidence exists in chunk (fuzzy match with threshold)
    
    Returns:
        (is_valid, rejection_reason)
    """
    evidence = candidate.evidence.strip()
    
    # Check length
    if len(evidence) < 10:
        return False, f"Evidence too short ({len(evidence)} chars, min 10)"
    
    # Fuzzy match evidence in chunk
    # Use sliding window to find best match
    chunk_text = chunk.text.lower()
    evidence_lower = evidence.lower()
    
    # Direct substring check first
    if evidence_lower in chunk_text:
        return True, ""
    
    # Fuzzy matching using SequenceMatcher
    best_ratio = 0.0
    window_size = len(evidence)
    
    for i in range(len(chunk_text) - window_size + 1):
        window = chunk_text[i:i + window_size]
        ratio = SequenceMatcher(None, evidence_lower, window).ratio()
        best_ratio = max(best_ratio, ratio)
        
        if best_ratio >= similarity_threshold:
            return True, ""
    
    return False, f"Evidence not found in chunk (best match: {best_ratio:.2f}, threshold: {similarity_threshold})"


def validate_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk],
    allowed_styles: Optional[List[QuestionStyle]] = None
) -> ValidationResult:
    """
    Validate candidates against KB-specific quality rules.
    
    Sequential filters:
    1. Required fields present (already validated during parsing)
    2. Question format (based on style: interrogative, imperative, contextual)
    3. Evidence validation (length, grounding in chunk)
    4. Trivial pattern rejection (for interrogative style)
    
    Args:
        candidates: Raw question candidates
        chunks: Source chunks for evidence validation
        allowed_styles: List of allowed question styles (None = allow all)
        
    Returns:
        ValidationResult with valid and rejected candidates
    """
    valid: List[QuestionCandidate] = []
    rejected: List[RejectedCandidate] = []
    
    # Create chunk lookup
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    
    for candidate in candidates:
        # Validate question format
        is_valid, reason = validate_question_format(candidate, allowed_styles)
        if not is_valid:
            rejected.append(RejectedCandidate(
                candidate=candidate,
                reason=reason
            ))
            continue
        
        # Validate evidence
        chunk = chunk_map.get(candidate.chunk_id)
        if chunk is None:
            rejected.append(RejectedCandidate(
                candidate=candidate,
                reason=f"Chunk {candidate.chunk_id} not found"
            ))
            continue
        
        is_valid, reason = validate_evidence(candidate, chunk)
        if not is_valid:
            rejected.append(RejectedCandidate(
                candidate=candidate,
                reason=reason
            ))
            continue
        
        # Passed all validations
        valid.append(candidate)
    
    return ValidationResult(valid=valid, rejected=rejected)


# ============================================================================
# Deduplication
# ============================================================================

def normalize_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison"""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute character n-gram similarity between two texts.
    
    Returns similarity score in [0, 1]
    """
    # Normalize texts
    t1 = normalize_for_comparison(text1)
    t2 = normalize_for_comparison(text2)
    
    if not t1 or not t2:
        return 0.0
    
    # Generate n-grams
    def get_ngrams(text: str, n: int) -> Set[str]:
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(t1, n)
    ngrams2 = get_ngrams(t2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    # Jaccard similarity
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union)


def deduplicate_candidates(
    candidates: List[QuestionCandidate],
    similarity_threshold: float = 0.7
) -> List[QuestionCandidate]:
    """
    Remove near-duplicate questions using n-gram similarity.
    
    Strategy:
    - Compute pairwise similarity (character n-gram overlap, n=3)
    - Merge questions with >70% similarity, retaining most specific version
    - Penalize questions from overlapping chunk regions
    
    Args:
        candidates: Valid question candidates
        similarity_threshold: Minimum similarity to consider duplicates
        
    Returns:
        Deduplicated list of candidates
    """
    if len(candidates) <= 1:
        return candidates
    
    # Track which candidates to keep
    keep_indices: Set[int] = set(range(len(candidates)))
    
    # Compare all pairs
    for i in range(len(candidates)):
        if i not in keep_indices:
            continue
        
        for j in range(i + 1, len(candidates)):
            if j not in keep_indices:
                continue
            
            # Compute similarity
            similarity = compute_ngram_similarity(
                candidates[i].question,
                candidates[j].question
            )
            
            if similarity >= similarity_threshold:
                # These are duplicates - decide which to keep
                # Prefer more specific questions (longer)
                # Prefer questions from different chunks
                
                c1, c2 = candidates[i], candidates[j]
                
                # Length-based specificity
                len1, len2 = len(c1.question), len(c2.question)
                
                # Chunk diversity bonus
                same_chunk = (c1.chunk_id == c2.chunk_id)
                
                # Keep the longer question, unless they're from same chunk
                if len1 >= len2 and not same_chunk:
                    keep_indices.discard(j)
                else:
                    keep_indices.discard(i)
                    break  # Move to next i
    
    return [candidates[i] for i in sorted(keep_indices)]


# ============================================================================
# Ranking
# ============================================================================

def score_evidence_specificity(evidence: str) -> float:
    """
    Score evidence based on specificity indicators.
    
    High scores for:
    - Numbers and quantities
    - UI elements (buttons, menus, etc.)
    - Technical terms
    - Specific action verbs
    
    Returns score in [0, 1]
    """
    score = 0.0
    
    # Check for numbers
    if re.search(r'\d+', evidence):
        score += 0.3
    
    # Check for UI elements
    ui_terms = ['button', 'menu', 'click', 'select', 'navigate', 'tab', 'field', 'dropdown']
    evidence_lower = evidence.lower()
    for term in ui_terms:
        if term in evidence_lower:
            score += 0.2
            break
    
    # Check for technical terms (capitalized words, acronyms)
    technical_terms = re.findall(r'\b[A-Z]{2,}\b', evidence)  # Acronyms
    score += min(0.3, len(technical_terms) * 0.1)
    
    # Check for specific action verbs
    action_verbs = ['configure', 'enable', 'disable', 'set up', 'create', 'delete', 'modify']
    for verb in action_verbs:
        if verb in evidence_lower:
            score += 0.2
            break
    
    return min(1.0, score)


def score_question_specificity(question: str) -> float:
    """
    Score question based on specificity.
    
    Factors:
    - Length (longer = more specific)
    - Presence of domain terms
    - Verb specificity
    
    Returns score in [0, 1]
    """
    score = 0.0
    
    # Length-based score (20-150 char range)
    length = len(question)
    if length >= 60:
        score += 0.4
    elif length >= 40:
        score += 0.2
    
    # Domain term presence
    domain_terms = [
        'campaign', 'conversion', 'tracking', 'audience', 'targeting',
        'bidding', 'creative', 'placement', 'budget', 'reporting'
    ]
    question_lower = question.lower()
    domain_count = sum(1 for term in domain_terms if term in question_lower)
    score += min(0.3, domain_count * 0.1)
    
    # Specific verbs
    specific_verbs = [
        'configure', 'troubleshoot', 'optimize', 'integrate',
        'customize', 'implement', 'analyze'
    ]
    for verb in specific_verbs:
        if verb in question_lower:
            score += 0.3
            break
    
    return min(1.0, score)


def rank_candidates(
    candidates: List[QuestionCandidate],
    chunks: List[Chunk]
) -> List[RankedCandidate]:
    """
    Rank candidates by composite score.
    
    Weighted scoring:
    - Evidence specificity (40%)
    - Question specificity (30%)
    - Chunk diversity (20%)
    - Type distribution (10%)
    
    Args:
        candidates: Valid, deduplicated candidates
        chunks: Source chunks for context
        
    Returns:
        Ranked candidates in descending score order
    """
    if not candidates:
        return []
    
    # Count candidates per chunk for diversity scoring
    chunk_counts = {}
    for c in candidates:
        chunk_counts[c.chunk_id] = chunk_counts.get(c.chunk_id, 0) + 1
    
    # Count question types for distribution scoring
    type_counts = {}
    for c in candidates:
        type_counts[c.type] = type_counts.get(c.type, 0) + 1
    
    # Score each candidate
    ranked = []
    for candidate in candidates:
        # Evidence specificity (40%)
        evidence_score = score_evidence_specificity(candidate.evidence)
        
        # Question specificity (30%)
        question_score = score_question_specificity(candidate.question)
        
        # Chunk diversity (20%) - prefer chunks with fewer questions
        chunk_diversity = 1.0 / chunk_counts[candidate.chunk_id]
        
        # Type distribution (10%) - slight boost for procedural/troubleshooting
        type_boost = 1.0
        if candidate.type in [QuestionType.PROCEDURAL, QuestionType.TROUBLESHOOTING]:
            type_boost = 1.2
        elif candidate.type == QuestionType.CONCEPTUAL:
            type_boost = 0.8
        
        # Composite score
        composite_score = (
            0.4 * evidence_score +
            0.3 * question_score +
            0.2 * chunk_diversity +
            0.1 * type_boost
        )
        
        ranked.append(RankedCandidate(
            candidate=candidate,
            score=composite_score,
            score_breakdown={
                'evidence_specificity': evidence_score,
                'question_specificity': question_score,
                'chunk_diversity': chunk_diversity,
                'type_boost': type_boost,
                'composite': composite_score
            }
        ))
    
    # Sort by score (descending via __lt__ implementation)
    ranked.sort()
    
    return ranked


# ============================================================================
# Selection
# ============================================================================

def select_final_questions(
    ranked: List[RankedCandidate],
    k: int,
    diversity_threshold: float = 0.6
) -> List[FinalQuestion]:
    """
    Select top K questions with diversity constraints.
    
    Strategy:
    1. Rank all valid candidates by composite score
    2. Select top K ensuring â‰¥60% from different chunks
    3. If <K remain, relax diversity constraints
    4. Final fallback: return all valid questions (min 1, target 3-4)
    
    Args:
        ranked: Ranked candidates in descending score order
        k: Desired number of questions
        diversity_threshold: Minimum fraction from different chunks
        
    Returns:
        Final selected questions
    """
    if not ranked:
        return []
    
    if len(ranked) <= k:
        # Return all candidates
        return [
            FinalQuestion(
                question=rc.candidate.question,
                type=rc.candidate.type.value,
                source_chunk_id=rc.candidate.chunk_id,
                confidence_score=rc.score
            )
            for rc in ranked
        ]
    
    # Try to select with diversity constraint
    selected = []
    chunk_ids_used = set()
    
    # First pass: select top candidates with diversity
    for rc in ranked:
        if len(selected) >= k:
            break
        
        # Check diversity constraint
        if len(selected) < k * diversity_threshold:
            # Still building diverse set - prefer new chunks
            if rc.candidate.chunk_id not in chunk_ids_used:
                selected.append(rc)
                chunk_ids_used.add(rc.candidate.chunk_id)
        else:
            # Diversity threshold met - take best remaining
            selected.append(rc)
    
    # If we didn't get enough, relax diversity and take top K
    if len(selected) < k:
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