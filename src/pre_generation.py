"""
Edge Question Generation Pipeline - Pre-Generation
Deterministic chunking and candidate allocation
"""

import re
from typing import List
from src.pipeline_types import ArticleInput, Chunk
from src.errors import ChunkingError


# ============================================================================
# Constants
# ============================================================================

MAX_CHUNK_WORDS = 800
MIN_OVERLAP_WORDS = 100
MAX_OVERLAP_WORDS = 150

# Chunking policy thresholds
POLICY = [
    (500, 800, 1),      # 500-800 words -> 1 chunk
    (800, 1600, 2),     # 800-1600 words -> 2 chunks
    (1600, 3200, 3),    # 1600-3200 words -> 3-4 chunks
    (3200, 5000, 5),    # 3200-5000 words -> 5-7 chunks
]


# ============================================================================
# Text Normalization
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize article text for consistent processing.
    
    - Collapses multiple whitespaces
    - Removes excessive newlines
    - Strips leading/trailing whitespace
    """
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Collapse multiple newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_words(text: str) -> int:
    """Count words in text (whitespace-separated tokens)"""
    return len(text.split())


# ============================================================================
# Chunking Logic
# ============================================================================

def calculate_chunk_count(word_count: int) -> int:
    """
    Determine number of chunks based on article length.
    
    Follows chunking policy with dynamic allocation:
    - 500-800 words: 1 chunk
    - 800-1600 words: 2 chunks
    - 1600-3200 words: 3-4 chunks (scaled)
    - 3200-5000 words: 5-7 chunks (scaled)
    """
    if word_count < 500:
        # Articles under 500 words get 1 chunk
        return 1
    
    for min_words, max_words, base_chunks in POLICY:
        if min_words <= word_count <= max_words:
            # For ranges with variable chunks, scale linearly
            if max_words - min_words > 800:
                # Large ranges (1600-3200, 3200-5000)
                range_ratio = (word_count - min_words) / (max_words - min_words)
                if max_words == 3200:
                    # 3-4 chunks for 1600-3200 range
                    return base_chunks + int(range_ratio * 1)
                else:
                    # 5-7 chunks for 3200-5000 range
                    return base_chunks + int(range_ratio * 2)
            else:
                # Fixed chunk count for small ranges
                return base_chunks
    
    # Articles over 5000 words: cap at 7 chunks
    return min(7, (word_count // MAX_CHUNK_WORDS) + 1)


def split_into_words(text: str) -> List[str]:
    """Split text into word tokens, preserving whitespace info"""
    return text.split()


def calculate_overlap_size(total_words: int, num_chunks: int) -> int:
    """
    Calculate optimal overlap size based on article and chunk count.
    
    Returns overlap in word count between MIN_OVERLAP_WORDS and MAX_OVERLAP_WORDS.
    Smaller articles get smaller overlap.
    """
    if num_chunks <= 1:
        return 0
    
    # Base overlap on article size
    if total_words < 1000:
        return MIN_OVERLAP_WORDS
    elif total_words > 3000:
        return MAX_OVERLAP_WORDS
    else:
        # Linear interpolation
        ratio = (total_words - 1000) / 2000
        return int(MIN_OVERLAP_WORDS + ratio * (MAX_OVERLAP_WORDS - MIN_OVERLAP_WORDS))


def chunk_article(article: ArticleInput) -> List[Chunk]:
    """
    Apply dynamic chunking policy with overlap for context preservation.
    
    Guarantees:
    - Chunk size bounds: max 800 words / ~1040 tokens
    - Context overlap: 100-150 words between consecutive chunks
    - Deterministic output for same input
    
    Args:
        article: Input article with text
        
    Returns:
        List of chunks with metadata
        
    Raises:
        ChunkingError: If text is empty or chunking fails
    """
    # Normalize input
    normalized_text = normalize_text(article.text)
    
    if not normalized_text:
        raise ChunkingError("Article text is empty after normalization")
    
    # Count words
    word_count = count_words(normalized_text)
    
    if word_count < 50:
        raise ChunkingError(f"Article too short: {word_count} words (minimum 50)")
    
    # Determine chunking strategy
    num_chunks = calculate_chunk_count(word_count)
    overlap_words = calculate_overlap_size(word_count, num_chunks)
    
    # Split into words
    words = split_into_words(normalized_text)
    
    # Single chunk case
    if num_chunks == 1:
        return [Chunk(
            chunk_id=0,
            text=normalized_text,
            start_offset=0,
            end_offset=len(normalized_text)
        )]
    
    # Calculate chunk size
    # Account for overlap: total_words = (num_chunks * chunk_size) - ((num_chunks - 1) * overlap)
    # Solving for chunk_size: chunk_size = (total_words + (num_chunks - 1) * overlap) / num_chunks
    effective_chunk_size = (word_count + (num_chunks - 1) * overlap_words) // num_chunks
    effective_chunk_size = min(effective_chunk_size, MAX_CHUNK_WORDS)
    
    # Create chunks with overlap
    chunks: List[Chunk] = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Calculate end index
        if i == num_chunks - 1:
            # Last chunk takes all remaining words
            end_idx = len(words)
        else:
            end_idx = min(start_idx + effective_chunk_size, len(words))
        
        # Extract chunk words
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        # Calculate character offsets in original text
        char_start = normalized_text.find(chunk_words[0]) if chunk_words else 0
        char_end = char_start + len(chunk_text)
        
        chunks.append(Chunk(
            chunk_id=i,
            text=chunk_text,
            start_offset=char_start,
            end_offset=char_end
        ))
        
        # Move start index forward, accounting for overlap
        start_idx = end_idx - overlap_words
        
        # Ensure we don't go backwards
        if start_idx < 0:
            start_idx = 0
    
    return chunks


# ============================================================================
# Candidate Allocation
# ============================================================================

def allocate_candidates(num_chunks: int, desired_questions: int) -> List[int]:
    """
    Calculate candidate allocation per chunk.
    
    Strategy: Generate ~2×K total candidates across all chunks for selection headroom.
    Individual chunk allocation: c_i = clamp(1, 3, ceil((2 * K) / N))
    
    Args:
        num_chunks: Number of chunks to process
        desired_questions: Target number of final questions (K)
        
    Returns:
        List of candidate counts aligned with chunk array (length = num_chunks)
    """
    if num_chunks <= 0:
        return []
    
    if desired_questions <= 0:
        desired_questions = 4  # Default
    
    # Total candidates to generate (2×K for selection headroom)
    total_candidates = 2 * desired_questions
    
    # Base allocation per chunk
    candidates_per_chunk = total_candidates / num_chunks
    
    # Clamp to [1, 3] and round up
    import math
    base_allocation = max(1, min(3, math.ceil(candidates_per_chunk)))
    
    # Create allocation list
    allocations = [base_allocation] * num_chunks
    
    # Adjust to hit target total (distribute remainder)
    current_total = sum(allocations)
    target_total = total_candidates
    
    if current_total < target_total:
        # Add extra candidates to first few chunks (up to limit of 3)
        diff = target_total - current_total
        for i in range(min(diff, num_chunks)):
            if allocations[i] < 3:
                allocations[i] += 1
    
    return allocations


# ============================================================================
# Public API Summary
# ============================================================================

__all__ = [
    'normalize_text',
    'chunk_article',
    'allocate_candidates',
    'calculate_chunk_count',
]