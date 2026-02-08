"""
Edge Question Generation Pipeline - Pre-Generation
Optimized chunking for small models with buffer strategy
"""

import re
from typing import List, Tuple
from local_llm.pipeline_types import ArticleInput, Chunk
from local_llm.errors import ChunkingError


# ============================================================================
# Constants
# ============================================================================

TARGET_CHUNK_WORDS = 40      # Optimal for small models
MIN_CHUNK_WORDS = 20         # Minimum viable chunk
MAX_CHUNK_WORDS = 80         # Maximum to stay focused

# Content selection: skip intro/outro percentages
SKIP_INTRO_PERCENT = 0.15     # Skip first 15%
SKIP_OUTRO_PERCENT = 0.10     # Skip last 10%
MIN_SUBSTANTIVE_WORDS = 300   # Need at least this much to apply skipping

# Buffer strategy
BUFFER_MULTIPLIER = 1       # Create 1.5x chunks for desired questions


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
# Content Selection
# ============================================================================

def extract_substantive_content(text: str, word_count: int) -> Tuple[str, int, int]:
    """
    Extract substantive middle content, skipping intro/outro.
    
    Strategy:
    - For short articles (<300 words): use entire text
    - For longer articles: skip first 15% and last 10%
    - Returns the extracted text and start/end word indices
    
    Args:
        text: Normalized article text
        word_count: Total word count
        
    Returns:
        Tuple of (substantive_text, start_word_idx, end_word_idx)
    """
    if word_count < MIN_SUBSTANTIVE_WORDS:
        # Too short to skip anything
        return text, 0, word_count
    
    words = text.split()
    
    # Calculate skip boundaries
    skip_intro_words = int(word_count * SKIP_INTRO_PERCENT)
    skip_outro_words = int(word_count * SKIP_OUTRO_PERCENT)
    
    # Extract middle section
    start_idx = skip_intro_words
    end_idx = word_count - skip_outro_words
    
    # Safety check
    if end_idx - start_idx < MIN_CHUNK_WORDS:
        # After skipping, too little remains - use full text
        return text, 0, word_count
    
    substantive_words = words[start_idx:end_idx]
    substantive_text = ' '.join(substantive_words)
    
    return substantive_text, start_idx, end_idx


# ============================================================================
# Chunking Logic
# ============================================================================

def calculate_chunk_count(word_count: int, desired_questions: int) -> int:
    """
    Calculate number of chunks needed with buffer strategy.
    
    Strategy:
    - Want N questions → create 1.5×N chunks (buffer for failures)
    - Cap at maximum possible given word count
    
    Args:
        word_count: Number of words in substantive content
        desired_questions: Target number of questions
        
    Returns:
        Number of chunks to create
    """
    if word_count < MIN_CHUNK_WORDS:
        return 1
    
    # Maximum possible chunks given word count and chunk size
    max_possible = word_count // MIN_CHUNK_WORDS
    
    # Desired chunks with buffer
    target_chunks = int(desired_questions * BUFFER_MULTIPLIER)
    
    # Ensure minimum (at least desired_questions)
    target_chunks = max(target_chunks, desired_questions)
    
    # Cap at what's possible
    return min(target_chunks, max_possible)


def create_chunks(
    text: str,
    num_chunks: int
) -> List[Chunk]:
    """
    Create non-overlapping chunks from text.
    
    Args:
        text: Text to chunk
        num_chunks: Target number of chunks
        
    Returns:
        List of chunks with metadata
    """
    words = text.split()
    total_words = len(words)
    
    if total_words < MIN_CHUNK_WORDS:
        # Single chunk
        return [Chunk(
            chunk_id=0,
            text=text,
            start_offset=0,
            end_offset=len(text)
        )]
    
    if num_chunks == 1:
        # Take first MAX_CHUNK_WORDS
        chunk_words = words[:MAX_CHUNK_WORDS]
        chunk_text = ' '.join(chunk_words)
        return [Chunk(
            chunk_id=0,
            text=chunk_text,
            start_offset=0,
            end_offset=len(chunk_text)
        )]
    
    # Calculate chunk size (evenly distributed)
    chunk_size = total_words // num_chunks
    chunk_size = max(MIN_CHUNK_WORDS, min(chunk_size, TARGET_CHUNK_WORDS))
    
    chunks: List[Chunk] = []
    position = 0
    
    for i in range(num_chunks):
        # Calculate chunk boundaries
        start_idx = position
        
        if i == num_chunks - 1:
            # Last chunk takes all remaining words
            end_idx = total_words
        else:
            end_idx = min(position + chunk_size, total_words)
        
        # Ensure minimum chunk size
        if end_idx - start_idx < MIN_CHUNK_WORDS and i > 0:
            # Too small, extend backwards if possible
            start_idx = max(0, end_idx - MIN_CHUNK_WORDS)
        
        # Extract chunk
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        # Calculate character offsets
        char_start = len(' '.join(words[:start_idx])) + (1 if start_idx > 0 else 0)
        char_end = char_start + len(chunk_text)
        
        chunks.append(Chunk(
            chunk_id=i,
            text=chunk_text,
            start_offset=char_start,
            end_offset=char_end
        ))
        
        # Move position forward
        position = end_idx
        
        if position >= total_words:
            break
    
    return chunks


def chunk_article(article: ArticleInput) -> List[Chunk]:
    """
    Create focused chunks optimized for single-question generation.
    
    Strategy:
    - Chunks of ~200 words (optimal for small models)
    - Skip intro/outro for better quality
    - Create 1.5x chunks as buffer (want 4 questions → create 6 chunks)
    - One question per chunk (stable JSON parsing)
    
    Args:
        article: Input article with text and desired_questions
        
    Returns:
        List of chunks optimized for question generation
        
    Raises:
        ChunkingError: If text is invalid
    """
    # Normalize input
    normalized_text = normalize_text(article.text)
    
    if not normalized_text:
        raise ChunkingError("Article text is empty after normalization")
    
    # Count words
    total_word_count = count_words(normalized_text)
    
    if total_word_count < 50:
        raise ChunkingError(f"Article too short: {total_word_count} words (minimum 50)")
    
    # Extract substantive content (skip intro/outro)
    text_to_chunk, start_word_idx, end_word_idx = extract_substantive_content(
        normalized_text, 
        total_word_count
    )
    substantive_word_count = count_words(text_to_chunk)
    
    # Calculate required chunks
    num_chunks = calculate_chunk_count(
        substantive_word_count,
        article.desired_questions
    )
    
    # Create chunks
    chunks = create_chunks(text_to_chunk, num_chunks)
    
    return chunks


# ============================================================================
# Candidate Allocation
# ============================================================================

def allocate_candidates(num_chunks: int, desired_questions: int) -> List[int]:
    """
    Calculate candidate allocation per chunk.
    
    New strategy: ALWAYS 1 question per chunk for stability.
    - Small models struggle with multi-question JSON
    - Parsing is more reliable with single outputs
    - Better fault tolerance: one chunk fails = lose 1 question, not 3
    
    Args:
        num_chunks: Number of chunks to process
        desired_questions: Target number of final questions (for reference)
        
    Returns:
        List of 1s (length = num_chunks), one candidate per chunk
    """
    # Simple: always generate 1 question per chunk
    return [1] * num_chunks


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'normalize_text',
    'chunk_article',
    'allocate_candidates',
    'calculate_chunk_count',
    'extract_substantive_content',
]