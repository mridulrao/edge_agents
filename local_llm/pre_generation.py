"""
Edge Question Generation Pipeline - Pre-Generation

UPDATED (for 4 configurations A/B/C/D):
✅ Adds `chunk_article_for_mode(article, mode)`:
   - CHUNKED modes (A/C): use your existing fixed/overlap chunking logic
   - FULL_ARTICLE modes (B/D): return a single chunk containing the full (substantive) article text
✅ Adds `calculate_required_chunks_for_mode(...)`:
   - CHUNKED: keeps BUFFER_MULTIPLIER behavior
   - FULL_ARTICLE: always 1
✅ Adds `allocate_candidates_for_mode(...)`:
   - CHUNKED: [1, 1, 1, ...]
   - FULL_ARTICLE: [mode.questions_per_call] (one call yields N)
✅ Keeps your existing functions intact as much as possible so minimal downstream breakage
"""

from __future__ import annotations

import re
import math
from typing import List, Tuple

from local_llm.pipeline_types import ArticleInput, Chunk
from local_llm.errors import ChunkingError

# NEW: modes (from model_adapter.py)
from local_llm.model_adapter import QuestionGenMode, GenerationScope


# ============================================================================
# Constants
# ============================================================================

# Chunk sizing tuned for small models (90M–350M)
TARGET_CHUNK_WORDS = 40        # Aim for ~40 words per chunk
MIN_CHUNK_WORDS = 20           # Minimum viable chunk size
MAX_CHUNK_WORDS = 80           # Cap for focus / JSON stability

# Content selection: skip intro/outro percentages (only for long docs)
SKIP_INTRO_PERCENT = 0.10      # Skip first 10%
SKIP_OUTRO_PERCENT = 0.05      # Skip last 5%
MIN_SUBSTANTIVE_WORDS = 1000   # Need at least this much to apply skipping

# Buffer strategy: create extra chunks for desired questions (bounded by available windows)
BUFFER_MULTIPLIER = 1.5

# If True: when non-overlapping chunks < required, use sliding window overlap
ALLOW_OVERLAP_TO_MEET_DESIRED = True


# ============================================================================
# Text Normalization
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize article text for consistent processing."""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text (whitespace-separated tokens)."""
    return len(text.split())


# ============================================================================
# Content Selection
# ============================================================================

def extract_substantive_content(text: str, word_count: int) -> Tuple[str, int, int]:
    """
    Extract substantive middle content, skipping intro/outro for very long docs.
    Returns: (substantive_text, start_word_idx, end_word_idx)
    """
    if word_count < MIN_SUBSTANTIVE_WORDS:
        return text, 0, word_count

    words = text.split()

    skip_intro_words = int(word_count * SKIP_INTRO_PERCENT)
    skip_outro_words = int(word_count * SKIP_OUTRO_PERCENT)

    start_idx = skip_intro_words
    end_idx = word_count - skip_outro_words

    if end_idx - start_idx < MIN_CHUNK_WORDS:
        return text, 0, word_count

    substantive_words = words[start_idx:end_idx]
    return " ".join(substantive_words), start_idx, end_idx


# ============================================================================
# Chunking Logic (existing)
# ============================================================================

def _word_start_char_positions(words: List[str]) -> Tuple[str, List[int]]:
    rebuilt = " ".join(words)
    starts: List[int] = []
    pos = 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1  # +1 space
    return rebuilt, starts


def _max_windows(total_words: int, window: int) -> int:
    """Max number of sliding windows with stride=1."""
    if total_words < window:
        return 1
    return (total_words - window) + 1


def _natural_nonoverlap_chunks(total_words: int, window: int) -> int:
    """How many chunks we get if we chunk sequentially by window size."""
    if total_words <= 0:
        return 0
    return max(1, math.ceil(total_words / window))


def calculate_required_chunks(total_words: int, desired_questions: int) -> int:
    """
    CHUNKED target:
    - want at least desired_questions
    - plus buffer (BUFFER_MULTIPLIER)
    """
    desired_questions = max(1, int(desired_questions))
    target = max(desired_questions, int(math.ceil(desired_questions * BUFFER_MULTIPLIER)))
    return max(1, target)


def _build_chunk(words: List[str], word_starts: List[int], span: Tuple[int, int], chunk_id: int) -> Chunk:
    s, e = span
    chunk_words = words[s:e]
    chunk_text = " ".join(chunk_words)

    char_start = word_starts[s] if 0 <= s < len(word_starts) else 0
    if (e - 1) >= 0 and (e - 1) < len(word_starts):
        char_end = word_starts[e - 1] + len(words[e - 1])
    else:
        char_end = len(" ".join(words))

    return Chunk(chunk_id=chunk_id, text=chunk_text, start_offset=char_start, end_offset=char_end)


def create_chunks_fixed_or_overlap(text: str, required_chunks: int) -> List[Chunk]:
    """
    Create ~TARGET_CHUNK_WORDS chunks.
    If non-overlapping chunks are fewer than required_chunks and ALLOW_OVERLAP_TO_MEET_DESIRED,
    use a sliding window to reach required_chunks (bounded by what's possible).
    """
    words = text.split()
    total_words = len(words)

    if total_words < MIN_CHUNK_WORDS:
        return [Chunk(chunk_id=0, text=text, start_offset=0, end_offset=len(text))]

    window = max(MIN_CHUNK_WORDS, min(TARGET_CHUNK_WORDS, MAX_CHUNK_WORDS))

    if total_words <= window:
        rebuilt, word_starts = _word_start_char_positions(words)
        _ = rebuilt
        return [_build_chunk(words, word_starts, (0, total_words), 0)]

    nonoverlap_n = _natural_nonoverlap_chunks(total_words, window)

    if nonoverlap_n >= required_chunks or not ALLOW_OVERLAP_TO_MEET_DESIRED:
        spans: List[Tuple[int, int]] = []
        start = 0
        while start < total_words:
            end = min(start + window, total_words)
            spans.append((start, end))
            start = end

        if len(spans) >= 2:
            last_s, last_e = spans[-1]
            if (last_e - last_s) < MIN_CHUNK_WORDS:
                prev_s, prev_e = spans[-2]
                spans[-2] = (prev_s, last_e)
                spans.pop()

        spans = spans[:max(1, required_chunks)]

        if len(spans) >= 2:
            last_s, last_e = spans[-1]
            if (last_e - last_s) < MIN_CHUNK_WORDS:
                prev_s, prev_e = spans[-2]
                spans[-2] = (prev_s, last_e)
                spans.pop()

        rebuilt, word_starts = _word_start_char_positions(words)
        _ = rebuilt
        return [_build_chunk(words, word_starts, sp, i) for i, sp in enumerate(spans)]

    # Overlap
    max_possible = _max_windows(total_words, window)
    final_n = min(required_chunks, max_possible)

    available = total_words - window
    if final_n <= 1:
        stride = available
    else:
        stride = max(1, available // (final_n - 1))

    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(final_n):
        if i == final_n - 1:
            start = total_words - window
        end = start + window
        spans.append((start, end))
        start = start + stride
        if start > total_words - window:
            start = total_words - window

    rebuilt, word_starts = _word_start_char_positions(words)
    _ = rebuilt
    chunks = [_build_chunk(words, word_starts, sp, i) for i, sp in enumerate(spans)]

    if len(chunks) >= 2:
        last_words = len(chunks[-1].text.split())
        if last_words < MIN_CHUNK_WORDS:
            prev = chunks[-2]
            last = chunks[-1]
            merged_text = (prev.text + " " + last.text).strip()
            chunks[-2] = Chunk(
                chunk_id=prev.chunk_id,
                text=merged_text,
                start_offset=prev.start_offset,
                end_offset=last.end_offset,
            )
            chunks.pop()

    return chunks


# ============================================================================
# NEW: Mode-aware chunking entrypoint
# ============================================================================

def calculate_required_chunks_for_mode(total_words: int, desired_questions: int, mode: QuestionGenMode) -> int:
    """
    FULL_ARTICLE modes: always 1 chunk (one call)
    CHUNKED modes: existing buffered behavior
    """
    if mode.scope == GenerationScope.FULL_ARTICLE:
        return 1
    return calculate_required_chunks(total_words, desired_questions)


def chunk_article_for_mode(article: ArticleInput, mode: QuestionGenMode) -> List[Chunk]:
    """
    Mode-aware chunking.

    - CHUNKED (A/C): return multiple chunks using fixed/overlap strategy.
    - FULL_ARTICLE (B/D): return a single chunk containing full substantive text.
      (Still applies normalization + optional intro/outro skipping for very long docs.)
    """
    normalized_text = normalize_text(article.text)
    if not normalized_text:
        raise ChunkingError("Article text is empty after normalization")

    total_word_count = count_words(normalized_text)
    if total_word_count < MIN_CHUNK_WORDS:
        raise ChunkingError(
            f"Article too short: {total_word_count} words (minimum {MIN_CHUNK_WORDS})"
        )

    text_to_use, _, _ = extract_substantive_content(normalized_text, total_word_count)
    substantive_word_count = count_words(text_to_use)

    desired_q = max(1, int(article.desired_questions))

    if mode.scope == GenerationScope.FULL_ARTICLE:
        # single chunk, preserve offsets roughly (0..len)
        return [Chunk(chunk_id=0, text=text_to_use, start_offset=0, end_offset=len(text_to_use))]

    # CHUNKED path (existing)
    required_chunks = calculate_required_chunks_for_mode(substantive_word_count, desired_q, mode)
    chunks = create_chunks_fixed_or_overlap(text_to_use, required_chunks)

    if not chunks:
        raise ChunkingError("Failed to create any chunks")

    return chunks


# ============================================================================
# Candidate Allocation (mode-aware)
# ============================================================================

def allocate_candidates_for_mode(num_chunks: int, desired_questions: int, mode: QuestionGenMode) -> List[int]:
    """
    For CHUNKED modes: 1 question per chunk (stable)
    For FULL_ARTICLE modes: single call produces N questions (mode.questions_per_call)
    """
    if mode.scope == GenerationScope.FULL_ARTICLE:
        n = max(1, int(mode.questions_per_call))
        return [n]

    # CHUNKED
    return [1] * max(1, int(num_chunks))


# ============================================================================
# Backwards-compatible API (existing names)
# ============================================================================

def chunk_article(article: ArticleInput) -> List[Chunk]:
    """
    Backwards-compatible: original behavior = CHUNKED JSON 1 (Config A).
    """
    # Avoid import cycle by constructing mode here
    mode = QuestionGenMode.A_chunked_json_1()
    return chunk_article_for_mode(article, mode)


def allocate_candidates(num_chunks: int, desired_questions: int) -> List[int]:
    """
    Backwards-compatible: original behavior = 1 question per chunk.
    """
    return [1] * max(1, int(num_chunks))


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "normalize_text",
    "count_words",
    "extract_substantive_content",
    "calculate_required_chunks",
    "create_chunks_fixed_or_overlap",
    # mode-aware
    "calculate_required_chunks_for_mode",
    "chunk_article_for_mode",
    "allocate_candidates_for_mode",
    # old API
    "chunk_article",
    "allocate_candidates",
]
