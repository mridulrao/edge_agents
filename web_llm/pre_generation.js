/**
 * Edge Question Generation Pipeline - Pre-Generation
 * Optimized chunking for small models with buffer strategy
 */

import { Chunk } from './pipeline_types.js';
import { ChunkingError } from './errors.js';

// ============================================================================
// Constants
// ============================================================================

const TARGET_CHUNK_WORDS = 40;      // Optimal for small models
const MIN_CHUNK_WORDS = 20;          // Minimum viable chunk
const MAX_CHUNK_WORDS = 80;         // Maximum to stay focused

// Content selection: skip intro/outro percentages
const SKIP_INTRO_PERCENT = 0.15;     // Skip first 15%
const SKIP_OUTRO_PERCENT = 0.10;     // Skip last 10%
const MIN_SUBSTANTIVE_WORDS = 300;   // Need at least this much to apply skipping

// Buffer strategy
const BUFFER_MULTIPLIER = 1;       // Create 1.5x chunks for desired questions

// ============================================================================
// Text Normalization
// ============================================================================

export function normalizeText(text) {
  // Collapse multiple spaces
  text = text.replace(/ +/g, ' ');
  
  // Collapse multiple newlines to max 2
  text = text.replace(/\n{3,}/g, '\n\n');
  
  // Strip leading/trailing whitespace
  text = text.trim();
  
  return text;
}

export function countWords(text) {
  return text.split(/\s+/).filter(w => w.length > 0).length;
}

// ============================================================================
// Content Selection
// ============================================================================

export function extractSubstantiveContent(text, wordCount) {
  if (wordCount < MIN_SUBSTANTIVE_WORDS) {
    return { text, startWordIdx: 0, endWordIdx: wordCount };
  }
  
  const words = text.split(/\s+/);
  
  // Calculate skip boundaries
  const skipIntroWords = Math.floor(wordCount * SKIP_INTRO_PERCENT);
  const skipOutroWords = Math.floor(wordCount * SKIP_OUTRO_PERCENT);
  
  // Extract middle section
  const startIdx = skipIntroWords;
  const endIdx = wordCount - skipOutroWords;
  
  // Safety check
  if (endIdx - startIdx < MIN_CHUNK_WORDS) {
    return { text, startWordIdx: 0, endWordIdx: wordCount };
  }
  
  const substantiveWords = words.slice(startIdx, endIdx);
  const substantiveText = substantiveWords.join(' ');
  
  return {
    text: substantiveText,
    startWordIdx: startIdx,
    endWordIdx: endIdx
  };
}

// ============================================================================
// Chunking Logic
// ============================================================================

export function calculateChunkCount(wordCount, desiredQuestions) {
  if (wordCount < MIN_CHUNK_WORDS) {
    return 1;
  }
  
  // Maximum possible chunks given word count and chunk size
  const maxPossible = Math.floor(wordCount / MIN_CHUNK_WORDS);
  
  // Desired chunks with buffer
  let targetChunks = Math.floor(desiredQuestions * BUFFER_MULTIPLIER);
  
  // Ensure minimum (at least desired_questions)
  targetChunks = Math.max(targetChunks, desiredQuestions);
  
  // Cap at what's possible
  return Math.min(targetChunks, maxPossible);
}

export function createChunks(text, numChunks) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const totalWords = words.length;
  
  if (totalWords < MIN_CHUNK_WORDS) {
    // Single chunk
    return [new Chunk(0, text, 0, text.length)];
  }
  
  if (numChunks === 1) {
    // Take first MAX_CHUNK_WORDS
    const chunkWords = words.slice(0, MAX_CHUNK_WORDS);
    const chunkText = chunkWords.join(' ');
    return [new Chunk(0, chunkText, 0, chunkText.length)];
  }
  
  // Calculate chunk size (evenly distributed)
  let chunkSize = Math.floor(totalWords / numChunks);
  chunkSize = Math.max(MIN_CHUNK_WORDS, Math.min(chunkSize, TARGET_CHUNK_WORDS));
  
  const chunks = [];
  let position = 0;
  
  for (let i = 0; i < numChunks; i++) {
    // Calculate chunk boundaries
    let startIdx = position;
    let endIdx;
    
    if (i === numChunks - 1) {
      // Last chunk takes all remaining words
      endIdx = totalWords;
    } else {
      endIdx = Math.min(position + chunkSize, totalWords);
    }
    
    // Ensure minimum chunk size
    if (endIdx - startIdx < MIN_CHUNK_WORDS && i > 0) {
      // Too small, extend backwards if possible
      startIdx = Math.max(0, endIdx - MIN_CHUNK_WORDS);
    }
    
    // Extract chunk
    const chunkWords = words.slice(startIdx, endIdx);
    const chunkText = chunkWords.join(' ');
    
    // Calculate character offsets
    const precedingText = words.slice(0, startIdx).join(' ');
    const charStart = precedingText.length + (startIdx > 0 ? 1 : 0);
    const charEnd = charStart + chunkText.length;
    
    chunks.push(new Chunk(i, chunkText, charStart, charEnd));
    
    // Move position forward
    position = endIdx;
    
    if (position >= totalWords) {
      break;
    }
  }
  
  return chunks;
}

export function chunkArticle(article) {
  // Normalize input
  const normalizedText = normalizeText(article.text);
  
  if (!normalizedText) {
    throw new ChunkingError("Article text is empty after normalization");
  }
  
  // Count words
  const totalWordCount = countWords(normalizedText);
  
  if (totalWordCount < 50) {
    throw new ChunkingError(`Article too short: ${totalWordCount} words (minimum 50)`);
  }
  
  // Extract substantive content (skip intro/outro)
  const { text: textToChunk, startWordIdx, endWordIdx } = extractSubstantiveContent(
    normalizedText,
    totalWordCount
  );
  const substantiveWordCount = countWords(textToChunk);
  
  // Calculate required chunks
  const numChunks = calculateChunkCount(
    substantiveWordCount,
    article.desiredQuestions
  );
  
  // Create chunks
  const chunks = createChunks(textToChunk, numChunks);
  
  return chunks;
}

// ============================================================================
// Candidate Allocation
// ============================================================================

export function allocateCandidates(numChunks, desiredQuestions) {
  // Simple: always generate 1 question per chunk
  return new Array(numChunks).fill(1);
}