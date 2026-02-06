/**
 * Edge Question Generation Pipeline - Post-Generation (Simplified)
 * Minimal validation, no evidence checking
 */

import {
  QuestionCandidate,
  ValidationResult,
  RejectedCandidate,
  RankedCandidate,
  FinalQuestion
} from './pipeline_types.js';

// ============================================================================
// Minimal Validation - No Evidence Checking
// ============================================================================

export function validateCandidates(candidates, chunks, allowedStyles = null) {
  const valid = [];
  const rejected = [];
  
  for (const candidate of candidates) {
    // Check 1: Question exists and not empty
    if (!candidate.question || candidate.question.trim().length < 10) {
      rejected.push(new RejectedCandidate(
        candidate,
        "Question too short or empty"
      ));
      continue;
    }
    
    // Passed all checks
    valid.push(candidate);
  }
  
  return new ValidationResult(valid, rejected);
}

// ============================================================================
// No Deduplication
// ============================================================================

export function deduplicateCandidates(candidates, similarityThreshold = 0.7) {
  // No deduplication - return all candidates as-is
  return candidates;
}

// ============================================================================
// Neutral Ranking - Keep Original Order
// ============================================================================

export function rankCandidates(candidates, chunks) {
  const ranked = [];
  
  for (const candidate of candidates) {
    ranked.push(new RankedCandidate(
      candidate,
      1.0,  // Neutral score
      { composite: 1.0 }
    ));
  }
  
  return ranked;
}

// ============================================================================
// Simple Selection - Take First K
// ============================================================================

export function selectFinalQuestions(ranked, k, diversityThreshold = 0.6) {
  // Take first K (or all if we have fewer)
  const selected = ranked.slice(0, k);
  
  // Convert to FinalQuestion
  return selected.map(rc => new FinalQuestion(
    rc.candidate.question,
    rc.candidate.type,
    rc.candidate.chunkId,
    rc.score
  ));
}