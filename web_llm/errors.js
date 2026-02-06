/**
 * Edge Question Generation Pipeline - Error Handling
 * Custom exceptions for pipeline stages
 */

import { PipelineStage, InferenceErrorCause } from './pipeline_types.js';
// ============================================================================
// Pipeline Errors
// ============================================================================

export class PipelineError extends Error {
  constructor(message, stage, recoverable = false) {
    super(message);
    this.name = 'PipelineError';
    this.stage = stage;
    this.recoverable = recoverable;
    this.message = message;
  }

  toString() {
    const recoveryStatus = this.recoverable ? "recoverable" : "non-recoverable";
    return `[${this.stage}] ${this.message} (${recoveryStatus})`;
  }
}

export class ValidationError extends PipelineError {
  constructor(message, rejectedCount, validCount) {
    super(
      message,
      PipelineStage.POST_GEN,
      validCount > 0
    );
    this.name = 'ValidationError';
    this.rejectedCount = rejectedCount;
    this.validCount = validCount;
  }

  toString() {
    return `[${this.stage}] ${this.message} (valid: ${this.validCount}, rejected: ${this.rejectedCount})`;
  }
}

export class ChunkingError extends PipelineError {
  constructor(message) {
    super(
      message,
      PipelineStage.PRE_GEN,
      false
    );
    this.name = 'ChunkingError';
  }
}

export class ModelInferenceError extends Error {
  constructor(message, cause, chunkId = null) {
    super(message);
    this.name = 'ModelInferenceError';
    this.message = message;
    this.cause = cause;
    this.chunkId = chunkId;
  }

  toString() {
    let base = `[${this.cause}] ${this.message}`;
    if (this.chunkId !== null) {
      base += ` (chunk_id=${this.chunkId})`;
    }
    return base;
  }

  isRetryable() {
    return this.cause === InferenceErrorCause.TIMEOUT ||
           this.cause === InferenceErrorCause.UNKNOWN;
  }
}