/**
 * Type definitions for the entire pipeline system
 * JavaScript equivalent of pipeline_types.py
 */

// ============================================================================
// Enums
// ============================================================================

export const QuestionType = {
  PROCEDURAL: "procedural",
  TROUBLESHOOTING: "troubleshooting",
  CONFIGURATION: "configuration",
  CONCEPTUAL: "conceptual",
  TEMPORAL: "temporal",
  COMPARATIVE: "comparative",
  DIAGNOSTIC: "diagnostic",
  DEFINITIONAL: "definitional"
};

export const QuestionStyle = {
  INTERROGATIVE: "interrogative",
  IMPERATIVE: "imperative",
  CONTEXTUAL: "contextual"
};

export const PipelineStage = {
  PRE_GEN: "pre-gen",
  GENERATION: "generation",
  POST_GEN: "post-gen"
};

export const InferenceErrorCause = {
  TIMEOUT: "timeout",
  OOM: "out_of_memory",
  PARSE_ERROR: "parse_error",
  MODEL_NOT_READY: "model_not_ready",
  MODEL_LOAD_ERROR: "model_load_error",
  INSUFFICIENT_OUTPUT: "insufficient_output",
  UNKNOWN: "unknown"
};

// ============================================================================
// Input Types
// ============================================================================

export class ArticleInput {
  constructor(text, desiredQuestions = 4) {
    this.text = text;
    this.desiredQuestions = desiredQuestions;
  }
}

// ============================================================================
// Pre-Generation Types
// ============================================================================

export class Chunk {
  constructor(chunkId, text, startOffset, endOffset) {
    this.chunkId = chunkId;
    this.text = text;
    this.startOffset = startOffset;
    this.endOffset = endOffset;
  }

  length() {
    return this.text.split(/\s+/).length;
  }
}

// ============================================================================
// Generation Types
// ============================================================================

export class QuestionCandidate {
  constructor(question, type, evidence, chunkId, style = QuestionStyle.INTERROGATIVE) {
    this.question = question;
    this.type = type;
    this.evidence = evidence;
    this.chunkId = chunkId;
    this.style = style;
  }

  toDict() {
    return {
      question: this.question,
      type: this.type,
      evidence: this.evidence,
      chunkId: this.chunkId,
      style: this.style
    };
  }
}

export class GenerationConfig {
  constructor(options = {}) {
    this.maxOutputTokens = options.maxOutputTokens || 200;
    this.temperature = options.temperature !== undefined ? options.temperature : 0.2;
    this.topP = options.topP !== undefined ? options.topP : 0.9;
    this.stopSequences = options.stopSequences || ["}]", "\n\n\n"];
    
    // Question generation preferences
    this.questionTypes = options.questionTypes || [
      QuestionType.PROCEDURAL,
      QuestionType.CONFIGURATION,
      QuestionType.CONCEPTUAL
    ];
    this.questionStyles = options.questionStyles || [
      QuestionStyle.INTERROGATIVE,
      QuestionStyle.CONTEXTUAL
    ];
    this.preferredQuestionWords = options.preferredQuestionWords || [
      "how", "what", "when", "which", "why", "where"
    ];
    this.allowNonQuestionFormat = options.allowNonQuestionFormat !== undefined 
      ? options.allowNonQuestionFormat 
      : true;
    
    this.validate();
  }

  validate() {
    if (this.temperature < 0.0 || this.temperature > 1.0) {
      throw new Error(`temperature must be in [0, 1], got ${this.temperature}`);
    }
    if (this.topP < 0.0 || this.topP > 1.0) {
      throw new Error(`topP must be in [0, 1], got ${this.topP}`);
    }
    if (this.maxOutputTokens <= 0) {
      throw new Error(`maxOutputTokens must be positive, got ${this.maxOutputTokens}`);
    }
  }
}

// ============================================================================
// Post-Generation Types
// ============================================================================

export class RejectedCandidate {
  constructor(candidate, reason) {
    this.candidate = candidate;
    this.reason = reason;
  }
}

export class ValidationResult {
  constructor(valid, rejected) {
    this.valid = valid;
    this.rejected = rejected;
  }

  get passRate() {
    const total = this.valid.length + this.rejected.length;
    return total > 0 ? this.valid.length / total : 0.0;
  }
}

export class RankedCandidate {
  constructor(candidate, score, scoreBreakdown = {}) {
    this.candidate = candidate;
    this.score = score;
    this.scoreBreakdown = scoreBreakdown;
  }
}

export class FinalQuestion {
  constructor(question, type, sourceChunkId, confidenceScore) {
    this.question = question;
    this.type = type;
    this.sourceChunkId = sourceChunkId;
    this.confidenceScore = confidenceScore;
  }

  toDict() {
    return {
      question: this.question,
      type: this.type,
      sourceChunkId: this.sourceChunkId,
      confidenceScore: this.confidenceScore
    };
  }
}

// ============================================================================
// Pipeline Metrics
// ============================================================================

export class PipelineMetrics {
  constructor(options = {}) {
    this.chunksCreated = options.chunksCreated || 0;
    this.candidatesGenerated = options.candidatesGenerated || 0;
    this.candidatesValidated = options.candidatesValidated || 0;
    this.candidatesDeduplicated = options.candidatesDeduplicated || 0;
    this.latencyMs = options.latencyMs || 0.0;
    this.memoryPeakMb = options.memoryPeakMb || 0.0;
    
    // Optional detailed metrics
    this.validationPassRate = options.validationPassRate || 0.0;
    this.deduplicationReduction = options.deduplicationReduction || 0.0;
    this.chunkProcessingTimes = options.chunkProcessingTimes || [];
  }

  toDict() {
    return {
      chunksCreated: this.chunksCreated,
      candidatesGenerated: this.candidatesGenerated,
      candidatesValidated: this.candidatesValidated,
      candidatesDeduplicated: this.candidatesDeduplicated,
      latencyMs: this.latencyMs,
      memoryPeakMb: this.memoryPeakMb,
      validationPassRate: this.validationPassRate,
      deduplicationReduction: this.deduplicationReduction
    };
  }
}

export class PipelineResult {
  constructor(questions, metrics) {
    this.questions = questions;
    this.metrics = metrics;
  }

  toDict() {
    return {
      questions: this.questions.map(q => q.toDict()),
      metrics: this.metrics.toDict()
    };
  }
}