/**
 * Edge Question Generation Pipeline - Orchestration
 * End-to-end pipeline with parallel processing support
 */

import {
  PipelineMetrics,
  PipelineResult,
  PipelineStage
} from './pipeline_types.js';
import { PipelineError, ValidationError } from './errors.js';
import { chunkArticle, allocateCandidates } from './pre_generation.js';
import {
  validateCandidates,
  deduplicateCandidates,
  rankCandidates,
  selectFinalQuestions
} from './post_generation.js';

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

export class QuestionGenerationPipeline {
  constructor(adapter, config = null, options = {}) {
    this.adapter = adapter;
    this.config = config;
    this.parallelProcessing = options.parallelProcessing || false;
    this.maxConcurrentChunks = options.maxConcurrentChunks || 3;
  }
  
  async generate(article) {
    const result = await this.generateWithMetrics(article);
    return result.questions;
  }
  
  async generateWithMetrics(article) {
    const startTime = performance.now();
    
    try {
      // ================================================================
      // Stage 1: Pre-Generation
      // ================================================================
      const chunks = this._preGeneration(article);
      
      // ================================================================
      // Stage 2: Generation
      // ================================================================
      const { candidates, chunkTimes } = await this._generation(
        chunks,
        article.desiredQuestions
      );
      
      // ================================================================
      // Stage 3: Post-Generation
      // ================================================================
      const questions = this._postGeneration(
        candidates,
        chunks,
        article.desiredQuestions
      );
      
      // ================================================================
      // Metrics Collection
      // ================================================================
      const endTime = performance.now();
      
      // Calculate metrics
      const totalCandidates = candidates.length;
      
      // Validation metrics
      const validationResult = validateCandidates(candidates, chunks);
      const validatedCount = validationResult.valid.length;
      
      // Deduplication metrics
      const deduplicated = deduplicateCandidates(validationResult.valid);
      const deduplicatedCount = deduplicated.length;
      
      const metrics = new PipelineMetrics({
        chunksCreated: chunks.length,
        candidatesGenerated: totalCandidates,
        candidatesValidated: validatedCount,
        candidatesDeduplicated: deduplicatedCount,
        latencyMs: endTime - startTime,
        memoryPeakMb: 0, // Not easily available in browser
        validationPassRate: validationResult.passRate,
        deduplicationReduction: validatedCount > 0
          ? (validatedCount - deduplicatedCount) / validatedCount
          : 0.0,
        chunkProcessingTimes: chunkTimes
      });
      
      return new PipelineResult(questions, metrics);
      
    } catch (e) {
      // Re-raise PipelineError as-is
      if (e instanceof PipelineError) {
        throw e;
      }
      
      // Wrap other exceptions
      throw new PipelineError(
        `Unexpected error in pipeline: ${e.message}`,
        PipelineStage.POST_GEN,
        false
      );
    }
  }
  
  _preGeneration(article) {
    try {
      const chunks = chunkArticle(article);
      
      console.log(`\n📦 Chunking complete:`);
      console.log(`   • Desired questions: ${article.desiredQuestions}`);
      console.log(`   • Chunks created: ${chunks.length}`);
      
      return chunks;
    } catch (e) {
      throw new PipelineError(
        `Pre-generation failed: ${e.message}`,
        PipelineStage.PRE_GEN,
        false
      );
    }
  }
  
  async _generation(chunks, desiredQuestions) {
    // Calculate candidate allocation (always 1 per chunk)
    const allocations = allocateCandidates(chunks.length, desiredQuestions);
    
    // Check model readiness
    if (!(await this.adapter.isReady())) {
      throw new PipelineError(
        "Model adapter is not ready",
        PipelineStage.GENERATION,
        true
      );
    }
    
    // Choose processing mode
    let allCandidates, chunkTimes, errors;
    
    if (this.parallelProcessing) {
      console.log(`\n⚡ Parallel processing (max ${this.maxConcurrentChunks} concurrent)`);
      ({ allCandidates, chunkTimes, errors } = await this._processChunksParallel(
        chunks,
        allocations
      ));
    } else {
      console.log(`\n🔄 Sequential processing`);
      ({ allCandidates, chunkTimes, errors } = await this._processChunksSequential(
        chunks,
        allocations
      ));
    }
    
    // Check if we got any candidates
    if (allCandidates.length === 0) {
      throw new PipelineError(
        `Generation failed for all chunks. Errors: ${errors.map(e => e.message).join(', ')}`,
        PipelineStage.GENERATION,
        false
      );
    }
    
    console.log(`   • Generated ${allCandidates.length} candidate questions`);
    if (errors.length > 0) {
      console.log(`   ⚠️  ${errors.length} chunks failed`);
    }
    
    return { candidates: allCandidates, chunkTimes };
  }
  
  async _processChunksSequential(chunks, allocations) {
    const allCandidates = [];
    const chunkTimes = [];
    const errors = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const numCandidates = allocations[i];
      const chunkStart = performance.now();
      
      try {
        const candidates = await this.adapter.generateQuestions(
          chunk,
          numCandidates,
          this.config
        );
        allCandidates.push(...candidates);
      } catch (e) {
        console.error(`Chunk ${chunk.chunkId} failed:`, e);
        errors.push(e);
      } finally {
        const chunkEnd = performance.now();
        chunkTimes.push(chunkEnd - chunkStart);
      }
    }
    
    return { allCandidates, chunkTimes, errors };
  }
  
  async _processChunksParallel(chunks, allocations) {
    const processChunkWithLimit = async (chunk, numCandidates, chunkIdx) => {
      const chunkStart = performance.now();
      
      try {
        const candidates = await this.adapter.generateQuestions(
          chunk,
          numCandidates,
          this.config
        );
        const chunkEnd = performance.now();
        const elapsedMs = chunkEnd - chunkStart;
        return { chunkIdx, candidates, elapsedMs, error: null };
      } catch (e) {
        const chunkEnd = performance.now();
        const elapsedMs = chunkEnd - chunkStart;
        return { chunkIdx, candidates: [], elapsedMs, error: e };
      }
    };
    
    // Create tasks for all chunks
    const tasks = chunks.map((chunk, idx) => 
      processChunkWithLimit(chunk, allocations[idx], idx)
    );
    
    // Execute in parallel (browser will handle concurrency)
    const results = await Promise.all(tasks);
    
    // Sort results by chunk index to maintain order
    results.sort((a, b) => a.chunkIdx - b.chunkIdx);
    
    // Extract candidates, times, and errors
    const allCandidates = [];
    const chunkTimes = [];
    const errors = [];
    
    for (const result of results) {
      chunkTimes.push(result.elapsedMs);
      
      if (result.error) {
        errors.push(result.error);
      } else {
        allCandidates.push(...result.candidates);
      }
    }
    
    return { allCandidates, chunkTimes, errors };
  }
  
  _postGeneration(candidates, chunks, k) {
    try {
      console.log(`\n🔍 Post-processing:`);
      console.log(`   • Candidates generated: ${candidates.length}`);
      console.log(`   • Requested questions: ${k}`);
      
      // Minimal validation (no evidence checking)
      const validationResult = validateCandidates(candidates, chunks);
      
      if (validationResult.valid.length === 0) {
        throw new ValidationError(
          "All candidates failed validation",
          validationResult.rejected.length,
          0
        );
      }
      
      console.log(`   • Valid candidates: ${validationResult.valid.length}`);
      
      // No deduplication
      const deduplicated = deduplicateCandidates(validationResult.valid);
      
      // Neutral ranking
      const ranked = rankCandidates(deduplicated, chunks);
      
      // Select top K
      const finalQuestions = selectFinalQuestions(ranked, k);
      
      console.log(`   • Final questions: ${finalQuestions.length}`);
      
      if (finalQuestions.length < k) {
        console.log(`   ⚠️  Got ${finalQuestions.length} questions, wanted ${k}`);
      }
      
      return finalQuestions;
      
    } catch (e) {
      if (e instanceof ValidationError) {
        throw e;
      }
      throw new PipelineError(
        `Post-generation failed: ${e.message}`,
        PipelineStage.POST_GEN,
        false
      );
    }
  }
}