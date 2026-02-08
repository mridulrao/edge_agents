/**
 * Edge Question Generation Pipeline - Memory Profiler
 * Tracks WebGPU memory usage, load times, and inference overhead
 */

// ============================================================================
// Memory Snapshot
// ============================================================================

export class MemorySnapshot {
  constructor(timestamp = Date.now()) {
    this.timestamp = timestamp;
    this.jsHeapSize = 0;
    this.jsHeapSizeLimit = 0;
    this.jsHeapUsedSize = 0;
    this.gpuMemoryEstimate = 0;
    this.totalMemoryEstimate = 0;
  }

  static async capture() {
    const snapshot = new MemorySnapshot();
    
    // JavaScript heap memory (Chrome/Edge)
    if (performance.memory) {
      snapshot.jsHeapSize = performance.memory.totalJSHeapSize;
      snapshot.jsHeapSizeLimit = performance.memory.jsHeapSizeLimit;
      snapshot.jsHeapUsedSize = performance.memory.usedJSHeapSize;
    }
    
    // Try to estimate GPU memory if available
    // Note: WebGPU doesn't expose actual GPU memory usage in browsers
    // This is a known limitation - we can only get adapter info, not actual memory usage
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter && adapter.info) {
          // Use the modern synchronous .info property (not requestAdapterInfo())
          const info = adapter.info;
          snapshot.gpuMemoryEstimate = snapshot._estimateGPUMemory(info);
        }
      }
    } catch (e) {
      // Silently ignore - GPU memory info not available
      // This is expected in most browser implementations
    }
    
    snapshot.totalMemoryEstimate = snapshot.jsHeapUsedSize + snapshot.gpuMemoryEstimate;
    
    return snapshot;
  }

  _estimateGPUMemory(adapterInfo) {
    // WebGPU doesn't expose actual memory usage in browsers
    // But we can log adapter info for debugging purposes
    if (adapterInfo) {
      // Log adapter details (useful for debugging/optimization)
      console.log('GPU Adapter Info:', {
        vendor: adapterInfo.vendor || 'unknown',
        architecture: adapterInfo.architecture || 'unknown',
        device: adapterInfo.device || 'unknown',
        description: adapterInfo.description || 'unknown'
      });
    }
    
    // Return 0 - actual GPU memory usage is not accessible via WebGPU API
    // Use browser DevTools → Performance → Memory for actual GPU memory
    return 0;
  }

  toMB(bytes) {
    return bytes / (1024 * 1024);
  }

  toDict() {
    return {
      timestamp: this.timestamp,
      jsHeapSizeMB: this.toMB(this.jsHeapSize),
      jsHeapSizeLimitMB: this.toMB(this.jsHeapSizeLimit),
      jsHeapUsedSizeMB: this.toMB(this.jsHeapUsedSize),
      gpuMemoryEstimateMB: this.toMB(this.gpuMemoryEstimate),
      totalMemoryEstimateMB: this.toMB(this.totalMemoryEstimate)
    };
  }
}

// ============================================================================
// Model Load Metrics
// ============================================================================

export class ModelLoadMetrics {
  constructor() {
    this.startTime = 0;
    this.endTime = 0;
    this.loadDurationMs = 0;
    
    this.memoryBefore = null;
    this.memoryAfter = null;
    this.memoryDeltaMB = 0;
    
    this.modelPath = '';
    this.modelDtype = '';
    this.modelSizeMB = 0; // If we can estimate from files
    
    this.deviceType = 'webgpu';
    this.success = false;
    this.error = null;
  }

  async startTracking(modelPath, dtype = 'q8') {
    this.modelPath = modelPath;
    this.modelDtype = dtype;
    this.startTime = performance.now();
    this.memoryBefore = await MemorySnapshot.capture();
  }

  async endTracking(success = true, error = null) {
    this.endTime = performance.now();
    this.loadDurationMs = this.endTime - this.startTime;
    this.memoryAfter = await MemorySnapshot.capture();
    
    this.memoryDeltaMB = 
      this.memoryAfter.toMB(this.memoryAfter.jsHeapUsedSize) -
      this.memoryBefore.toMB(this.memoryBefore.jsHeapUsedSize);
    
    this.success = success;
    this.error = error;
  }

  async estimateModelSize() {
    try {
      // Transformers.js can use different directory structures:
      // - /onnx_adaption_web/{model}/onnx/model.onnx (standard)
      // - /onnx_adaption_web/{model}/model.onnx (flat)
      
      const possiblePaths = [
        // Standard structure (with onnx subdirectory)
        [`/onnx_adaption_web/${this.modelPath}/onnx/model.onnx`, 'model.onnx'],
      ];
      
      let totalSize = 0;
      const foundFiles = [];
      
      for (const [path, name] of possiblePaths) {
        try {
          const response = await fetch(path, {
            method: 'HEAD'
          });
          
          if (response.ok) {
            const contentLength = response.headers.get('content-length');
            if (contentLength) {
              const sizeMB = parseInt(contentLength) / (1024 * 1024);
              totalSize += parseInt(contentLength);
              foundFiles.push(`${name}: ${sizeMB.toFixed(2)} MB`);
              console.log(`Found model file: ${path} (${sizeMB.toFixed(2)} MB)`);
            }
          }
        } catch (e) {
          // File doesn't exist or HEAD not allowed, continue
        }
      }
      
      this.modelSizeMB = totalSize / (1024 * 1024);
      
      if (foundFiles.length > 0) {
        console.log(`Total model size: ${this.modelSizeMB.toFixed(2)} MB from ${foundFiles.length} file(s)`);
        console.log('Files found:', foundFiles.join(', '));
      } else {
        console.log('Could not determine model size (no files found or HEAD requests blocked)');
      }
    } catch (e) {
      console.warn('Could not estimate model size:', e);
    }
  }

  toDict() {
    return {
      modelPath: this.modelPath,
      modelDtype: this.modelDtype,
      modelSizeMB: this.modelSizeMB,
      loadDurationMs: this.loadDurationMs,
      loadDurationSec: this.loadDurationMs / 1000,
      memoryDeltaMB: this.memoryDeltaMB,
      memoryBefore: this.memoryBefore ? this.memoryBefore.toDict() : null,
      memoryAfter: this.memoryAfter ? this.memoryAfter.toDict() : null,
      deviceType: this.deviceType,
      success: this.success,
      error: this.error ? this.error.message : null
    };
  }
}

// ============================================================================
// Inference Metrics
// ============================================================================

export class InferenceMetrics {
  constructor(chunkId = null) {
    this.chunkId = chunkId;
    this.startTime = 0;
    this.endTime = 0;
    this.inferenceDurationMs = 0;
    
    this.memoryBefore = null;
    this.memoryAfter = null;
    this.memoryDeltaMB = 0;
    this.peakMemoryMB = 0;
    
    this.inputTokens = 0;
    this.outputTokens = 0;
    this.tokensPerSecond = 0;
    
    this.success = false;
    this.error = null;
  }

  async startTracking(inputTokens = 0) {
    this.inputTokens = inputTokens;
    this.startTime = performance.now();
    this.memoryBefore = await MemorySnapshot.capture();
  }

  async endTracking(outputTokens = 0, success = true, error = null) {
    this.endTime = performance.now();
    this.inferenceDurationMs = this.endTime - this.startTime;
    this.memoryAfter = await MemorySnapshot.capture();
    
    this.outputTokens = outputTokens;
    this.tokensPerSecond = outputTokens > 0 
      ? (outputTokens / (this.inferenceDurationMs / 1000))
      : 0;
    
    this.memoryDeltaMB = 
      this.memoryAfter.toMB(this.memoryAfter.jsHeapUsedSize) -
      this.memoryBefore.toMB(this.memoryBefore.jsHeapUsedSize);
    
    this.peakMemoryMB = Math.max(
      this.memoryBefore.toMB(this.memoryBefore.jsHeapUsedSize),
      this.memoryAfter.toMB(this.memoryAfter.jsHeapUsedSize)
    );
    
    this.success = success;
    this.error = error;
  }

  toDict() {
    return {
      chunkId: this.chunkId,
      inferenceDurationMs: this.inferenceDurationMs,
      inferenceDurationSec: this.inferenceDurationMs / 1000,
      memoryDeltaMB: this.memoryDeltaMB,
      peakMemoryMB: this.peakMemoryMB,
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      tokensPerSecond: this.tokensPerSecond,
      memoryBefore: this.memoryBefore ? this.memoryBefore.toDict() : null,
      memoryAfter: this.memoryAfter ? this.memoryAfter.toDict() : null,
      success: this.success,
      error: this.error ? this.error.message : null
    };
  }
}

// ============================================================================
// Pipeline Memory Profiler
// ============================================================================

export class PipelineMemoryProfiler {
  constructor() {
    this.modelLoadMetrics = null;
    this.inferenceMetrics = [];
    this.baselineMemory = null;
    this.peakMemoryMB = 0;
    this.totalInferenceTimeMs = 0;
    this.averageInferenceTimeMs = 0;
    this.averageMemoryDeltaMB = 0;
  }

  async captureBaseline() {
    this.baselineMemory = await MemorySnapshot.capture();
    console.log('📊 Baseline memory captured:', this.baselineMemory.toDict());
  }

  createModelLoadMetrics() {
    this.modelLoadMetrics = new ModelLoadMetrics();
    return this.modelLoadMetrics;
  }

  createInferenceMetrics(chunkId = null) {
    const metrics = new InferenceMetrics(chunkId);
    this.inferenceMetrics.push(metrics);
    return metrics;
  }

  computeAggregates() {
    if (this.inferenceMetrics.length === 0) {
      return;
    }

    // Total inference time
    this.totalInferenceTimeMs = this.inferenceMetrics.reduce(
      (sum, m) => sum + m.inferenceDurationMs,
      0
    );

    // Average inference time
    this.averageInferenceTimeMs = this.totalInferenceTimeMs / this.inferenceMetrics.length;

    // Average memory delta
    this.averageMemoryDeltaMB = this.inferenceMetrics.reduce(
      (sum, m) => sum + m.memoryDeltaMB,
      0
    ) / this.inferenceMetrics.length;

    // Peak memory across all snapshots
    const allSnapshots = [
      this.baselineMemory,
      this.modelLoadMetrics?.memoryBefore,
      this.modelLoadMetrics?.memoryAfter,
      ...this.inferenceMetrics.map(m => m.memoryBefore),
      ...this.inferenceMetrics.map(m => m.memoryAfter)
    ].filter(s => s !== null);

    this.peakMemoryMB = Math.max(
      ...allSnapshots.map(s => s.toMB(s.jsHeapUsedSize))
    );
  }

  generateReport() {
    this.computeAggregates();

    return {
      baseline: this.baselineMemory ? this.baselineMemory.toDict() : null,
      modelLoad: this.modelLoadMetrics ? this.modelLoadMetrics.toDict() : null,
      inference: {
        totalInferences: this.inferenceMetrics.length,
        totalTimeMs: this.totalInferenceTimeMs,
        totalTimeSec: this.totalInferenceTimeMs / 1000,
        averageTimeMs: this.averageInferenceTimeMs,
        averageMemoryDeltaMB: this.averageMemoryDeltaMB,
        peakMemoryMB: this.peakMemoryMB,
        perChunk: this.inferenceMetrics.map(m => m.toDict())
      },
      summary: {
        totalMemoryOverheadMB: this.modelLoadMetrics 
          ? this.modelLoadMetrics.memoryDeltaMB 
          : 0,
        peakMemoryMB: this.peakMemoryMB,
        averageInferenceLatencyMs: this.averageInferenceTimeMs,
        modelLoadTimeMs: this.modelLoadMetrics?.loadDurationMs || 0
      }
    };
  }

  printReport() {
    const report = this.generateReport();
    
    console.log('\n' + '='.repeat(70));
    console.log('📊 MEMORY PROFILER REPORT');
    console.log('='.repeat(70));
    
    if (report.baseline) {
      console.log('\n🔵 Baseline Memory:');
      console.log(`   • JS Heap Used: ${report.baseline.jsHeapUsedSizeMB.toFixed(2)} MB`);
      console.log(`   • JS Heap Limit: ${report.baseline.jsHeapSizeLimitMB.toFixed(2)} MB`);
    }
    
    if (report.modelLoad) {
      console.log('\n🔴 Model Load:');
      console.log(`   • Model: ${report.modelLoad.modelPath}`);
      console.log(`   • Dtype: ${report.modelLoad.modelDtype}`);
      console.log(`   • Load Time: ${report.modelLoad.loadDurationSec.toFixed(2)}s`);
      console.log(`   • Memory Delta: ${report.modelLoad.memoryDeltaMB.toFixed(2)} MB`);
      if (report.modelLoad.modelSizeMB > 0) {
        console.log(`   • Model Size: ${report.modelLoad.modelSizeMB.toFixed(2)} MB`);
      }
    }
    
    if (report.inference.totalInferences > 0) {
      console.log('\n🟢 Inference:');
      console.log(`   • Total Inferences: ${report.inference.totalInferences}`);
      console.log(`   • Total Time: ${report.inference.totalTimeSec.toFixed(2)}s`);
      console.log(`   • Average Time: ${report.inference.averageTimeMs.toFixed(2)} ms`);
      console.log(`   • Average Memory Delta: ${report.inference.averageMemoryDeltaMB.toFixed(2)} MB`);
      console.log(`   • Peak Memory: ${report.inference.peakMemoryMB.toFixed(2)} MB`);
    }
    
    console.log('\n📈 Summary:');
    console.log(`   • Total Memory Overhead: ${report.summary.totalMemoryOverheadMB.toFixed(2)} MB`);
    console.log(`   • Peak Memory Usage: ${report.summary.peakMemoryMB.toFixed(2)} MB`);
    console.log(`   • Model Load Time: ${(report.summary.modelLoadTimeMs / 1000).toFixed(2)}s`);
    console.log(`   • Avg Inference Latency: ${report.summary.averageInferenceLatencyMs.toFixed(2)} ms`);
    
    console.log('='.repeat(70) + '\n');
    
    return report;
  }

  exportJSON() {
    return JSON.stringify(this.generateReport(), null, 2);
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

export async function profileModelLoad(loadFn, modelPath, dtype) {
  const profiler = new PipelineMemoryProfiler();
  await profiler.captureBaseline();
  
  const loadMetrics = profiler.createModelLoadMetrics();
  await loadMetrics.startTracking(modelPath, dtype);
  
  let result = null;
  let error = null;
  
  try {
    result = await loadFn();
    await loadMetrics.endTracking(true, null);
  } catch (e) {
    error = e;
    await loadMetrics.endTracking(false, e);
    throw e;
  } finally {
    await loadMetrics.estimateModelSize();
  }
  
  return { result, profiler };
}

export async function profileInference(inferenceFn, chunkId, profiler) {
  const inferenceMetrics = profiler.createInferenceMetrics(chunkId);
  await inferenceMetrics.startTracking();
  
  let result = null;
  let error = null;
  
  try {
    result = await inferenceFn();
    await inferenceMetrics.endTracking(0, true, null);
  } catch (e) {
    error = e;
    await inferenceMetrics.endTracking(0, false, e);
    throw e;
  }
  
  return result;
}

// ============================================================================
// Browser Compatibility Checks
// ============================================================================

export function checkMemoryProfilingSupport() {
  const support = {
    performanceMemory: typeof performance !== 'undefined' && 'memory' in performance,
    webGPU: typeof navigator !== 'undefined' && 'gpu' in navigator,
    performanceTiming: typeof performance !== 'undefined' && 'now' in performance
  };
  
  return support;
}

export function printProfilingSupport() {
  const support = checkMemoryProfilingSupport();
  
  console.log('\n🔍 Memory Profiling Support:');
  console.log(`   • Performance Memory API: ${support.performanceMemory ? '✓' : '✗'}`);
  console.log(`   • WebGPU: ${support.webGPU ? '✓' : '✗'}`);
  console.log(`   • Performance Timing: ${support.performanceTiming ? '✓' : '✗'}`);
  
  if (!support.performanceMemory) {
    console.log('\n⚠️  Note: Performance Memory API not available.');
    console.log('   For Chrome/Edge, enable chrome://flags/#enable-precise-memory-info');
  }
  
  if (support.webGPU) {
    console.log('\nℹ️  GPU Memory: WebGPU does not expose GPU memory usage in browsers.');
    console.log('   Only JS heap memory will be tracked. Use browser DevTools for GPU stats.');
  }
  
  return support;
}