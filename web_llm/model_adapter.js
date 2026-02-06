/**
 * Edge Question Generation Pipeline - Model Adapter Interface
 * Optimized for instruction-tuned models with WebGPU support
 */

import { QuestionCandidate, QuestionType, QuestionStyle, InferenceErrorCause } from './pipeline_types.js';
import { ModelInferenceError } from './errors.js';

// ============================================================================
// Prompt Template
// ============================================================================

export function buildPrompt(chunkText, numCandidates = 1, config = null) {
  return [
    {
      role: "system",
      content: "You are a strict JSON generator. Output ONLY one JSON object and nothing else. {\"question\":\"Insert your question here...\"}"
    },
    {
      role: "user",
      content: `Write ONE question from the given text.
        Text:
        ${chunkText}

        Return JSON now like this - {"question":"Insert your question here..."}`
    }
  ];
}

// ============================================================================
// Response Parsing
// ============================================================================

function extractFirstJSONObject(s) {
  if (!s) return null;
  
  const start = s.indexOf("{");
  if (start === -1) return null;
  
  let depth = 0;
  let inStr = false;
  let esc = false;
  
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    
    if (inStr) {
      if (esc) {
        esc = false;
        continue;
      }
      if (c === "\\") {
        esc = true;
        continue;
      }
      if (c === '"') {
        inStr = false;
      }
      continue;
    }
    
    if (c === '"') {
      inStr = true;
      continue;
    }
    
    if (c === "{") {
      depth++;
      continue;
    }
    
    if (c === "}") {
      depth--;
      if (depth === 0) {
        return s.substring(start, i + 1);
      }
      if (depth < 0) {
        return null;
      }
    }
  }
  
  return null;
}

export function parseModelResponse(responseText, chunkId) {
  const originalText = responseText;
  let cleaned = (responseText || "").trim();
  
  // Remove markdown code fences
  cleaned = cleaned.replace(/^\s*```(?:json)?\s*/i, "");
  cleaned = cleaned.replace(/\s*```\s*$/, "");
  cleaned = cleaned.trim();
  
  let data = null;
  
  // Strategy 1: direct JSON parse
  try {
    console.log("*****************************");
    console.log(`Original response text: ${originalText}`);
    console.log(`Cleaned response text: ${cleaned}`);
    console.log("*****************************");
    
    const data0 = JSON.parse(cleaned);
    if (typeof data0 === 'object' && data0 !== null && !Array.isArray(data0)) {
      data = data0;
    }
  } catch (e) {
    // Continue to next strategy
  }
  
  // Strategy 2: balanced-brace extraction
  if (data === null) {
    const obj = extractFirstJSONObject(cleaned);
    if (obj) {
      try {
        const data0 = JSON.parse(obj);
        if (typeof data0 === 'object' && data0 !== null && !Array.isArray(data0)) {
          data = data0;
        }
      } catch (e) {
        // Continue to next strategy
      }
    }
  }
  
  // Strategy 3: regex extraction
  if (data === null) {
    const jsonMatch = cleaned.match(/\{[\s\S]*?"(?:question|questions)"[\s\S]*?\}/i);
    if (jsonMatch) {
      try {
        const data0 = JSON.parse(jsonMatch[0]);
        if (typeof data0 === 'object' && data0 !== null && !Array.isArray(data0)) {
          data = data0;
        }
      } catch (e) {
        // Continue to next strategy
      }
    }
  }
  
  // Strategy 4: array-only output
  if (data === null) {
    const arrayMatch = cleaned.match(/\[[\s\S]*?\]/);
    if (arrayMatch) {
      try {
        const arr = JSON.parse(arrayMatch[0]);
        if (Array.isArray(arr)) {
          data = { questions: arr };
        }
      } catch (e) {
        // Continue to next strategy
      }
    }
  }
  
  // Strategy 5: question-like text
  if (data === null) {
    const questionPatterns = [
      /Question:\s*(.+?)(?:\n|$)/,
      /Q:\s*(.+?)(?:\n|$)/,
      /(?:^|\n)([A-Z][^.!?]*\?)/
    ];
    
    for (const pattern of questionPatterns) {
      const match = cleaned.match(pattern);
      if (match) {
        const q = (match[1] || "").trim();
        if (q) {
          data = { question: q };
          break;
        }
      }
    }
  }
  
  if (data === null) {
    throw new ModelInferenceError(
      `Failed to parse model response as JSON.\nOriginal: ${originalText.substring(0, 500)}`,
      InferenceErrorCause.PARSE_ERROR,
      chunkId
    );
  }
  
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new ModelInferenceError(
      `Model response is not a JSON object: ${typeof data}`,
      InferenceErrorCause.PARSE_ERROR,
      chunkId
    );
  }
  
  // Normalize supported schemas into a questions list
  let questions = [];
  
  if ("questions" in data) {
    const qv = data.questions;
    if (Array.isArray(qv)) {
      questions = qv;
    } else if (typeof qv === 'string') {
      questions = [qv];
    } else {
      throw new ModelInferenceError(
        `'questions' field is not a list or string: ${typeof qv}`,
        InferenceErrorCause.PARSE_ERROR,
        chunkId
      );
    }
  } else if ("question" in data) {
    const qv = data.question;
    if (typeof qv === 'string') {
      questions = [qv];
    } else if (Array.isArray(qv)) {
      questions = qv;
    } else {
      throw new ModelInferenceError(
        `'question' field is not a string or list: ${typeof qv}`,
        InferenceErrorCause.PARSE_ERROR,
        chunkId
      );
    }
  } else {
    throw new ModelInferenceError(
      `Model response missing 'question'/'questions'. Keys: ${Object.keys(data).join(', ')}`,
      InferenceErrorCause.PARSE_ERROR,
      chunkId
    );
  }
  
  // Build candidates
  const candidates = [];
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    try {
      let questionText;
      
      if (typeof q === 'string') {
        questionText = q.trim();
      } else if (typeof q === 'object' && q !== null) {
        questionText = (q.question || q.text || q.query || "").trim();
      } else {
        continue;
      }
      
      if (!questionText) continue;
      
      // Normalize: ensure it ends with '?'
      if (!questionText.endsWith("?")) {
        questionText = questionText.trimEnd() + "?";
      }
      
      candidates.push(
        new QuestionCandidate(
          questionText,
          QuestionType.PROCEDURAL,
          "",
          chunkId,
          QuestionStyle.INTERROGATIVE
        )
      );
    } catch (e) {
      console.warn(`Skipped malformed question ${i}:`, e);
      continue;
    }
  }
  
  if (candidates.length === 0) {
    throw new ModelInferenceError(
      `No valid questions parsed from response.\nResponse: ${originalText.substring(0, 500)}`,
      InferenceErrorCause.PARSE_ERROR,
      chunkId
    );
  }
  
  // Enforce "always 1" question contract
  return candidates.slice(0, 1);
}

// ============================================================================
// Model Adapter Interface
// ============================================================================

export class ModelAdapter {
  async generateQuestions(chunk, numCandidates, generationConfig) {
    throw new Error("generateQuestions() must be implemented by subclass");
  }
  
  async isReady() {
    throw new Error("isReady() must be implemented by subclass");
  }
  
  estimateMemoryUsage(chunk) {
    throw new Error("estimateMemoryUsage() must be implemented by subclass");
  }
}

// ============================================================================
// WebGPU Adapter (Transformers.js)
// ============================================================================

export class WebGPUAdapter extends ModelAdapter {
  constructor(generatorPipeline) {
    super();
    this.generator = generatorPipeline;
    this._ready = false;
  }
  
  async initialize() {
    // Generator should already be loaded
    if (this.generator) {
      this._ready = true;
    }
    return this._ready;
  }
  
  async generateQuestions(chunk, numCandidates, generationConfig) {
    if (!this._ready) {
      throw new ModelInferenceError(
        "Model is not ready",
        InferenceErrorCause.MODEL_NOT_READY,
        chunk.chunkId
      );
    }
    
    try {
      // Build prompt
      const messages = buildPrompt(chunk.text, numCandidates, generationConfig);
      
      console.log(`Generating for chunk ${chunk.chunkId}...`);
      
      // Call model
      const result = await this.generator(messages, {
        max_new_tokens: generationConfig.maxOutputTokens,
        temperature: generationConfig.temperature,
        top_p: generationConfig.topP,
        do_sample: true
      });
      
      // Extract response text
      let responseText;
      if (result && result[0] && result[0].generated_text) {
        const messages = result[0].generated_text;
        const lastMessage = messages[messages.length - 1];
        responseText = lastMessage.content;
      } else {
        responseText = JSON.stringify(result);
      }
      
      // Parse response
      const candidates = parseModelResponse(responseText, chunk.chunkId);
      
      return candidates;
      
    } catch (e) {
      if (e instanceof ModelInferenceError) {
        throw e;
      }
      
      throw new ModelInferenceError(
        `Generation failed: ${e.message}`,
        InferenceErrorCause.UNKNOWN,
        chunk.chunkId
      );
    }
  }
  
  async isReady() {
    return this._ready;
  }
  
  estimateMemoryUsage(chunk) {
    // Rough estimate for WebGPU models
    return 100 * 1024 * 1024; // 100 MB
  }
}

// ============================================================================
// Mock Adapter (for testing)
// ============================================================================

export class MockAdapter extends ModelAdapter {
  constructor(latencyMs = 100) {
    super();
    this.latencyMs = latencyMs;
    this._ready = true;
  }
  
  async generateQuestions(chunk, numCandidates, generationConfig) {
    await new Promise(resolve => setTimeout(resolve, this.latencyMs));
    
    const templates = [
      "How do I configure this feature?",
      "What is the purpose of this functionality?",
      "When should I use this approach?",
      "Which option is best for my use case?"
    ];
    
    const question = templates[chunk.chunkId % templates.length];
    
    return [new QuestionCandidate(
      question,
      QuestionType.PROCEDURAL,
      "",
      chunk.chunkId,
      QuestionStyle.INTERROGATIVE
    )];
  }
  
  async isReady() {
    return this._ready;
  }
  
  estimateMemoryUsage(chunk) {
    return 100 * 1024 * 1024;
  }
}