"""
Gemma Model Adapter for Edge Question Generation Pipeline
Uses llama.cpp for efficient GGUF quantized model inference
"""

import os
import asyncio
import subprocess
import json
from pathlib import Path
from typing import List, Optional
import time

from pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from errors import ModelInferenceError, InferenceErrorCause
from model_adapter import ModelAdapter, build_prompt, parse_model_response


class GemmaAdapter(ModelAdapter):
    """
    Adapter for Gemma using llama.cpp (GGUF quantized models).
    
    Supports both CPU-only and GPU-accelerated inference on edge devices.
    Compatible with Gemma 2B and 7B quantized models (Q4_K_M, Q8_0, etc.)
    """
    
    def __init__(
        self,
        model_path: str,
        llama_cpp_path: str = "./llama.cpp",
        max_new_tokens: int = 512,
        timeout_seconds: float = 30.0,
        n_threads: int = 4,
        use_gpu: bool = True,
        context_size: int = 2048,
        n_gpu_layers: int = 99  # Offload all layers to GPU if available
    ):
        """
        Initialize Gemma adapter.
        
        Args:
            model_path: Path to GGUF model file
            llama_cpp_path: Path to llama.cpp directory
            max_new_tokens: Maximum tokens to generate per request
            timeout_seconds: Timeout for inference operations
            n_threads: Number of CPU threads to use
            use_gpu: Enable GPU acceleration (Metal on macOS, CUDA on Linux)
            context_size: Context window size
            n_gpu_layers: Number of layers to offload to GPU (99 = all)
        """
        self.model_path = Path(model_path)
        self.llama_cpp_path = Path(llama_cpp_path)
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.n_threads = n_threads
        self.use_gpu = use_gpu
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        
        self._llama_cli = None
        self._ready = False
        self._model_size_mb = 0
    
    def _find_llama_cli(self) -> Path:
        """
        Find llama-cli binary in llama.cpp directory.
        
        Returns:
            Path to llama-cli executable
            
        Raises:
            FileNotFoundError: If llama-cli binary not found
        """
        possible_binaries = [
            self.llama_cpp_path / "build" / "bin" / "llama-cli",  # CMake build (preferred)
            self.llama_cpp_path / "llama-cli",                     # Source build
            self.llama_cpp_path / "main",                          # Legacy name
            self.llama_cpp_path / "bin" / "llama-cli",             # Alternative location
            self.llama_cpp_path / "libexec" / "llama-cli",         # Homebrew
        ]
        
        for binary in possible_binaries:
            if binary.exists() and os.access(binary, os.X_OK):
                return binary
        
        # Provide helpful error message
        raise FileNotFoundError(
            f"llama-cli binary not found in {self.llama_cpp_path}. "
            f"Searched locations:\n" + 
            "\n".join(f"  - {b}" for b in possible_binaries) +
            f"\n\nMake sure llama.cpp is compiled. Run:\n"
            f"  cd {self.llama_cpp_path}\n"
            f"  mkdir build && cd build\n"
            f"  cmake .. -DGGML_METAL=ON  # For macOS\n"
            f"  cmake --build . --config Release -j"
        )
    
    async def initialize(self) -> None:
        """
        Initialize the adapter and validate model.
        
        Raises:
            ModelInferenceError: If model not found or initialization fails
        """
        # Find llama-cli binary
        try:
            self._llama_cli = self._find_llama_cli()
        except FileNotFoundError as e:
            raise ModelInferenceError(
                message=str(e),
                cause=InferenceErrorCause.MODEL_LOAD_ERROR
            )
        
        # Validate model exists
        if not self.model_path.exists():
            raise ModelInferenceError(
                message=f"Model not found: {self.model_path}",
                cause=InferenceErrorCause.MODEL_LOAD_ERROR
            )
        
        # Get model size
        self._model_size_mb = self.model_path.stat().st_size / (1024 ** 2)
        
        # Test inference to ensure model loads correctly
        try:
            test_prompt = "Hello"
            cmd = [
                str(self._llama_cli),
                "-m", str(self.model_path),
                "-p", test_prompt,
                "-n", "1",  # Generate just 1 token
                "-t", str(self.n_threads),
                "--log-disable",
                "--no-display-prompt",  # Don't show the prompt
                "-e",  # Exit after processing
                "--simple-io",  # Simple input/output mode
            ]
            
            if self.use_gpu:
                cmd.extend(["-ngl", str(self.n_gpu_layers)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout for initialization
            )
            
            if result.returncode != 0:
                raise ModelInferenceError(
                    message=f"Model failed to load: {result.stderr}",
                    cause=InferenceErrorCause.MODEL_LOAD_ERROR
                )
            
            self._ready = True
            
        except subprocess.TimeoutExpired:
            raise ModelInferenceError(
                message="Model initialization timeout",
                cause=InferenceErrorCause.TIMEOUT
            )
        except Exception as e:
            raise ModelInferenceError(
                message=f"Model initialization failed: {str(e)}",
                cause=InferenceErrorCause.MODEL_LOAD_ERROR
            )
    
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        """
        Generate questions from a chunk using llama-cli subprocess.
        
        Args:
            chunk: Text chunk to process
            num_candidates: Number of questions to generate
            generation_config: Generation parameters
            
        Returns:
            List of parsed question candidates
            
        Raises:
            ModelInferenceError: On timeout, generation failure, or parse error
        """
        if not self._ready:
            raise ModelInferenceError(
                message="Adapter not initialized. Call initialize() first.",
                cause=InferenceErrorCause.MODEL_NOT_READY,
                chunk_id=chunk.chunk_id
            )
        
        # Build prompt with config
        prompt = build_prompt(chunk.text, num_candidates, generation_config)
        
        # Build command
        cmd = [
            str(self._llama_cli),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(min(self.max_new_tokens, generation_config.max_output_tokens)),
            "--temp", str(generation_config.temperature),
            "--top-p", str(generation_config.top_p),
            "-c", str(self.context_size),
            "-t", str(self.n_threads),
            "--log-disable",
            "--no-display-prompt",  # CRITICAL: Don't show the prompt
            "-e",  # Exit after processing (no interactive mode)
            "--simple-io",  # Simple input/output mode
        ]
        
        # Add GPU offloading if enabled
        if self.use_gpu:
            cmd.extend(["-ngl", str(self.n_gpu_layers)])
        
        # Add stop sequences if provided
        for stop_seq in generation_config.stop_sequences:
            cmd.extend(["--reverse-prompt", stop_seq])
        
        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            
            start_time = time.time()
            
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds
                    )
                ),
                timeout=self.timeout_seconds + 5  # Extra 5s for executor overhead
            )
            
            elapsed_time = time.time() - start_time
            
            # Check for subprocess errors
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                
                # Detect specific error types
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    cause = InferenceErrorCause.OOM
                elif "model" in error_msg.lower() and "load" in error_msg.lower():
                    cause = InferenceErrorCause.MODEL_LOAD_ERROR
                else:
                    cause = InferenceErrorCause.UNKNOWN
                
                raise ModelInferenceError(
                    message=f"llama-cli failed: {error_msg}",
                    cause=cause,
                    chunk_id=chunk.chunk_id
                )
            
            # Extract generated text
            response_text = result.stdout.strip()
            
            # llama-cli may echo the prompt, extract just the generated part
            # With --no-display-prompt and --simple-io, output should be cleaner
            # Look for JSON response (it should start with { or be after the prompt)
            if "{" in response_text:
                # Find the first { and take everything from there
                json_start = response_text.find("{")
                response_text = response_text[json_start:]
            
            # Parse the response
            try:
                candidates = parse_model_response(response_text, chunk.chunk_id)
            except ModelInferenceError as e:
                # Add timing info to parse errors
                e.message = f"{e.message} (generation took {elapsed_time:.2f}s)"
                raise
            
            # Return requested number of candidates
            return candidates[:num_candidates]
            
        except asyncio.TimeoutError:
            raise ModelInferenceError(
                message=f"Inference timeout after {self.timeout_seconds}s",
                cause=InferenceErrorCause.TIMEOUT,
                chunk_id=chunk.chunk_id
            )
        except subprocess.TimeoutExpired:
            raise ModelInferenceError(
                message=f"Subprocess timeout after {self.timeout_seconds}s",
                cause=InferenceErrorCause.TIMEOUT,
                chunk_id=chunk.chunk_id
            )
        except ModelInferenceError:
            # Re-raise our own errors
            raise
        except Exception as e:
            raise ModelInferenceError(
                message=f"Unexpected error during generation: {str(e)}",
                cause=InferenceErrorCause.UNKNOWN,
                chunk_id=chunk.chunk_id
            )
    
    async def is_ready(self) -> bool:
        """
        Check if model is loaded and ready for inference.
        
        Returns:
            True if ready, False otherwise
        """
        return self._ready
    
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        """
        Estimate memory footprint for this model + chunk.
        
        For GGUF models, memory usage is primarily:
        - Model size (depends on quantization)
        - Context buffer (depends on context_size)
        - KV cache (depends on context_size and n_ctx)
        
        Args:
            chunk: Chunk to estimate for
            
        Returns:
            Estimated memory usage in bytes
        """
        # Base model size
        model_memory = int(self._model_size_mb * 1024 * 1024)
        
        # Context buffer (rough estimate: 1.5 bytes per token per context position)
        context_memory = int(self.context_size * 1.5)
        
        # KV cache (depends on model architecture, rough estimate)
        kv_cache_memory = int(self.context_size * 512)  # Rough estimate
        
        # Add 20% overhead for miscellaneous allocations
        total_memory = int((model_memory + context_memory + kv_cache_memory) * 1.2)
        
        return total_memory
    
    def get_info(self) -> dict:
        """
        Get adapter configuration and status information.
        
        Returns:
            Dictionary with adapter details
        """
        return {
            "adapter_type": "GemmaAdapter",
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "model_size_mb": self._model_size_mb,
            "llama_cpp_path": str(self.llama_cpp_path),
            "ready": self._ready,
            "config": {
                "max_new_tokens": self.max_new_tokens,
                "timeout_seconds": self.timeout_seconds,
                "n_threads": self.n_threads,
                "use_gpu": self.use_gpu,
                "context_size": self.context_size,
                "n_gpu_layers": self.n_gpu_layers,
            }
        }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'GemmaAdapter',
]