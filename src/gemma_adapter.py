"""
Gemma Model Adapter - Server-Based Implementation
Uses llama-server with OpenAI-compatible API for clean, production-ready inference
"""

import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Optional
import time

from pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from errors import ModelInferenceError, InferenceErrorCause
from model_adapter import ModelAdapter, build_prompt, parse_model_response


class GemmaServerAdapter(ModelAdapter):
    """
    Gemma adapter using llama-server API.
    
    This adapter connects to a running llama-server instance, providing:
    - Clean API-based inference (no subprocess management)
    - OpenAI-compatible endpoints
    - Support for concurrent requests
    - Streaming support (optional)
    
    Usage:
        1. Start llama-server separately:
           ./llama-server -m model.gguf -c 2048 --port 8080
        
        2. Create adapter:
           adapter = GemmaServerAdapter(server_url="http://localhost:8080")
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize adapter for llama-server.
        
        Args:
            server_url: URL of running llama-server (e.g., http://localhost:8080)
            timeout_seconds: Request timeout
            max_retries: Number of retries for failed requests
        """
        self.server_url = server_url.rstrip('/')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        self._ready = False
        self._model_info = None
    
    async def initialize(self) -> None:
        """
        Verify server is running and get model info.
        
        Raises:
            ModelInferenceError: If server is not accessible
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check health endpoint
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        raise ModelInferenceError(
                            message=f"Server health check failed: {response.status}",
                            cause=InferenceErrorCause.MODEL_NOT_READY
                        )
                
                # Get model info
                async with session.get(
                    f"{self.server_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._model_info = data.get('data', [{}])[0] if data.get('data') else {}
            
            self._ready = True
            
        except aiohttp.ClientError as e:
            raise ModelInferenceError(
                message=f"Cannot connect to llama-server at {self.server_url}: {str(e)}",
                cause=InferenceErrorCause.MODEL_NOT_READY
            )
        except asyncio.TimeoutError:
            raise ModelInferenceError(
                message=f"Timeout connecting to llama-server at {self.server_url}",
                cause=InferenceErrorCause.TIMEOUT
            )
    
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        """
        Generate questions via llama-server API.
        
        Uses the /v1/completions endpoint (OpenAI-compatible).
        
        Args:
            chunk: Text chunk to process
            num_candidates: Number of questions to generate
            generation_config: Generation parameters
            
        Returns:
            List of parsed question candidates
            
        Raises:
            ModelInferenceError: On API errors or parsing failures
        """
        if not self._ready:
            raise ModelInferenceError(
                message="Adapter not initialized. Call initialize() first.",
                cause=InferenceErrorCause.MODEL_NOT_READY,
                chunk_id=chunk.chunk_id
            )
        
        # Build prompt
        prompt = build_prompt(chunk.text, num_candidates, generation_config)
        
        # Prepare request payload
        payload = {
            "prompt": prompt,
            "n_predict": generation_config.max_output_tokens,  # llama-server uses n_predict
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "stop": generation_config.stop_sequences,
            "stream": False,  # Disable streaming for simpler parsing
        }
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_url}/completion",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                    ) as response:
                        elapsed_time = time.time() - start_time
                        
                        if response.status != 200:
                            error_text = await response.text()
                            raise ModelInferenceError(
                                message=f"Server returned {response.status}: {error_text}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk.chunk_id
                            )
                        
                        result = await response.json()
                
                # Extract generated text
                generated_text = result.get('content', '').strip()
                
                if not generated_text:
                    raise ModelInferenceError(
                        message="Server returned empty response",
                        cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                        chunk_id=chunk.chunk_id
                    )
                
                # Parse response
                try:
                    candidates = parse_model_response(generated_text, chunk.chunk_id)
                except ModelInferenceError as e:
                    e.message = f"{e.message} (took {elapsed_time:.2f}s)"
                    raise
                
                return candidates[:num_candidates]
                
            except aiohttp.ClientError as e:
                last_error = ModelInferenceError(
                    message=f"Request failed: {str(e)}",
                    cause=InferenceErrorCause.UNKNOWN,
                    chunk_id=chunk.chunk_id
                )
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                
            except asyncio.TimeoutError:
                raise ModelInferenceError(
                    message=f"Request timeout after {self.timeout_seconds}s",
                    cause=InferenceErrorCause.TIMEOUT,
                    chunk_id=chunk.chunk_id
                )
            except ModelInferenceError:
                # Re-raise our own errors
                raise
        
        # All retries failed
        if last_error:
            raise last_error
    
    async def is_ready(self) -> bool:
        """Check if adapter is ready"""
        return self._ready
    
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        """
        Memory usage is handled by the server.
        Return 0 since this adapter doesn't manage memory directly.
        """
        return 0
    
    def get_info(self) -> dict:
        """Get adapter information"""
        return {
            "adapter_type": "GemmaServerAdapter",
            "server_url": self.server_url,
            "ready": self._ready,
            "model_info": self._model_info,
            "config": {
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            }
        }


__all__ = ['GemmaServerAdapter']