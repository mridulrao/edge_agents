"""
Edge Question Generation Pipeline - LlamaServer Adapter (Enhanced)
Adapter for llama.cpp server with integrated server management
"""

import asyncio
import aiohttp
from typing import List, Optional
from pathlib import Path

from src.model_adapter import ModelAdapter, parse_model_response, build_prompt
from src.pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from src.errors import ModelInferenceError, InferenceErrorCause

from llama_server_manager import LlamaServerManager, ServerConfig


class LlamaServerAdapter(ModelAdapter):
    """
    Adapter for llama.cpp server (llama-server).
    
    Can connect to existing server OR manage its own server instance.
    
    Supports:
    - Local server instances (development)
    - Remote deployments (production)
    - WebGPU, CPU, Metal backends
    - GGUF quantized models
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        server_manager: Optional['LlamaServerManager'] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LlamaServer adapter.
        
        Args:
            base_url: Server endpoint (e.g., "http://localhost:8080")
                     If None and server_manager provided, uses manager's URL
            server_manager: Optional LlamaServerManager instance for lifecycle management
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.server_manager = server_manager
        
        # Determine base URL
        if base_url:
            self.base_url = base_url.rstrip('/')
        elif server_manager:
            self.base_url = server_manager.base_url
        else:
            raise ValueError("Either base_url or server_manager must be provided")
        
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._ready = False
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Create aiohttp session if needed"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def is_ready(self) -> bool:
        """
        Check if llama-server is ready for inference.
        
        Checks /health endpoint to verify model is loaded.
        """
        # If we have a server manager, check process is running first
        if self.server_manager and not self.server_manager.is_running:
            self._ready = False
            return False
        
        await self._ensure_session()
        
        try:
            async with self._session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # llama-server returns {"status": "ok"} when ready
                    self._ready = data.get("status") == "ok"
                    return self._ready
                else:
                    self._ready = False
                    return False
        except Exception:
            self._ready = False
            return False
    
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        """
        Generate questions via llama-server's /v1/completions endpoint.
        
        Implements retry logic for transient failures.
        """
        await self._ensure_session()
        
        # Build prompt
        prompt = build_prompt(chunk.text, num_candidates, generation_config)
        
        # Try chat endpoint first (better for instruction-tuned models)
        # If that fails, fall back to completions
        use_chat_endpoint = True
        
        if use_chat_endpoint:
            # Use chat completions endpoint (works better with instruct models)
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
                "max_tokens": generation_config.max_output_tokens,
                "stop": generation_config.stop_sequences,
                "stream": False,
            }
            endpoint = f"{self.base_url}/v1/chat/completions"
        else:
            # Use completions endpoint (raw text completion)
            payload = {
                "prompt": prompt,
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
                "max_tokens": generation_config.max_output_tokens,
                "stop": generation_config.stop_sequences,
                "stream": False,
            }
            endpoint = f"{self.base_url}/v1/completions"
        
        # Retry loop
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                async with self._session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    
                    # Handle error status codes
                    if response.status == 503:
                        error = ModelInferenceError(
                            message="Server not ready or model not loaded",
                            cause=InferenceErrorCause.MODEL_NOT_READY,
                            chunk_id=chunk.chunk_id
                        )
                        # Retry for 503
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        raise error
                    
                    if response.status == 500:
                        error_text = await response.text()
                        if "out of memory" in error_text.lower():
                            raise ModelInferenceError(
                                message="Server out of memory",
                                cause=InferenceErrorCause.OOM,
                                chunk_id=chunk.chunk_id
                            )
                        else:
                            raise ModelInferenceError(
                                message=f"Server error: {error_text}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk.chunk_id
                            )
                    
                    if response.status != 200:
                        raise ModelInferenceError(
                            message=f"HTTP {response.status}: {await response.text()}",
                            cause=InferenceErrorCause.UNKNOWN,
                            chunk_id=chunk.chunk_id
                        )
                    
                    # Parse response
                    result = await response.json()
                    
                    # Debug: Log the raw response
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Raw API response: {result}")
                    
                    # Extract generated text (handle both chat and completion formats)
                    if "choices" not in result or len(result["choices"]) == 0:
                        raise ModelInferenceError(
                            message="No completion choices returned",
                            cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                            chunk_id=chunk.chunk_id
                        )
                    
                    # Chat format: choices[0].message.content
                    # Completion format: choices[0].text
                    choice = result["choices"][0]
                    if "message" in choice:
                        response_text = choice["message"]["content"]
                    elif "text" in choice:
                        response_text = choice["text"]
                    else:
                        raise ModelInferenceError(
                            message=f"Unexpected response format: {choice}",
                            cause=InferenceErrorCause.PARSE_ERROR,
                            chunk_id=chunk.chunk_id
                        )
                    
                    # Debug: Log the response text
                    logger.info(f"Response text (first 500 chars): {response_text[:500]}")
                    
                    # Check if response is empty
                    if not response_text or not response_text.strip():
                        raise ModelInferenceError(
                            message="Model returned empty response",
                            cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                            chunk_id=chunk.chunk_id
                        )
                    
                    # Parse into QuestionCandidate objects
                    candidates = parse_model_response(response_text, chunk.chunk_id)
                    
                    return candidates
                    
            except asyncio.TimeoutError as e:
                last_exception = ModelInferenceError(
                    message=f"Request timed out after {self.timeout_seconds}s",
                    cause=InferenceErrorCause.TIMEOUT,
                    chunk_id=chunk.chunk_id
                )
                # Retry timeouts
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise last_exception
            
            except ModelInferenceError as e:
                # Check if retryable
                if e.is_retryable() and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise
            
            except Exception as e:
                raise ModelInferenceError(
                    message=f"Unexpected error during inference: {str(e)}",
                    cause=InferenceErrorCause.UNKNOWN,
                    chunk_id=chunk.chunk_id
                ) from e
        
        # If we exhausted retries
        if last_exception:
            raise last_exception
    
    def estimate_memory_usage(self, chunk: Chunk) -> int:
        """
        Estimate memory for this request.
        
        Note: llama-server manages model memory, so this is an approximation
        of the request's incremental memory (context + generation).
        """
        # Rough estimate: ~1 byte per token
        # Chunk + prompt overhead + generation buffer
        chunk_tokens = len(chunk.text) // 4  # ~4 chars per token
        prompt_overhead = 200  # Template tokens
        generation_buffer = 500  # Max generation tokens
        
        total_tokens = chunk_tokens + prompt_overhead + generation_buffer
        
        return total_tokens * 4  # 4 bytes per token (conservative)
    
    async def wait_until_ready(self, max_wait_seconds: float = 60.0) -> bool:
        """
        Wait for server to become ready.
        
        Useful during startup when model is still loading.
        """
        if self.server_manager:
            # Use server manager's health check
            return await self.server_manager._wait_for_health(max_wait_seconds)
        
        # Manual polling
        elapsed = 0.0
        interval = 1.0
        
        while elapsed < max_wait_seconds:
            if await self.is_ready():
                return True
            
            await asyncio.sleep(interval)
            elapsed += interval
        
        return False
    
    async def shutdown(self):
        """Clean up resources"""
        await self._close_session()
        
        # If we own the server manager, stop it
        if self.server_manager:
            await self.server_manager.stop()
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._session and not self._session.closed:
            # Schedule cleanup (best effort)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_session())
            except Exception:
                pass


# ============================================================================
# Convenience Factory Functions
# ============================================================================

async def create_adapter_with_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_gpu_layers: int = 0,
    ctx_size: int = 4096,
    llama_server_path: str = "llama-server",
    **kwargs
) -> LlamaServerAdapter:
    """
    Create adapter with managed server instance.
    
    Starts llama-server automatically and manages its lifecycle.
    
    Args:
        model_path: Path to GGUF model file
        host: Server host
        port: Server port
        n_gpu_layers: Number of GPU layers (0=CPU, -1=all)
        ctx_size: Context window size
        llama_server_path: Path to llama-server binary (default: "llama-server")
        **kwargs: Additional ServerConfig parameters
        
    Returns:
        LlamaServerAdapter with running server
    """
    
    config = ServerConfig(
        model_path=model_path,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        **kwargs
    )
    
    manager = LlamaServerManager(config, llama_server_path=llama_server_path)
    
    # Start server
    success = await manager.start()
    if not success:
        raise RuntimeError("Failed to start llama-server")
    
    # Create adapter
    adapter = LlamaServerAdapter(server_manager=manager)
    
    return adapter


def create_adapter_for_remote(base_url: str, **kwargs) -> LlamaServerAdapter:
    """
    Create adapter for existing remote server.
    
    Args:
        base_url: Server URL (e.g., "http://localhost:8080")
        **kwargs: Additional LlamaServerAdapter parameters
        
    Returns:
        LlamaServerAdapter configured for remote server
    """
    return LlamaServerAdapter(base_url=base_url, **kwargs)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'LlamaServerAdapter',
    'create_adapter_with_server',
    'create_adapter_for_remote',
]