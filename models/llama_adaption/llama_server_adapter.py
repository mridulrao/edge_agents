#!/usr/bin/env python3
"""
Edge Question Generation Pipeline - LlamaServer Adapter (Enhanced)
Adapter for llama.cpp server with integrated server management

Supports:
- Local server instances (development)
- Remote deployments (production)
- CPU / Metal / GPU backends (depending on llama-server build)
- GGUF models (BF16/F16/Q4/Q5/Q8, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Any

import aiohttp

from local_llm.model_adapter import ModelAdapter, parse_model_response, build_prompt
from local_llm.pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from local_llm.errors import ModelInferenceError, InferenceErrorCause

from models.llama_adaption.llama_server_manager import LlamaServerManager, ServerConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _content_to_string(content: Any) -> str:
    """
    Normalize OpenAI-style "content parts" or arbitrary prompt payloads into a plain string.

    llama-server /v1/chat/completions commonly supports ONLY:
      messages: [{ role: "...", content: "<string>" }]

    It often rejects:
      content: [{type:"text", text:"..."}]  -> "unsupported content[].type"
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                    continue
                if "text" in item:
                    parts.append(str(item["text"]))
                    continue
                parts.append(str(item))
                continue
            parts.append(str(item))
        return "\n".join(p for p in parts if p is not None)

    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        if "text" in content:
            return str(content["text"])
        return str(content)

    return str(content)


def _normalize_stop(stop: Any) -> Optional[List[str]]:
    """
    llama-server expects stop to be either:
      - omitted / null
      - a list of strings
      - sometimes a single string
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        s = stop.strip()
        return [s] if s else None
    if isinstance(stop, list):
        out = []
        for x in stop:
            if x is None:
                continue
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
            else:
                s = str(x).strip()
                if s:
                    out.append(s)
        return out or None
    s = str(stop).strip()
    return [s] if s else None


# ============================================================================
# LlamaServer Adapter
# ============================================================================

class LlamaServerAdapter(ModelAdapter):
    """
    Adapter for llama.cpp server (llama-server).

    Can connect to existing server OR manage its own server instance.

    Uses OpenAI-compatible endpoints:
      - /v1/chat/completions (preferred for instruct/chat models)
      - /v1/completions (fallback)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        server_manager: Optional["LlamaServerManager"] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        prefer_chat_endpoint: bool = True,
        debug_log_prompts: bool = False,
    ):
        self.server_manager = server_manager

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif server_manager:
            self.base_url = server_manager.base_url
        else:
            raise ValueError("Either base_url or server_manager must be provided")

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prefer_chat_endpoint = prefer_chat_endpoint
        self.debug_log_prompts = debug_log_prompts

        self._ready = False
        self._session: Optional[aiohttp.ClientSession] = None

    # NEW: convenience
    def get_server_pid(self) -> Optional[int]:
        return self.server_manager.pid if self.server_manager else None

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_ready(self) -> bool:
        if self.server_manager and not self.server_manager.is_running:
            self._ready = False
            return False

        await self._ensure_session()

        try:
            async with self._session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._ready = data.get("status") == "ok"
                    return self._ready
                self._ready = False
                return False
        except Exception:
            self._ready = False
            return False

    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig,
    ) -> List[QuestionCandidate]:
        await self._ensure_session()

        prompt_obj = build_prompt(chunk.text, num_candidates, generation_config)
        prompt_str = _content_to_string(prompt_obj)

        if self.debug_log_prompts:
            logger.info("build_prompt() returned type=%s", type(prompt_obj))
            logger.info("prompt (first 500 chars): %s", prompt_str[:500])

        stop_list = _normalize_stop(getattr(generation_config, "stop_sequences", None))

        chat_payload = {
            "messages": [{"role": "user", "content": prompt_str}],
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_output_tokens,
            "stream": False,
        }
        if stop_list:
            chat_payload["stop"] = stop_list

        completion_payload = {
            "prompt": prompt_str,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_output_tokens,
            "stream": False,
        }
        if stop_list:
            completion_payload["stop"] = stop_list

        endpoints_to_try = []
        if self.prefer_chat_endpoint:
            endpoints_to_try.append((f"{self.base_url}/v1/chat/completions", chat_payload, "chat"))
            endpoints_to_try.append((f"{self.base_url}/v1/completions", completion_payload, "completion"))
        else:
            endpoints_to_try.append((f"{self.base_url}/v1/completions", completion_payload, "completion"))
            endpoints_to_try.append((f"{self.base_url}/v1/chat/completions", chat_payload, "chat"))

        last_exception: Optional[Exception] = None

        for endpoint, payload, mode in endpoints_to_try:
            for attempt in range(self.max_retries):
                try:
                    async with self._session.post(
                        endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                    ) as response:

                        if response.status == 503:
                            err = ModelInferenceError(
                                message="Server not ready or model not loaded",
                                cause=InferenceErrorCause.MODEL_NOT_READY,
                                chunk_id=chunk.chunk_id,
                            )
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (2**attempt))
                                continue
                            raise err

                        if response.status == 500:
                            error_text = await response.text()
                            if "out of memory" in error_text.lower():
                                raise ModelInferenceError(
                                    message="Server out of memory",
                                    cause=InferenceErrorCause.OOM,
                                    chunk_id=chunk.chunk_id,
                                )
                            raise ModelInferenceError(
                                message=f"Server error: {error_text}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk.chunk_id,
                            )

                        if response.status == 400:
                            body = await response.text()
                            if mode == "chat" and "unsupported content[].type" in body:
                                logger.warning(
                                    "llama-server chat endpoint rejected content parts; "
                                    "falling back to /v1/completions. body=%s",
                                    body[:300],
                                )
                                last_exception = ModelInferenceError(
                                    message=f"HTTP 400: {body}",
                                    cause=InferenceErrorCause.PARSE_ERROR,
                                    chunk_id=chunk.chunk_id,
                                )
                                break
                            raise ModelInferenceError(
                                message=f"HTTP 400: {body}",
                                cause=InferenceErrorCause.PARSE_ERROR,
                                chunk_id=chunk.chunk_id,
                            )

                        if response.status != 200:
                            raise ModelInferenceError(
                                message=f"HTTP {response.status}: {await response.text()}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk.chunk_id,
                            )

                        result = await response.json()
                        logger.info("Raw API response (%s): %s", mode, result)

                        if "choices" not in result or not result["choices"]:
                            raise ModelInferenceError(
                                message="No completion choices returned",
                                cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                                chunk_id=chunk.chunk_id,
                            )

                        choice = result["choices"][0]
                        if "message" in choice and isinstance(choice["message"], dict):
                            response_text = choice["message"].get("content", "")
                        elif "text" in choice:
                            response_text = choice.get("text", "")
                        else:
                            raise ModelInferenceError(
                                message=f"Unexpected response format: {choice}",
                                cause=InferenceErrorCause.PARSE_ERROR,
                                chunk_id=chunk.chunk_id,
                            )

                        logger.info("Response text (first 500 chars): %s", (response_text or "")[:500])

                        if not response_text or not response_text.strip():
                            raise ModelInferenceError(
                                message="Model returned empty response",
                                cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                                chunk_id=chunk.chunk_id,
                            )

                        candidates = parse_model_response(response_text, chunk.chunk_id)
                        return candidates

                except asyncio.TimeoutError:
                    last_exception = ModelInferenceError(
                        message=f"Request timed out after {self.timeout_seconds}s",
                        cause=InferenceErrorCause.TIMEOUT,
                        chunk_id=chunk.chunk_id,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    break

                except ModelInferenceError as e:
                    last_exception = e
                    if e.is_retryable() and attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    if mode == "chat":
                        break
                    raise

                except Exception as e:
                    last_exception = ModelInferenceError(
                        message=f"Unexpected error during inference: {str(e)}",
                        cause=InferenceErrorCause.UNKNOWN,
                        chunk_id=chunk.chunk_id,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    break

        if isinstance(last_exception, ModelInferenceError):
            raise last_exception
        raise ModelInferenceError(
            message="Inference failed (no usable endpoint succeeded)",
            cause=InferenceErrorCause.UNKNOWN,
            chunk_id=chunk.chunk_id,
        )

    def estimate_memory_usage(self, chunk: Chunk) -> int:
        chunk_tokens = max(1, len(chunk.text) // 4)
        prompt_overhead = 200
        generation_buffer = 500
        total_tokens = chunk_tokens + prompt_overhead + generation_buffer
        return total_tokens * 4

    async def wait_until_ready(self, max_wait_seconds: float = 60.0) -> bool:
        if self.server_manager:
            return await self.server_manager._wait_for_health(max_wait_seconds)

        elapsed = 0.0
        interval = 1.0
        while elapsed < max_wait_seconds:
            if await self.is_ready():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False

    async def shutdown(self) -> None:
        await self._close_session()
        if self.server_manager:
            await self.server_manager.stop()

    def __del__(self):
        try:
            if self._session and not self._session.closed:
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
    **kwargs,
) -> LlamaServerAdapter:
    config = ServerConfig(
        model_path=model_path,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        **kwargs,
    )

    manager = LlamaServerManager(config, llama_server_path=llama_server_path)

    success = await manager.start()
    if not success:
        raise RuntimeError("Failed to start llama-server")

    adapter = LlamaServerAdapter(server_manager=manager)
    return adapter


def create_adapter_for_remote(base_url: str, **kwargs) -> LlamaServerAdapter:
    return LlamaServerAdapter(base_url=base_url, **kwargs)


__all__ = [
    "LlamaServerAdapter",
    "create_adapter_with_server",
    "create_adapter_for_remote",
]
