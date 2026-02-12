#!/usr/bin/env python3
"""
Edge Question Generation Pipeline - LlamaServer Adapter (UPDATED)
Mode-aware adapter for llama.cpp server with integrated server management.

Fixes:
✅ Don't stringify chat-message lists into prompts (proper formatting for /v1/completions)
✅ Correct build_prompt_for_mode(...) call signature (was passing args in wrong order)
✅ Use messages list for /v1/chat/completions and formatted string for /v1/completions
✅ Mode-aware prompt building + parsing (JSON vs PLAINTEXT, 1 vs N)
✅ Prefer /v1/completions for determinism (no nested chat confusion)
✅ Stop sequences disabled by default (prevents returning just '}' etc.)
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Any, Tuple

import aiohttp

from local_llm.pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from local_llm.errors import ModelInferenceError, InferenceErrorCause

from models.llama_adaption.llama_server_manager import LlamaServerManager, ServerConfig

# UPDATED imports (mode-aware)
from local_llm.model_adapter import (
    ModelAdapter,
    QuestionGenMode,
    build_prompt_for_mode,
    parse_model_response_for_mode,
    format_chat_prompt,  # ✅ FIX: needed to turn messages -> prompt string for /v1/completions
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _normalize_stop(stop: Any) -> Optional[List[str]]:
    """
    llama-server expects stop to be either:
      - omitted / null
      - a list of strings
      - sometimes a single string

    NOTE: while debugging JSON/brace truncation issues, it's safer to return None.
    We'll still keep this helper, but the adapter below defaults to disabling stop.
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
            s = (x if isinstance(x, str) else str(x)).strip()
            if s:
                out.append(s)
        return out or None
    s = str(stop).strip()
    return [s] if s else None


def _extract_text_from_openai_like(result: dict) -> str:
    """
    Supports both:
    - /v1/completions -> choices[0].text
    - /v1/chat/completions -> choices[0].message.content
    """
    if "choices" not in result or not result["choices"]:
        return ""
    choice = result["choices"][0]
    if isinstance(choice, dict):
        if "text" in choice and isinstance(choice["text"], str):
            return choice["text"]
        if "message" in choice and isinstance(choice["message"], dict):
            c = choice["message"].get("content", "")
            return c if isinstance(c, str) else ""
    return ""


# ============================================================================
# LlamaServer Adapter
# ============================================================================

class LlamaServerAdapter(ModelAdapter):
    """
    Adapter for llama.cpp server (llama-server).

    Uses OpenAI-compatible endpoints:
      - /v1/completions (preferred)
      - /v1/chat/completions (fallback)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        server_manager: Optional["LlamaServerManager"] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.75,
        prefer_completion_endpoint: bool = True,
        debug_log_prompts: bool = False,
        debug_log_raw_response: bool = False,
        disable_stop_sequences: bool = True,   # ✅ important for your current failures
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
        self.prefer_completion_endpoint = prefer_completion_endpoint
        self.debug_log_prompts = debug_log_prompts
        self.debug_log_raw_response = debug_log_raw_response
        self.disable_stop_sequences = disable_stop_sequences

        self._ready = False
        self._session: Optional[aiohttp.ClientSession] = None

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

    # ----------------------------------------------------------------------
    # LEGACY API (kept for compatibility): A = chunked + JSON + 1
    # ----------------------------------------------------------------------
    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig,
    ) -> List[QuestionCandidate]:
        mode = QuestionGenMode.A_chunked_json_1()
        return await self.generate_questions_mode(
            text=chunk.text,
            chunk_id=chunk.chunk_id,
            mode=mode,
            generation_config=generation_config,
        )

    # ----------------------------------------------------------------------
    # MODE-AWARE API used by updated pipeline
    # ----------------------------------------------------------------------
    async def generate_questions_mode(
        self,
        text: str,
        chunk_id: int,
        mode: QuestionGenMode,
        generation_config: GenerationConfig,
    ) -> List[QuestionCandidate]:
        await self._ensure_session()

        if not (await self.is_ready()):
            raise ModelInferenceError(
                message="Server not ready",
                cause=InferenceErrorCause.MODEL_NOT_READY,
                chunk_id=chunk_id,
            )

        # ==================================================================
        # ✅ FIX #1: Correct build_prompt_for_mode(...) signature
        # Your build_prompt_for_mode returns a LIST OF CHAT MESSAGES.
        # It expects: (text, desired_questions, mode, config)
        # ==================================================================
        messages = build_prompt_for_mode(
            text=text,
            desired_questions=mode.questions_per_call,
            mode=mode,
            config=generation_config,
        )

        # ==================================================================
        # ✅ FIX #2: /v1/completions needs a STRING prompt, not messages list
        # ==================================================================
        prompt_str = format_chat_prompt(messages)

        if self.debug_log_prompts:
            logger.info("MODE=%s prompt(first 500)=%s", getattr(mode, "name", str(mode)), prompt_str[:500])

        # ✅ IMPORTANT: disable stop sequences while debugging JSON truncation
        stop_list = None
        if not self.disable_stop_sequences:
            stop_list = _normalize_stop(getattr(generation_config, "stop_sequences", None))

        # Prefer /v1/completions (more deterministic; no nested chat templates)
        completion_payload = {
            "prompt": prompt_str,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_output_tokens,
            "stream": False,
        }
        if stop_list:
            completion_payload["stop"] = stop_list

        # ✅ /v1/chat/completions expects messages list (NOT a single user with prompt_str)
        chat_payload = {
            "messages": messages,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_output_tokens,
            "stream": False,
        }
        if stop_list:
            chat_payload["stop"] = stop_list

        endpoints: List[Tuple[str, dict, str]] = []
        if self.prefer_completion_endpoint:
            endpoints.append((f"{self.base_url}/v1/completions", completion_payload, "completion"))
            endpoints.append((f"{self.base_url}/v1/chat/completions", chat_payload, "chat"))
        else:
            endpoints.append((f"{self.base_url}/v1/chat/completions", chat_payload, "chat"))
            endpoints.append((f"{self.base_url}/v1/completions", completion_payload, "completion"))

        last_exception: Optional[Exception] = None

        for endpoint, payload, api_mode in endpoints:
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
                                chunk_id=chunk_id,
                            )
                            last_exception = err
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                                continue
                            break

                        if response.status == 500:
                            error_text = await response.text()
                            if "out of memory" in error_text.lower():
                                raise ModelInferenceError(
                                    message="Server out of memory",
                                    cause=InferenceErrorCause.OOM,
                                    chunk_id=chunk_id,
                                )
                            raise ModelInferenceError(
                                message=f"Server error: {error_text}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk_id,
                            )

                        if response.status != 200:
                            body = await response.text()
                            raise ModelInferenceError(
                                message=f"HTTP {response.status}: {body}",
                                cause=InferenceErrorCause.UNKNOWN,
                                chunk_id=chunk_id,
                            )

                        result = await response.json()

                        if self.debug_log_raw_response:
                            logger.info("Raw API response (%s): %s", api_mode, result)

                        response_text = _extract_text_from_openai_like(result)

                        if self.debug_log_raw_response:
                            logger.info("Response text (first 300): %s", (response_text or "")[:300])

                        if not response_text or not response_text.strip():
                            raise ModelInferenceError(
                                message="Model returned empty response",
                                cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                                chunk_id=chunk_id,
                            )

                        # ✅ MODE-AWARE parsing (JSON or PLAINTEXT)
                        candidates = parse_model_response_for_mode(
                            response_text,
                            chunk_id=chunk_id,
                            mode=mode
                        )
                        return candidates

                except asyncio.TimeoutError:
                    last_exception = ModelInferenceError(
                        message=f"Request timed out after {self.timeout_seconds}s",
                        cause=InferenceErrorCause.TIMEOUT,
                        chunk_id=chunk_id,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    break

                except ModelInferenceError as e:
                    last_exception = e
                    # retry only for retryable causes
                    if getattr(e, "is_retryable", None) and e.is_retryable() and attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    break

                except Exception as e:
                    last_exception = ModelInferenceError(
                        message=f"Unexpected error during inference: {str(e)}",
                        cause=InferenceErrorCause.UNKNOWN,
                        chunk_id=chunk_id,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    break

        if isinstance(last_exception, ModelInferenceError):
            raise last_exception

        raise ModelInferenceError(
            message="Inference failed (no usable endpoint succeeded)",
            cause=InferenceErrorCause.UNKNOWN,
            chunk_id=chunk_id,
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
