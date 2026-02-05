"""
Edge Question Generation Pipeline - ONNX Runtime Adapter (SmolLM / HF ONNX)

Key points:
- Loads ONNX weights from Hugging Face Hub (e.g. onnx-community/SmolLM-360M-ONNX)
- Uses Hugging Face tokenizer/config
- Implements text generation (greedy / top-p sampling) with optional KV-cache if present
- Keeps your existing pipeline contract: build_prompt() -> generate() -> parse_model_response()

Dependencies:
  pip install onnxruntime transformers huggingface_hub numpy

Important (CoreML EP):
- Some ONNX exports REQUIRE past_key_values inputs even for the first (prefill) pass.
- Empty past implies past_len=0 -> zero-sized tensors, which CoreML EP does NOT support.
- Solution: run prefill on CPU EP (supports zero-sized), then run cached decode on CoreML EP
  once past_len >= 1 (no zero-sized tensors). This file implements that automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from src.model_adapter import ModelAdapter, parse_model_response, build_prompt
from src.pipeline_types import Chunk, QuestionCandidate, GenerationConfig
from src.errors import ModelInferenceError, InferenceErrorCause

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _top_p_sample(probs: np.ndarray, top_p: float) -> int:
    """Nucleus sampling over 1D probs."""
    if top_p <= 0.0 or top_p >= 1.0:
        return int(np.random.choice(len(probs), p=probs))

    idx = np.argsort(probs)[::-1]
    sorted_probs = probs[idx]
    cdf = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cdf, top_p, side="left") + 1
    cutoff = max(1, min(cutoff, len(sorted_probs)))

    idx = idx[:cutoff]
    p = probs[idx]
    p = p / (p.sum() + 1e-12)
    return int(np.random.choice(idx, p=p))


def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature is None or temperature <= 0:
        return logits
    return logits / float(temperature)


def _find_stop_on_text(text: str, stop_seqs: Optional[List[str]]) -> bool:
    if not stop_seqs:
        return False
    return any(s in text for s in stop_seqs if s)


def _np_i64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# KV-cache helpers (for models that REQUIRE past_key_values inputs)
# -----------------------------------------------------------------------------

def _ort_type_to_np_dtype(ort_type: str) -> np.dtype:
    t = (ort_type or "").lower()
    if "float16" in t:
        return np.float16
    if "float32" in t or "float" in t:
        return np.float32
    return np.float32


def _resolve_dim(
    dim,
    *,
    batch: int,
    kv_heads: int,
    head_dim: int,
    past_len: int,
) -> int:
    if isinstance(dim, int):
        return dim
    if dim is None:
        return 1
    if isinstance(dim, str):
        d = dim.lower()
        if "batch" in d:
            return batch
        if "head" in d and ("num" in d or "n_" in d or "kv" in d):
            return kv_heads
        if "head_dim" in d or ("dim" in d and "head" in d):
            return head_dim
        if "past" in d or "cache" in d:
            return past_len
        if "seq" in d or "time" in d or "len" in d:
            return past_len
    return 1


def _guess_and_force_past_dim(resolved: List[int], *, batch: int, kv_heads: int, head_dim: int, past_len: int) -> List[int]:
    if len(resolved) != 4:
        return resolved

    # Prefer last dim == head_dim and common layouts [B, H, P, D] or [B, P, H, D]
    if resolved[-1] == head_dim:
        if resolved[1] == kv_heads:
            resolved[2] = past_len
        elif resolved[2] == kv_heads:
            resolved[1] = past_len
        else:
            resolved[2] = past_len
        return resolved

    # Otherwise force the "other" dim to past_len
    candidates = []
    for i, v in enumerate(resolved):
        if i == 0 and v == batch:
            continue
        if v == kv_heads:
            continue
        if v == head_dim:
            continue
        candidates.append(i)
    if candidates:
        resolved[candidates[0]] = past_len
        return resolved

    resolved[2] = past_len
    return resolved


def _build_past_kv(
    session: ort.InferenceSession,
    config,
    *,
    batch: int,
    past_len: int,
) -> Dict[str, np.ndarray]:
    """
    Build KV-cache feed tensors for ALL past_key_values.* inputs.
    past_len can be 0 (CPU supports this; CoreML EP does not).
    """
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is None or num_heads is None:
        raise RuntimeError("Config missing hidden_size/num_attention_heads; cannot build past_key_values inputs.")

    head_dim = int(hidden_size) // int(num_heads)
    kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
    kv_heads = int(kv_heads)

    out: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if not name.startswith("past_key_values."):
            continue

        np_dtype = _ort_type_to_np_dtype(inp.type)
        shape = list(inp.shape)

        resolved = [
            _resolve_dim(d, batch=batch, kv_heads=kv_heads, head_dim=head_dim, past_len=past_len)
            for d in shape
        ]
        resolved = _guess_and_force_past_dim(resolved, batch=batch, kv_heads=kv_heads, head_dim=head_dim, past_len=past_len)

        out[name] = np.zeros(resolved, dtype=np_dtype)

    return out


# -----------------------------------------------------------------------------
# ONNX causal LM runner (with KV-cache + CoreML-safe prefill)
# -----------------------------------------------------------------------------

@dataclass
class OrtxModelSpec:
    model_id: str
    onnx_filename: str  # e.g. "model_int8.onnx"
    subfolder: str = "onnx"


class OrtxCausalLM:
    """
    Minimal causal LM runner on ONNX Runtime.

    Supports:
    - KV-cache if the ONNX graph exposes past/present key/value inputs/outputs.
    - Some graphs REQUIRE past_key_values even for prefill.
    - CoreML EP cannot accept past_len=0 (zero-sized tensors), so:
        * Prefill runs on CPU EP if empty past is needed.
        * Cached decode runs on accelerated EP (CoreML/NNAPI/CUDA) when available.
    """

    def __init__(
        self,
        spec: OrtxModelSpec,
        providers: Optional[List[str]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
        hf_token: Optional[str] = None,
    ):
        self.spec = spec
        self.sess_options = sess_options or self._default_sess_options()

        model_path = Path(spec.model_id).expanduser().resolve()
        
        if model_path.exists() and model_path.is_dir():
            logger.info(f"Loading ONNX model from local path: {model_path}")
            
            # Construct ONNX file path
            if spec.subfolder and spec.subfolder.strip():
                onnx_file = model_path / spec.subfolder / spec.onnx_filename
            else:
                onnx_file = model_path / spec.onnx_filename
            
            # Fallback: try without subfolder if file not found
            if not onnx_file.exists():
                onnx_file = model_path / spec.onnx_filename
            
            if not onnx_file.exists():
                # List what files ARE in the directory
                available_files = list(model_path.glob("*.onnx"))
                if model_path.is_dir():
                    available_files.extend(list(model_path.glob("**/*.onnx")))
                
                error_msg = (
                    f"ONNX file '{spec.onnx_filename}' not found in {model_path}\n\n"
                    f"Searched locations:\n"
                    f"  1. {model_path / spec.subfolder / spec.onnx_filename if spec.subfolder else 'N/A'}\n"
                    f"  2. {model_path / spec.onnx_filename}\n\n"
                    f"Available ONNX files in directory:\n"
                )
                if available_files:
                    for f in available_files:
                        error_msg += f"  - {f.relative_to(model_path)}\n"
                else:
                    error_msg += "  (no .onnx files found)\n"
                
                raise FileNotFoundError(error_msg)
            
            self.onnx_path = str(onnx_file)
            logger.info(f"Using ONNX file: {self.onnx_path}")
            
            # Load tokenizer/config from local directory
            logger.info(f"Loading tokenizer and config from {model_path}")
            self.config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), 
                use_fast=True, 
                local_files_only=True
            )
            logger.info(f"Tokenizer and config loaded successfully")
            
        else:
            # HUGGING FACE HUB MODEL
            logger.info(f"Path not found locally, treating as HuggingFace model ID")
            logger.info(f"Downloading ONNX model from Hugging Face: {spec.model_id}")
            
            # Download ONNX file
            self.onnx_path = hf_hub_download(
                repo_id=spec.model_id,
                filename=spec.onnx_filename,
                subfolder=spec.subfolder,
                token=hf_token,
            )
            logger.info(f"Downloaded to: {self.onnx_path}")

            # Load tokenizer/config from HF
            logger.info(f"Loading tokenizer and config from HuggingFace")
            self.config = AutoConfig.from_pretrained(spec.model_id, token=hf_token)
            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, use_fast=True, token=hf_token)
            logger.info(f"Tokenizer and config loaded successfully")

        # Common setup (both local and HF)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token")

        # Decide providers
        self.available_providers = ort.get_available_providers()
        self.user_providers = providers

        # Always create a CPU-only session (for prefill and fallback)
        logger.info(f"Creating CPU session...")
        self.session_cpu = ort.InferenceSession(
            self.onnx_path,
            sess_options=self.sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"CPU session created")

        # Create an "accelerated" session if possible (CoreML/NNAPI/CUDA/etc.)
        accel_providers = self._select_accel_providers(providers)
        self.session_accel = None
        if accel_providers and accel_providers != ["CPUExecutionProvider"]:
            try:
                logger.info(f"Creating accelerated session with providers: {accel_providers}")
                self.session_accel = ort.InferenceSession(
                    self.onnx_path,
                    sess_options=self.sess_options,
                    providers=accel_providers,
                )
                logger.info(f"Accelerated session created")
            except Exception as e:
                logger.warning(f"Failed to create accel session ({accel_providers}). Falling back to CPU only. Error: {e}")
                self.session_accel = None

        # Introspect I/O from CPU session (should match accel session)
        self.input_names = [i.name for i in self.session_cpu.get_inputs()]
        self.output_names = [o.name for o in self.session_cpu.get_outputs()]

        self._kv = self._detect_kv_cache_io(self.input_names, self.output_names)
        self._requires_past_inputs = any(n.startswith("past_key_values.") for n in self.input_names)

        logger.info(
            "ONNX model loaded successfully:\n"
            f"   Model: {spec.model_id}\n"
            f"   ONNX file: {spec.onnx_filename}\n"
            f"   Accelerated providers: {accel_providers or ['CPU only']}\n"
            f"   KV cache enabled: {self._kv['enabled']}\n"
            f"   Requires past inputs: {self._requires_past_inputs}"
        )

    @staticmethod
    def _default_sess_options() -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return so

    def _select_accel_providers(self, providers: Optional[List[str]]) -> List[str]:
        """
        Choose providers for the accelerated session.
        If user provided providers, we respect them (but keep CPU as fallback).
        Else we choose best available in a sensible order.
        """
        if providers:
            # Ensure CPU fallback exists at the end
            p = [x for x in providers if x in self.available_providers]
            if "CPUExecutionProvider" not in p:
                p.append("CPUExecutionProvider")
            return p or ["CPUExecutionProvider"]

        preferred = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
        p = [x for x in preferred if x in self.available_providers]
        return p or ["CPUExecutionProvider"]

    @staticmethod
    def _detect_kv_cache_io(inputs: List[str], outputs: List[str]) -> Dict[str, object]:
        past_key = re.compile(r"^past_key_values\.(\d+)\.key$")
        past_val = re.compile(r"^past_key_values\.(\d+)\.value$")
        pres_key = re.compile(r"^(present|present_key_values)\.(\d+)\.key$")
        pres_val = re.compile(r"^(present|present_key_values)\.(\d+)\.value$")

        past_layers = set()
        for n in inputs:
            m = past_key.match(n) or past_val.match(n)
            if m:
                past_layers.add(int(m.group(1)))

        pres_layers = set()
        for n in outputs:
            m = pres_key.match(n) or pres_val.match(n)
            if m:
                pres_layers.add(int(m.group(2)))

        enabled = (len(past_layers) > 0) and (len(past_layers) == len(pres_layers) or len(pres_layers) > 0)

        return {
            "enabled": enabled,
            "past_layers": sorted(past_layers),
            "present_layers": sorted(pres_layers),
        }

    def _build_feeds_base(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Dict[str, np.ndarray]:
        feeds: Dict[str, np.ndarray] = {}

        if "input_ids" in self.input_names:
            feeds["input_ids"] = _np_i64(input_ids)
        else:
            raise RuntimeError("ONNX model missing required input 'input_ids'")

        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = _np_i64(attention_mask)

        if "position_ids" in self.input_names:
            seq_len = int(input_ids.shape[1])
            pos = np.arange(seq_len, dtype=np.int64)[None, :]
            feeds["position_ids"] = pos

        return feeds

    def _extract_present(self, ort_outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        present: Dict[str, np.ndarray] = {}
        for name, arr in zip(self.output_names, ort_outputs):
            if name.startswith("present") or name.startswith("present_key_values"):
                present[name] = arr
        return present

    def _logits_from_outputs(self, ort_outputs: List[np.ndarray]) -> np.ndarray:
        if "logits" in self.output_names:
            idx = self.output_names.index("logits")
            return ort_outputs[idx]
        return ort_outputs[0]

    def _present_to_past(self, present: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        past: Dict[str, np.ndarray] = {}
        for name, arr in present.items():
            m = re.match(r"^(present|present_key_values)\.(\d+)\.(key|value)$", name)
            if not m:
                continue
            layer = m.group(2)
            kv = m.group(3)
            past[f"past_key_values.{layer}.{kv}"] = arr
        return past

    def _run_cpu(self, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session_cpu.run(None, feeds)

    def _run_accel_or_cpu(self, feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Prefer accel session if available; fall back to CPU if accel fails.
        """
        if self.session_accel is None:
            return self.session_cpu.run(None, feeds)
        try:
            return self.session_accel.run(None, feeds)
        except Exception as e:
            logger.warning("Accel session run failed; falling back to CPU. Error=%s", e)
            return self.session_cpu.run(None, feeds)

    def generate_text(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
    ) -> str:
        # Handle chat format with proper template application
        original_prompt = prompt  # Keep for later stripping
        
        if isinstance(prompt, list):
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                try:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True  # This adds the assistant prefix
                    )
                    logger.info(f"Applied chat template, length: {len(formatted_prompt)}")
                except Exception as e:
                    logger.warning(f"Chat template failed: {e}, using fallback")
                    from src.model_adapter import format_chat_prompt
                    formatted_prompt = format_chat_prompt(prompt, self.tokenizer)
            else:
                logger.warning("No chat template found, using fallback formatting")
                from src.model_adapter import format_chat_prompt
                formatted_prompt = format_chat_prompt(prompt, self.tokenizer)
        else:
            formatted_prompt = prompt
        
        # Log the actual prompt being sent (truncated)
        logger.info(f"Formatted prompt (first 300 chars):\n{formatted_prompt[:300]}")
        logger.info(f"Formatted prompt (last 200 chars):\n{formatted_prompt[-200:]}")
        
        enc = self.tokenizer(formatted_prompt, return_tensors="np", add_special_tokens=False)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        # Store the input length to strip the prompt later
        input_length = int(input_ids.shape[1])
        logger.info(f"Input tokens: {input_length}")
        
        generated_ids = input_ids.copy()

        use_cache = bool(self._kv["enabled"])
        past: Optional[Dict[str, np.ndarray]] = None
        past_length = int(generated_ids.shape[1])

        # [... rest of prefill code stays the same ...]
        feeds = self._build_feeds_base(generated_ids, attention_mask)

        if self._requires_past_inputs:
            feeds.update(_build_past_kv(self.session_cpu, self.config, batch=int(generated_ids.shape[0]), past_len=0))
            ort_out = self._run_cpu(feeds)
        else:
            ort_out = self._run_accel_or_cpu(feeds)

        logits = self._logits_from_outputs(ort_out)

        if use_cache:
            present = self._extract_present(ort_out)
            past = self._present_to_past(present)

        # Decode loop
        for step in range(int(max_new_tokens)):
            last_logits = logits[:, -1, :].astype(np.float32)
            last_logits = _apply_temperature(last_logits, temperature)

            if temperature is None or temperature <= 0:
                next_id = int(np.argmax(last_logits, axis=-1)[0])
            else:
                probs = _softmax(last_logits[0])
                p = float(top_p) if top_p is not None else 1.0
                next_id = _top_p_sample(probs, p)

            next_token = np.array([[next_id]], dtype=np.int64)
            generated_ids = np.concatenate([generated_ids, next_token], axis=1)

            # Check for EOS
            if next_id == int(self.tokenizer.eos_token_id):
                logger.info(f"EOS token encountered at step {step}")
                break

            # Decode only the NEW tokens (strip the prompt)
            new_tokens = generated_ids[0, input_length:]  # ← Key change
            text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Check stop sequences on the generated portion only
            if _find_stop_on_text(text_out, stop_sequences):
                logger.info(f"Stop sequence found at step {step}")
                break

            # [... rest of cache/no-cache logic stays the same ...]
            if use_cache:
                input_step = next_token
                attention_mask = np.ones_like(generated_ids, dtype=np.int64)
                feeds = self._build_feeds_base(input_step, attention_mask)
                if "position_ids" in self.input_names:
                    feeds["position_ids"] = np.array([[past_length]], dtype=np.int64)
                if self._requires_past_inputs:
                    if past is None:
                        feeds.update(_build_past_kv(self.session_cpu, self.config, batch=1, past_len=0))
                    else:
                        feeds.update(past)
                ort_out = self._run_accel_or_cpu(feeds)
                logits = self._logits_from_outputs(ort_out)
                present = self._extract_present(ort_out)
                past = self._present_to_past(present)
                past_length += 1
            else:
                attention_mask = np.ones_like(generated_ids, dtype=np.int64)
                feeds = self._build_feeds_base(generated_ids, attention_mask)
                if self._requires_past_inputs:
                    feeds.update(_build_past_kv(self.session_cpu, self.config, batch=1, past_len=0))
                    ort_out = self._run_cpu(feeds)
                else:
                    ort_out = self._run_accel_or_cpu(feeds)
                logits = self._logits_from_outputs(ort_out)

        # Final decode - only the generated portion
        new_tokens = generated_ids[0, input_length:]
        text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        logger.info(f"Generated {len(new_tokens)} new tokens")
        logger.info(f"Output (first 500 chars): {text_out[:500]}")
        
        return text_out.strip()


# -----------------------------------------------------------------------------
# Adapter: matches your ModelAdapter interface
# -----------------------------------------------------------------------------

class OnnxRuntimeAdapter(ModelAdapter):
    """
    Adapter for local ONNX Runtime inference.

    Mirrors your LlamaServerAdapter:
      - generate_questions(chunk, num_candidates, generation_config) -> List[QuestionCandidate]
      - retries + consistent ModelInferenceError mapping
    """

    def __init__(
        self,
        model_id: str = "onnx-community/SmolLM-360M-ONNX",
        onnx_filename: str = "model_int8.onnx",
        onnx_subfolder: str = "onnx",
        providers: Optional[List[str]] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.75,
        hf_token: Optional[str] = None,
        run_in_thread: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.run_in_thread = run_in_thread

        self._ready = False
        self._model = OrtxCausalLM(
            OrtxModelSpec(model_id=model_id, onnx_filename=onnx_filename, subfolder=onnx_subfolder),
            providers=providers,
            hf_token=hf_token,
        )
        self._ready = True

    async def is_ready(self) -> bool:
        return self._ready

    async def wait_until_ready(self, max_wait_seconds: float = 60.0) -> bool:
        return self._ready

    def estimate_memory_usage(self, chunk: Chunk) -> int:
        chunk_tokens = len(chunk.text) // 4
        prompt_overhead = 250
        generation_buffer = 600
        total_tokens = chunk_tokens + prompt_overhead + generation_buffer
        return total_tokens * 4

    async def generate_questions(
        self,
        chunk: Chunk,
        num_candidates: int,
        generation_config: GenerationConfig
    ) -> List[QuestionCandidate]:
        prompt = build_prompt(chunk.text, num_candidates, generation_config)

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                def _call_model() -> str:
                    return self._model.generate_text(
                        prompt=prompt,
                        max_new_tokens=int(generation_config.max_output_tokens),
                        temperature=float(getattr(generation_config, "temperature", 0.7) or 0.7),
                        top_p=float(getattr(generation_config, "top_p", 0.9) or 0.9),
                        stop_sequences=getattr(generation_config, "stop_sequences", None),
                    )

                if self.run_in_thread:
                    response_text = await asyncio.wait_for(
                        asyncio.to_thread(_call_model),
                        timeout=self.timeout_seconds,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    response_text = await asyncio.wait_for(
                        loop.run_in_executor(None, _call_model),
                        timeout=self.timeout_seconds,
                    )

                if not response_text or not response_text.strip():
                    raise ModelInferenceError(
                        message="Model returned empty response",
                        cause=InferenceErrorCause.INSUFFICIENT_OUTPUT,
                        chunk_id=chunk.chunk_id,
                    )

                logger.info("Response text (first 500 chars): %s", response_text[:500])
                return parse_model_response(response_text, chunk.chunk_id)

            except asyncio.TimeoutError:
                last_exc = ModelInferenceError(
                    message=f"Inference timed out after {self.timeout_seconds}s",
                    cause=InferenceErrorCause.TIMEOUT,
                    chunk_id=chunk.chunk_id,
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise last_exc

            except ModelInferenceError as e:
                last_exc = e
                if e.is_retryable() and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise

            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
                    last_exc = ModelInferenceError(
                        message=f"OOM during ONNX inference: {e}",
                        cause=InferenceErrorCause.OOM,
                        chunk_id=chunk.chunk_id,
                    )
                else:
                    last_exc = ModelInferenceError(
                        message=f"Unexpected error during ONNX inference: {e}",
                        cause=InferenceErrorCause.UNKNOWN,
                        chunk_id=chunk.chunk_id,
                    )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise last_exc from e

        if isinstance(last_exc, Exception):
            raise last_exc

        raise ModelInferenceError(
            message="Inference failed (unknown)",
            cause=InferenceErrorCause.UNKNOWN,
            chunk_id=chunk.chunk_id,
        )

    async def shutdown(self):
        return


# -----------------------------------------------------------------------------
# Convenience factory
# -----------------------------------------------------------------------------

def create_onnx_adapter(
    model_id: str = "onnx-community/SmolLM-360M-ONNX",
    onnx_filename: str = "model_quantized.onnx",
    dtype: Optional[str] = None,
    providers: Optional[List[str]] = None,
    **kwargs,
) -> OnnxRuntimeAdapter:
    # Resolve path relative to current working directory
    model_path = Path(model_id).expanduser()

    # If it looks like a filesystem path, treat it as local intent
    looks_like_path = (
        model_id.startswith((".", "~")) or
        (os.sep in model_id) or
        ("/" in model_id)  # safe on mac/linux
    )

    # Try to resolve (but don't force HF if it doesn't exist)
    resolved = model_path.resolve()

    if resolved.exists() and resolved.is_dir():
        logger.info("Detected local ONNX model directory")

        onnx_files = list(resolved.glob("*.onnx")) or list(resolved.glob("**/*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {resolved}")

        # Prefer explicit filename if user provided it
        if onnx_filename:
            matching = [f for f in onnx_files if f.name == onnx_filename]
            if matching:
                detected_filename = matching[0].name
            else:
                detected_filename = onnx_files[0].name
                logger.warning(f"Requested file '{onnx_filename}' not found, using: {detected_filename}")
        else:
            detected_filename = onnx_files[0].name

        return OnnxRuntimeAdapter(
            model_id=str(resolved),
            onnx_filename=detected_filename,
            onnx_subfolder="",
            providers=providers,
            **kwargs,
        )

    # If it *looks* like a path but doesn't exist, raise a helpful error instead of HF
    if looks_like_path and not resolved.exists():
        parent = resolved.parent
        suggestions = []
        if parent.exists():
            # show nearby dirs as suggestions
            suggestions = [p.name for p in parent.iterdir() if p.is_dir()]
        msg = (
            f"Local model path not found:\n  {resolved}\n\n"
            f"Did you mean one of these folders in {parent}?\n"
            + "".join(f"  - {s}\n" for s in suggestions[:20])
        )
        raise FileNotFoundError(msg)

    # Otherwise treat as HuggingFace repo id (original logic below)
    logger.info(f"🌐 Not a local path, treating as HuggingFace model ID: {model_id}")

    dtype = (dtype or "int8").lower()
    fname_map = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "int8": "model_int8.onnx",
        "q8": "model_int8.onnx",
        "q4": "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
    }
    onnx_filename = fname_map.get(dtype, "model_int8.onnx")

    return OnnxRuntimeAdapter(
        model_id=model_id,
        onnx_filename=onnx_filename,
        providers=providers,
        **kwargs,
    )



__all__ = [
    "OnnxRuntimeAdapter",
    "create_onnx_adapter",
]