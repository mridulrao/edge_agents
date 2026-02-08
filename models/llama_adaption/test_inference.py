#!/usr/bin/env python3
"""
Profiles llama-server load + inference and writes JSONL.

Usage:
  pip install psutil
  python profile_llama_server_test.py --out profiles/falcon90m_bf16.jsonl --interval 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import psutil

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.llama_adaption.llama_server_adapter import create_adapter_with_server
from local_llm.pipeline import QuestionGenerationPipeline
from local_llm.pipeline_types import ArticleInput, GenerationConfig


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj)
        obj.setdefault("ts_unix", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


@dataclass
class ProcMem:
    rss_mb: float
    vms_mb: float
    uss_mb: Optional[float] = None


def read_process_memory(pid: int) -> ProcMem:
    p = psutil.Process(pid)
    mi = p.memory_info()
    rss = _bytes_to_mb(mi.rss)
    vms = _bytes_to_mb(getattr(mi, "vms", 0))

    uss = None
    try:
        mfi = p.memory_full_info()
        if hasattr(mfi, "uss"):
            uss = _bytes_to_mb(mfi.uss)
    except Exception:
        pass

    return ProcMem(rss_mb=rss, vms_mb=vms, uss_mb=uss)


async def sample_process_memory(
    pid: int,
    logger: JsonlLogger,
    stage: str,
    interval_s: float,
    stop_event: asyncio.Event,
) -> Dict[str, Any]:
    rss_peak = 0.0
    vms_peak = 0.0
    uss_peak = None
    samples = 0

    while not stop_event.is_set():
        try:
            mem = read_process_memory(pid)
            rss_peak = max(rss_peak, mem.rss_mb)
            vms_peak = max(vms_peak, mem.vms_mb)
            if mem.uss_mb is not None:
                uss_peak = mem.uss_mb if uss_peak is None else max(uss_peak, mem.uss_mb)

            logger.write({
                "event": "mem_sample",
                "stage": stage,
                "pid": pid,
                "rss_mb": mem.rss_mb,
                "vms_mb": mem.vms_mb,
                "uss_mb": mem.uss_mb,
            })
            samples += 1

        except psutil.NoSuchProcess:
            logger.write({"event": "mem_sample", "stage": stage, "pid": pid, "error": "process_exited"})
            break
        except Exception as e:
            logger.write({"event": "mem_sample", "stage": stage, "pid": pid, "error": str(e)})

        await asyncio.sleep(interval_s)

    return {
        "rss_peak_mb": rss_peak,
        "vms_peak_mb": vms_peak,
        "uss_peak_mb": uss_peak,
        "samples": samples,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profiles/llama_profile.jsonl")
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--ctx", type=int, default=1048)
    ap.add_argument("--max_out", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    logger = JsonlLogger(args.out)

    MODEL_PATH = "models/llama_adaption/models/unsloth_gemma-3-270m-it-GGUF/gemma-3-270m-it-UD-Q4_K_XL.gguf"
    LLAMA_SERVER_PATH = "models/llama.cpp/build/bin/llama-server"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return 1

    sample_article = ArticleInput(
        text="""
Skateboarding began as a desperate attempt to catch a wave on land. In the late 1950s, surfers in California sought a way to practice their balance when the ocean was calm. They nailed clay roller skate wheels to wooden planks and crates, creating a crude vehicle they called sidewalk surfing. These early riders emulated the fluid, carving movements of the water, unaware that they were laying the foundation for a global cultural phenomenon.
The sport underwent a massive technological leap in the early 1970s with the invention of the polyurethane wheel. Before this, wheels were made of metal or clay, which offered no grip and turned every pebble into a hazard. The new urethane wheels provided a smooth, grippy ride that allowed skaters to reach higher speeds and perform tighter turns. This era also saw the legendary Z-Boys of Santa Monica take advantage of a severe California drought by sneaking into empty, bowl-shaped swimming pools. This transition from flat ground to vertical walls gave birth to aerial maneuvers and transformed the urban landscape into a giant playground.
By the 1980s, skateboarding had moved away from the surf-inspired style and into the streets. Led by innovators like Alan Gelfand, who invented the ollie, and Rodney Mullen, who pioneered technical flip tricks, the board became an extension of the body that could leap over obstacles and slide down handrails. This period solidified skateboarding as a counterculture movement, deeply entwined with punk rock and DIY aesthetics. It was more than a sport; it was a rebellious identity that challenged the traditional use of public space.
The 1990s and 2000s brought skateboarding into the mainstream through the X Games and popular video games, turning professional skaters into household names. Despite this commercial success, the core of the activity remained rooted in the community and the relentless pursuit of individual style. In 2021, the journey reached a historic milestone when skateboarding debuted at the Tokyo Olympic Games. Today, it is recognized as a legitimate athletic discipline, yet it retains its soul as a creative art form. Whether in a high-tech Olympic park or on a gritty city curb, skateboarding continues to represent a unique blend of resilience, freedom, and the endless reimagining of the world around us.""".strip(),
        desired_questions=4,
    )

    adapter = None
    try:
        logger.write({
            "event": "run_start",
            "model_path": MODEL_PATH,
            "ctx_size": args.ctx,
            "max_output_tokens": args.max_out,
            "threads": args.threads,
            "port": args.port,
        })

        # ----------------------------
        # Start server + sample memory during load
        # ----------------------------
        t0 = time.perf_counter()
        adapter = await create_adapter_with_server(
            model_path=MODEL_PATH,
            host="127.0.0.1",
            port=args.port,
            n_gpu_layers=0,
            ctx_size=args.ctx,
            n_threads=args.threads,
            llama_server_path=LLAMA_SERVER_PATH,
        )
        t1 = time.perf_counter()

        pid = adapter.get_server_pid() or (adapter.server_manager.pid if adapter.server_manager else None)
        if pid is None:
            logger.write({"event": "error", "error": "missing_pid"})
            print("❌ Could not determine llama-server PID")
            return 1

        logger.write({
            "event": "server_spawned",
            "pid": pid,
            "spawn_latency_ms": (t1 - t0) * 1000.0,
        })

        # If manager captured load stats, log them
        if adapter.server_manager and adapter.server_manager.last_start_stats:
            logger.write({
                "event": "manager_start_stats",
                "pid": pid,
                "stats": adapter.server_manager.last_start_stats,
            })

        # Optionally log /props
        try:
            props = await adapter.server_manager.get_server_info() if adapter.server_manager else None
            logger.write({"event": "server_props", "pid": pid, "props": props})
        except Exception as e:
            logger.write({"event": "server_props", "pid": pid, "error": str(e)})

        # ----------------------------
        # Inference profiling
        # ----------------------------
        config = GenerationConfig(max_output_tokens=args.max_out, temperature=0.3, top_p=1)
        pipeline = QuestionGenerationPipeline(adapter=adapter, config=config)

        mem_before = read_process_memory(pid)
        logger.write({
            "event": "infer_start",
            "pid": pid,
            "rss_mb_before": mem_before.rss_mb,
            "vms_mb_before": mem_before.vms_mb,
            "uss_mb_before": mem_before.uss_mb,
        })

        stop = asyncio.Event()
        sampler_task = asyncio.create_task(
            sample_process_memory(pid, logger, stage="infer", interval_s=args.interval, stop_event=stop)
        )

        t_inf0 = time.perf_counter()
        result = await pipeline.generate_with_metrics(sample_article)
        t_inf1 = time.perf_counter()

        stop.set()
        infer_peak = await sampler_task

        mem_after = read_process_memory(pid)
        logger.write({
            "event": "infer_end",
            "pid": pid,
            "infer_latency_ms": (t_inf1 - t_inf0) * 1000.0,
            "rss_mb_after": mem_after.rss_mb,
            "vms_mb_after": mem_after.vms_mb,
            "uss_mb_after": mem_after.uss_mb,
            "rss_delta_mb": mem_after.rss_mb - mem_before.rss_mb,
            "vms_delta_mb": mem_after.vms_mb - mem_before.vms_mb,
            "uss_delta_mb": (mem_after.uss_mb - mem_before.uss_mb) if (mem_after.uss_mb is not None and mem_before.uss_mb is not None) else None,
            "infer_peaks": infer_peak,
            "pipeline_metrics": {
                "latency_ms": getattr(result.metrics, "latency_ms", None),
                "memory_peak_mb": getattr(result.metrics, "memory_peak_mb", None),
                "chunks_created": getattr(result.metrics, "chunks_created", None),
                "candidates_generated": getattr(result.metrics, "candidates_generated", None),
                "validation_pass_rate": getattr(result.metrics, "validation_pass_rate", None),
                "deduplication_reduction": getattr(result.metrics, "deduplication_reduction", None),
            },
            "questions_count": len(result.questions),
        })

        print("\n✓ Done")
        print(f"PID: {pid}")
        print(f"Infer latency: {(t_inf1 - t_inf0) * 1000.0:.1f}ms")
        print(f"RSS before/after: {mem_before.rss_mb:.1f}MB -> {mem_after.rss_mb:.1f}MB (Δ {mem_after.rss_mb - mem_before.rss_mb:.1f}MB)")
        print(f"RSS peak during infer: {infer_peak['rss_peak_mb']:.1f}MB")
        print(f"JSONL log: {args.out}")

        return 0

    except Exception as e:
        logger.write({"event": "error", "error": str(e)})
        raise
    finally:
        if adapter is not None:
            try:
                await adapter.shutdown()
            except Exception:
                pass
        logger.write({"event": "run_end"})


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
