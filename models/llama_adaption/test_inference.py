#!/usr/bin/env python3
"""
Profiles llama-server load + inference and writes JSONL.

UPDATED (grounded follow-up CSV mode):
- Reads: extracted_dataset_with_grounded_followup.csv
- Only processes rows where grounded_followup == True
- Generates follow-up questions using your pipeline
- Generates N questions per row where N = len(provided_questions)
  (fallback to --gen_n if provided_questions missing/unparseable)
- Writes a NEW CSV with:
    - generated_question (JSON list string of whatever was produced)
    - generation_status (ok/partial/fail)
    - failure_reason (why N was not met)
    - parse_failures (JSON list of parse failures if any)

Important behavior change:
- If fewer than desired questions are produced, we still keep the generated ones
  and record *why* (pipeline error, JSON parse failures, empty outputs, too few chunks, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import psutil
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.llama_adaption.llama_server_adapter import create_adapter_with_server
from local_llm.pipeline import QuestionGenerationPipeline
from local_llm.pipeline_types import ArticleInput, GenerationConfig


# -----------------------------
# Logging / JSON extraction utils
# -----------------------------

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    m = _FENCED_JSON_RE.search(s)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def normalize_question_payload(payload: Any) -> Optional[str]:
    """
    Best-effort extractor for a question string.
    Returns None if it can't confidently extract a question.
    """
    if payload is None:
        return None

    if isinstance(payload, dict):
        q = payload.get("question")
        if isinstance(q, str):
            q = q.strip()
            return q if q else None
        return None

    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return None

        obj = extract_json_object(s)
        if isinstance(obj, dict) and isinstance(obj.get("question"), str):
            q = obj["question"].strip()
            return q if q else None

        # If model returned plain question text (no JSON), optionally accept it
        # (This helps when JSON parsing fails but content is still usable.)
        # Heuristic: must end with '?'
        if s.endswith("?") and len(s.split()) <= 30:
            return s

        return None

    s = str(payload).strip()
    if s.endswith("?") and len(s.split()) <= 30:
        return s
    return None


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger(__name__).setLevel(logging.INFO)

    if not verbose:
        logging.getLogger("models.llama_adaption.llama_server_adapter").setLevel(logging.WARNING)
        logging.getLogger("models.llama_adaption").setLevel(logging.WARNING)
        logging.getLogger("local_llm").setLevel(logging.WARNING)


# -----------------------------
# JSONL logger + memory helpers
# -----------------------------

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


# -----------------------------
# CSV helpers
# -----------------------------

_TRUE_STRS = {"true", "1", "yes", "y", "t"}


def is_true_cell(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in _TRUE_STRS


def parse_questions_cell(cell: Any) -> List[str]:
    """
    Parses provided_questions which may be:
    - list already
    - JSON list string: ["q1", ...]
    - python list string: ['q1', ...]
    - delimited string (newline / ; / |)
    """
    if cell is None:
        return []
    try:
        if pd.isna(cell):
            return []
    except Exception:
        pass

    if isinstance(cell, list):
        return [str(q).strip() for q in cell if str(q).strip()]

    s = str(cell).strip()
    if not s:
        return []

    # JSON
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(q).strip() for q in v if str(q).strip()]
    except Exception:
        pass

    # python literal
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(q).strip() for q in v if str(q).strip()]
    except Exception:
        pass

    # delimiters
    for delim in ["\n", ";", "|"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            return [p for p in parts if p]

    return [s]


def load_grounded_true_rows(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(p)

    if "article" not in df.columns:
        raise ValueError(f"Missing required column 'article'. Found: {list(df.columns)}")
    if "grounded_followup" not in df.columns:
        raise ValueError("Missing required column 'grounded_followup'.")

    mask = df["grounded_followup"].apply(is_true_cell)
    df_true = df[mask].copy()

    if limit is not None:
        df_true = df_true.head(limit).copy()

    return df_true


# -----------------------------
# Helpers to run pipeline + failure reasoning
# -----------------------------

def infer_failure_reason(
    desired_n: int,
    got_n: int,
    err: Optional[str],
    parse_failures: List[Dict[str, Any]],
    result: Optional[Any],
) -> str:
    """
    Classify why we did not get desired_n questions.
    """
    if desired_n <= 0:
        return "invalid_desired_n"

    if err:
        return f"pipeline_error: {err}"

    if got_n >= desired_n:
        return ""

    # No exception, but got too few
    if result is None:
        return "no_result_returned"

    total_candidates = None
    try:
        total_candidates = len(getattr(result, "questions", []) or [])
    except Exception:
        total_candidates = None

    if total_candidates == 0:
        return "empty_output_or_no_candidates"

    if parse_failures and got_n == 0:
        return "json_parse_failed_all_candidates"

    if parse_failures and got_n < desired_n:
        return f"json_parse_failed_some_candidates ({len(parse_failures)} failed)"

    # Likely chunking/selection limited generation before post-processing
    # This happens when your pre-gen produces fewer chunks than desired questions.
    if total_candidates is not None and total_candidates < desired_n:
        return f"too_few_candidates_generated ({total_candidates} < {desired_n})"

    return "insufficient_valid_candidates_after_postprocessing"


async def run_and_clean(
    pipeline: QuestionGenerationPipeline,
    article_text: str,
    desired_questions: int,
) -> Tuple[Optional[Any], Optional[str], List[str], List[Dict[str, Any]], float]:
    """
    Returns: (result, error_str, questions_clean, parse_failures, latency_ms)

    Note: questions_clean includes only strings we could extract confidently.
    """
    sample_article = ArticleInput(text=article_text, desired_questions=int(desired_questions))

    t0 = time.perf_counter()
    result = None
    err = None
    try:
        result = await pipeline.generate_with_metrics(sample_article)
    except Exception as e:
        err = str(e)
    t1 = time.perf_counter()

    questions_clean: List[str] = []
    parse_failures: List[Dict[str, Any]] = []

    if result is not None:
        raw_questions = getattr(result, "questions", []) or []
        for i, q in enumerate(raw_questions):
            if hasattr(q, "question"):
                q_text = normalize_question_payload(getattr(q, "question"))
            elif isinstance(q, dict):
                q_text = normalize_question_payload(q)
            elif isinstance(q, str):
                q_text = normalize_question_payload(q)
            else:
                q_text = normalize_question_payload(str(q))

            if q_text:
                questions_clean.append(q_text)
            else:
                parse_failures.append({"index": i, "raw": str(q)[:500]})

    return result, err, questions_clean, parse_failures, (t1 - t0) * 1000.0


# -----------------------------
# Main
# -----------------------------

async def main() -> int:
    ap = argparse.ArgumentParser()

    # JSONL profiling output
    ap.add_argument("--out", default="profiles/llama_profile.jsonl")

    # CSV input/output
    ap.add_argument("--csv", type=str, default="extracted_dataset_with_grounded_followup.csv")
    ap.add_argument("--out_csv", type=str, default="extracted_dataset_with_generated_questions.csv")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of TRUE rows processed")

    # llama-server config
    ap.add_argument("--ctx", type=int, default=1048)
    ap.add_argument("--max_out", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail_fast", action="store_true")

    # Fallback generation behavior if provided_questions missing
    ap.add_argument(
        "--gen_n",
        type=int,
        default=4,
        help="Fallback number of follow-up questions to generate if provided_questions is missing/unparseable.",
    )

    args = ap.parse_args()
    configure_logging(verbose=args.verbose)
    logger = JsonlLogger(args.out)

    MODEL_PATH = "models/llama_adaption/models/LiquidAI_LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf"
    LLAMA_SERVER_PATH = "models/llama.cpp/build/bin/llama-server"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return 1

    # Load only TRUE grounded_followup rows
    try:
        df_true = load_grounded_true_rows(args.csv, limit=args.limit)
    except Exception as e:
        print(f"❌ Failed reading grounded-followup CSV: {e}")
        return 1

    if df_true.empty:
        print("❌ No rows with grounded_followup == True found.")
        return 1

    adapter = None
    try:
        logger.write({
            "event": "run_start",
            "mode": "grounded_followup_true_only",
            "csv_path": args.csv,
            "true_rows": int(len(df_true)),
            "model_path": MODEL_PATH,
            "ctx_size": args.ctx,
            "max_output_tokens": args.max_out,
            "threads": args.threads,
            "port": args.port,
            "verbose": args.verbose,
            "fallback_gen_n": args.gen_n,
        })

        # Start server once
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

        # Conservative decoding for small models
        config = GenerationConfig(max_output_tokens=args.max_out, temperature=0.2, top_p=0.9)
        pipeline = QuestionGenerationPipeline(adapter=adapter, config=config)

        # Prepare output columns
        df_true = df_true.copy()
        df_true["generated_question"] = ""     # JSON list string (store what we got)
        df_true["generated_n"] = ""            # requested per-row
        df_true["got_n"] = ""                  # actually extracted
        df_true["generation_status"] = ""      # ok/partial/fail
        df_true["failure_reason"] = ""         # why not equal
        df_true["parse_failures"] = ""         # JSON list string
        df_true["gen_latency_ms"] = ""
        df_true["gen_error"] = ""
        df_true["rss_mb_before"] = ""
        df_true["rss_mb_after"] = ""
        df_true["rss_delta_mb"] = ""

        latencies: List[float] = []

        print(f"\nProcessing grounded_followup==True rows: {len(df_true)}")
        print(f"PID: {pid}")
        print("-" * 60)

        for j, (row_index, row) in enumerate(df_true.iterrows(), start=1):
            article_text = str(row.get("article", "")).strip()
            if not article_text:
                df_true.at[row_index, "generated_question"] = json.dumps([], ensure_ascii=False)
                df_true.at[row_index, "generated_n"] = "0"
                df_true.at[row_index, "got_n"] = "0"
                df_true.at[row_index, "generation_status"] = "fail"
                df_true.at[row_index, "failure_reason"] = "empty_article"
                df_true.at[row_index, "parse_failures"] = json.dumps([], ensure_ascii=False)
                df_true.at[row_index, "gen_error"] = "empty_article"
                continue

            # Determine per-row desired_n from provided_questions length (fallback to --gen_n)
            desired_n = None
            provided_list: List[str] = []
            if "provided_questions" in df_true.columns:
                provided_list = parse_questions_cell(row.get("provided_questions"))
                if provided_list:
                    desired_n = len(provided_list)

            if desired_n is None or desired_n <= 0:
                desired_n = max(1, int(args.gen_n))

            mem_before = read_process_memory(pid)

            result, err, questions_clean, parse_failures, latency_ms = await run_and_clean(
                pipeline=pipeline,
                article_text=article_text,
                desired_questions=desired_n,
            )

            mem_after = read_process_memory(pid)
            latencies.append(latency_ms)

            got_n = len(questions_clean)

            # Status + failure reasoning
            if err is not None:
                status = "fail"
            elif got_n == 0:
                status = "fail"
            elif got_n < desired_n:
                status = "partial"
            else:
                status = "ok"

            reason = infer_failure_reason(
                desired_n=desired_n,
                got_n=got_n,
                err=err,
                parse_failures=parse_failures,
                result=result,
            )

            # Always store what we got
            df_true.at[row_index, "generated_question"] = json.dumps(questions_clean, ensure_ascii=False)
            df_true.at[row_index, "generated_n"] = str(desired_n)
            df_true.at[row_index, "got_n"] = str(got_n)
            df_true.at[row_index, "generation_status"] = status
            df_true.at[row_index, "failure_reason"] = reason
            df_true.at[row_index, "parse_failures"] = json.dumps(parse_failures, ensure_ascii=False)
            df_true.at[row_index, "gen_latency_ms"] = f"{latency_ms:.3f}"
            df_true.at[row_index, "gen_error"] = err or ""
            df_true.at[row_index, "rss_mb_before"] = f"{mem_before.rss_mb:.3f}"
            df_true.at[row_index, "rss_mb_after"] = f"{mem_after.rss_mb:.3f}"
            df_true.at[row_index, "rss_delta_mb"] = f"{(mem_after.rss_mb - mem_before.rss_mb):.3f}"

            ok = (status == "ok")

            logger.write({
                "event": "csv_row_result",
                "row_num": j,
                "source_row_index": int(row_index),

                "expected_questions": desired_n,
                "got_questions": got_n,
                "status": status,
                "failure_reason": reason,
                "error": err,

                "latency_ms": latency_ms,
                "questions_clean": questions_clean,
                "question_parse_failures": parse_failures,

                "rss_mb_before": mem_before.rss_mb,
                "rss_mb_after": mem_after.rss_mb,
                "rss_delta_mb": mem_after.rss_mb - mem_before.rss_mb,
            })

            if ok:
                print(f"[{j:03d}/{len(df_true):03d}] ✅ OK      latency={latency_ms:.0f}ms  got={got_n}/{desired_n}")
            elif status == "partial":
                print(f"[{j:03d}/{len(df_true):03d}] ⚠️ PARTIAL latency={latency_ms:.0f}ms  got={got_n}/{desired_n}  reason={reason}")
            else:
                print(f"[{j:03d}/{len(df_true):03d}] ❌ FAIL    latency={latency_ms:.0f}ms  got={got_n}/{desired_n}  reason={reason}")

            if (not ok) and args.fail_fast:
                print("Fail-fast enabled. Stopping.")
                break

        # Write output CSV
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_true.to_csv(out_path, index=False)
        print("-" * 60)
        print(f"✅ Wrote CSV: {out_path}")

        # Summary
        latencies = [x for x in latencies if isinstance(x, (int, float))]
        if latencies:
            s = sorted(latencies)
            p50 = s[int(0.50 * (len(s) - 1))]
            p95 = s[int(0.95 * (len(s) - 1))]
            avg = sum(latencies) / len(latencies)
        else:
            avg = p50 = p95 = None

        # Count statuses
        status_counts = df_true["generation_status"].value_counts(dropna=False).to_dict()

        summary = {
            "event": "run_summary",
            "processed_true_rows": int(len(df_true)),
            "status_counts": status_counts,
            "latency_ms_avg": avg,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "out_csv": str(out_path),
            "jsonl_log": args.out,
        }
        logger.write(summary)

        if avg is not None:
            print(f"Latency: avg={avg:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms")
        print(f"Status counts: {status_counts}")
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
