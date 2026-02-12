#!/usr/bin/env python3
"""
Extract article + follow-up questions from ALL files in a directory (even if no extension).

Input dir: iab-data-training/
Each file contains JSON objects one per line (JSONL).
Each line has a "response" field that is a *stringified Python dict*:
  "{'status': 'success', 'response': {'suggestions': [...], 'message': '...'}}"

Output CSV columns:
  - article
  - follow_up_questions
  - follow_up_question_count

Usage:
  python extract_iab_followups_dir.py --dir iab-data-training --out iab_followups.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_inner_response(inner_raw: Any) -> Optional[Dict[str, Any]]:
    """
    inner_raw is expected to be a string that looks like a Python dict.
    Try ast.literal_eval first, then JSON loads as fallback.
    """
    if inner_raw is None:
        return None
    s = str(inner_raw).strip()
    if not s:
        return None

    # Common case: python-literal string with single quotes
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: sometimes it might be JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def extract_article_and_questions(outer_obj: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Returns (article_text, suggestions_list) from a single outer JSON object.
    Article field preference: message first, then llm_message.
    """
    inner = parse_inner_response(outer_obj.get("response"))
    if not inner:
        return "", []

    inner_payload = inner.get("response") or {}
    if not isinstance(inner_payload, dict):
        inner_payload = {}

    suggestions = inner_payload.get("suggestions") or []
    if not isinstance(suggestions, list):
        suggestions = []

    message = inner_payload.get("message")
    llm_message = inner_payload.get("llm_message")

    article = ""
    if isinstance(message, str) and message.strip():
        article = message.strip()
    elif isinstance(llm_message, str) and llm_message.strip():
        article = llm_message.strip()

    qs: List[str] = []
    for q in suggestions:
        if isinstance(q, str) and q.strip():
            qs.append(q.strip())
        else:
            qs.append(str(q))

    return article, qs


def list_all_files(dir_path: Path) -> List[Path]:
    """
    Return all files in dir_path (non-recursive), regardless of extension.
    """
    return sorted([p for p in dir_path.iterdir() if p.is_file()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", dest="dir_path", default="iab-data-training",
                    help="Directory containing input files (default: iab-data-training)")
    ap.add_argument("--out", dest="out", default="iab_followups.csv",
                    help="Output CSV path (default: iab_followups.csv)")
    ap.add_argument("--questions-format", choices=["json", "join"], default="json",
                    help="Store follow_up_questions as JSON array string (default) or joined string")
    ap.add_argument("--recursive", action="store_true",
                    help="Also read files in subdirectories recursively")
    ap.add_argument("--max-lines-per-file", type=int, default=0,
                    help="If >0, process only first N lines per file (debug)")
    args = ap.parse_args()

    dir_path = Path(args.dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"ERROR: directory not found: {dir_path}", file=sys.stderr)
        return 2

    if args.recursive:
        files = sorted([p for p in dir_path.rglob("*") if p.is_file()])
    else:
        files = list_all_files(dir_path)

    if not files:
        print(f"ERROR: no files found in {dir_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_bad_outer_json = 0
    total_empty_inner = 0

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["article", "follow_up_questions", "follow_up_question_count"]
        )
        writer.writeheader()

        for fp in files:
            rows = 0
            bad_outer_json = 0
            empty_inner = 0

            # Some BigQuery exports can be large; read line-by-line
            with fp.open("r", encoding="utf-8", errors="replace") as f_in:
                for idx, line in enumerate(f_in, start=1):
                    if args.max_lines_per_file and idx > args.max_lines_per_file:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    outer = safe_json_loads(line)
                    if not outer:
                        bad_outer_json += 1
                        continue

                    article, qs = extract_article_and_questions(outer)
                    if not article and not qs:
                        empty_inner += 1

                    if args.questions_format == "join":
                        qs_cell = " ||| ".join(qs)
                    else:
                        qs_cell = json.dumps(qs, ensure_ascii=False)

                    writer.writerow({
                        "article": article,
                        "follow_up_questions": qs_cell,
                        "follow_up_question_count": len(qs),
                    })
                    rows += 1

            print(f"[{fp.name}] rows={rows} bad_outer_json={bad_outer_json} empty_inner={empty_inner}")
            total_rows += rows
            total_bad_outer_json += bad_outer_json
            total_empty_inner += empty_inner

    print(
        f"\nDONE\n"
        f"Output: {out_path}\n"
        f"Total rows written: {total_rows}\n"
        f"Total skipped (bad outer JSON lines): {total_bad_outer_json}\n"
        f"Total empty inner parses: {total_empty_inner}\n"
        f"Files processed: {len(files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
