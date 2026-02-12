#!/usr/bin/env python3
"""
Extract article + questions from a single 'response' column.

Input:  reference_dataset.csv  (must have a column named 'response')
Output: extracted_dataset.csv  with columns:
        - article
        - provided_questions
        - number_of_questions

Assumptions (based on your example):
- response is a string that looks like a Python dict:
  {'status': 'success', 'response': {'type': ..., 'suggestions': [...], 'llm_message': '...', 'message': '...'}}
- questions are in response['response']['suggestions']
- article is in response['response']['llm_message']
  - if llm_message is empty, fallback to response['response']['message']
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def parse_response_cell(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a single cell from the 'response' column.

    Handles:
    - Python dict-like strings via ast.literal_eval
    - JSON strings via json.loads (fallback)
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    # Most likely: Python dict representation with single quotes
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: JSON (if sometimes stored as valid JSON)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def extract_fields(parsed: Dict[str, Any]) -> Tuple[str, List[str], int]:
    """
    Extract (article, suggestions, num_suggestions) from parsed dict.
    """
    inner = parsed.get("response") or {}
    if not isinstance(inner, dict):
        inner = {}

    suggestions = inner.get("suggestions") or []
    if not isinstance(suggestions, list):
        suggestions = []

    # Article content: llm_message preferred; fallback to message if empty
    llm_message = inner.get("llm_message")
    message = inner.get("message")

    article = ""
    if isinstance(llm_message, str) and llm_message.strip():
        article = llm_message.strip()
    elif isinstance(message, str) and message.strip():
        article = message.strip()

    # Normalize suggestions to list[str]
    norm_suggestions: List[str] = []
    for q in suggestions:
        if isinstance(q, str) and q.strip():
            norm_suggestions.append(q.strip())
        else:
            # keep non-string items as JSON if they appear
            try:
                norm_suggestions.append(json.dumps(q, ensure_ascii=False))
            except Exception:
                norm_suggestions.append(str(q))

    return article, norm_suggestions, len(norm_suggestions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="reference_dataset.csv",
                    help="Input CSV path (default: reference_dataset.csv)")
    ap.add_argument("--out", dest="out", default="extracted_dataset.csv",
                    help="Output CSV path (default: extracted_dataset.csv)")
    ap.add_argument("--response-col", dest="response_col", default="response",
                    help="Name of column containing the raw response (default: response)")
    ap.add_argument("--join-questions", action="store_true",
                    help="If set, store provided_questions as a single string joined by ' ||| ' instead of JSON list")

    args = ap.parse_args()

    inp_path = Path(args.inp)
    if not inp_path.exists():
        print(f"ERROR: input file not found: {inp_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(inp_path)

    if args.response_col not in df.columns:
        print(f"ERROR: column '{args.response_col}' not found in {inp_path}. "
              f"Found columns: {list(df.columns)}", file=sys.stderr)
        return 2

    articles: List[str] = []
    questions_out: List[Any] = []
    counts: List[int] = []

    for raw in df[args.response_col].tolist():
        parsed = parse_response_cell(raw)
        if not parsed:
            articles.append("")
            questions_out.append([] if not args.join_questions else "")
            counts.append(0)
            continue

        article, qs, n = extract_fields(parsed)
        articles.append(article)

        if args.join_questions:
            questions_out.append(" ||| ".join(qs))
        else:
            # Store as JSON list string so it round-trips cleanly in CSV
            questions_out.append(json.dumps(qs, ensure_ascii=False))

        counts.append(n)

    out_df = pd.DataFrame({
        "article": articles,
        "provided_questions": questions_out,
        "number_of_questions": counts
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}  (rows={len(out_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
