#!/usr/bin/env python3
"""
Validate whether provided questions are *grounded follow-up questions* for an article
(using OpenAI gpt-4.1), and write a new CSV with a boolean column:
  grounded_followup

NEW BEHAVIOR (vs "answerable"):
- A question is considered valid if it is a natural follow-up grounded in the article's content.
- It does NOT need to be explicitly answerable from the article.
- It MUST relate to entities/topics in the article and not introduce unrelated assumptions.

EARLY STOP:
- Stop processing once we have found EARLY_STOP_TRUE_TARGET True rows.
- Remaining rows are left as missing (blank in output CSV).

Input CSV (same directory):
  extracted_dataset.csv
Columns used:
  - article
  - provided_questions (list of questions; JSON list, python list string, or delimited)
Ignores:
  - number_of_quesitons (or similar)

Output CSV:
  extracted_dataset_with_grounded_followup.csv

Env var required:
  OPENAI_API_KEY

Install:
  pip install openai pandas
"""

from __future__ import annotations

import os
import json
import ast
import time
from typing import Any, List, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("Missing dependency. Install with: pip install openai pandas") from e

load_dotenv()
INPUT_CSV = "extracted_dataset.csv"
OUTPUT_CSV = "extracted_dataset_with_grounded_followup.csv"
MODEL = "gpt-4.1"

# Early stop: stop once we have this many True rows
EARLY_STOP_TRUE_TARGET = 100

# Safety limits
MAX_ARTICLE_CHARS = 50_000
MAX_QUESTION_CHARS = 500


SYSTEM_PROMPT = """You are a strict validator.

Task:
Given an article and a list of questions, decide if EVERY question is a correct, natural follow-up
grounded in the article.

What "grounded follow-up" means:
- The question should naturally follow from the article's content (topics, entities, events, claims).
- It must NOT be unrelated to the article.
- It must NOT assume facts that are not introduced by the article.
- It can ask for clarification, details, implications, next steps, comparisons, pros/cons, etc.,
  as long as it stays within the scope set by the article.
- It does NOT need to be directly answerable from the article (it may request more details),
  but it must be motivated by what the article talks about.

Decision rule:
- Return true ONLY if all questions are grounded follow-ups.
- If ANY question is off-topic, speculative beyond the article, or introduces unsupported assumptions, return false.

Output rules:
- Output ONLY valid JSON.
- Schema:
  {"all_grounded_followups": true|false}
"""

USER_TEMPLATE = """ARTICLE:
{article}

QUESTIONS (JSON array of strings):
{questions_json}
"""


def parse_questions_cell(cell: Any) -> List[str]:
    """
    Parse provided_questions which may be:
    - a proper list
    - a JSON string list: ["q1","q2",...]
    - a python literal list string: ['q1','q2',...]
    - a delimited string (newline / semicolon / pipe)
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, list):
        return [str(q).strip() for q in cell if str(q).strip()]

    s = str(cell).strip()
    if not s:
        return []

    # Try JSON
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(q).strip() for q in v if str(q).strip()]
    except Exception:
        pass

    # Try python literal list
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(q).strip() for q in v if str(q).strip()]
    except Exception:
        pass

    # Fallback delimiters
    for delim in ["\n", ";", "|"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            return [p for p in parts if p]

    return [s]


def clamp_text(article: str, questions: List[str]) -> Tuple[str, List[str]]:
    a = (article or "").strip()
    if len(a) > MAX_ARTICLE_CHARS:
        a = a[:MAX_ARTICLE_CHARS] + "\n\n[TRUNCATED]"
    qs: List[str] = []
    for q in questions:
        q = (q or "").strip()
        if not q:
            continue
        if len(q) > MAX_QUESTION_CHARS:
            q = q[:MAX_QUESTION_CHARS] + " [TRUNCATED]"
        qs.append(q)
    return a, qs


def call_validator(client: OpenAI, article: str, questions: List[str], retries: int = 3) -> Optional[bool]:
    """
    Returns True/False if successfully validated, else None.
    """
    article, questions = clamp_text(article, questions)
    if not article or not questions:
        return False

    user_prompt = USER_TEMPLATE.format(
        article=article,
        questions_json=json.dumps(questions, ensure_ascii=False),
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = resp.choices[0].message.content or ""
            obj = json.loads(content)
            val = obj.get("all_grounded_followups", None)

            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("true", "false"):
                    return v == "true"

            raise ValueError(f"Unexpected JSON schema: {obj}")

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))

    print(f"[WARN] Validation failed after retries: {last_err}")
    return None


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment.")

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if "article" not in df.columns or "provided_questions" not in df.columns:
        raise SystemExit(
            "CSV must contain columns: 'article' and 'provided_questions'. "
            f"Found columns: {list(df.columns)}"
        )

    client = OpenAI(api_key=api_key)

    # Nullable boolean: True/False/NA. NA = not processed due to early stop.
    df["grounded_followup"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    true_count = 0
    total = len(df)

    for i, row in df.iterrows():
        if true_count >= EARLY_STOP_TRUE_TARGET:
            print(f"🛑 Early stop: reached {EARLY_STOP_TRUE_TARGET} True rows at index {i}.")
            break

        article = row.get("article", "")
        provided = row.get("provided_questions", "")
        questions = parse_questions_cell(provided)

        verdict = call_validator(client, str(article), questions)
        if verdict is None:
            verdict = False  # conservative on failures

        df.at[i, "grounded_followup"] = bool(verdict)

        if verdict:
            true_count += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processed {i+1}/{total} | true_count={true_count}")

    # Write CSV with blanks for unprocessed rows
    out_df = df.copy()
    out_df["grounded_followup"] = out_df["grounded_followup"].astype("object").where(
        out_df["grounded_followup"].notna(), ""
    )
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote: {OUTPUT_CSV} (true_count={true_count})")


if __name__ == "__main__":
    main()
