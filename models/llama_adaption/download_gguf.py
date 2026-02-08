#!/usr/bin/env python3
"""
download_gguf.py

Model-agnostic GGUF downloader:
- Lists repo files first
- Finds all .gguf files
- Picks the best match for requested quant (e.g., Q4_K_M, Q8_0, F16)
- Downloads into ./models/<org_repo>/...

Usage:
  python download_gguf.py --model_id tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF --quant_type Q4_K_M
  python download_gguf.py --model_id LiquidAI/LFM2-350M-GGUF --quant_type Q4_K_M
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download, list_repo_files


KNOWN_QUANTS = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "F16", "BF16",
]


def _norm_quant(q: str) -> str:
    return q.strip().upper().replace("-", "_")


def _extract_quant_from_filename(fname: str) -> Optional[str]:
    u = fname.upper()
    for q in KNOWN_QUANTS:
        if q in u:
            return q
    return None


def _score_candidate(fname: str, target_quant: str) -> int:
    """
    Higher score is better.
    - Exact quant match wins
    - Prefer shorter filenames (less weird variants)
    - Prefer files in root (no subdirs)
    """
    score = 0
    uq = target_quant.upper()

    f_up = fname.upper()
    q_in_name = _extract_quant_from_filename(fname)

    if q_in_name == uq:
        score += 1000
    elif q_in_name is None:
        score += 0
    else:
        # mild preference for "close" quants if exact not found
        # (e.g., prefer Q5_K_M when user asked Q4_K_M? not really)
        score -= 50

    # Prefer root files (no "/")
    if "/" not in fname:
        score += 20

    # Prefer filenames that end with .gguf (obviously) and are not huge multi-part names
    score -= min(len(fname), 300) // 10  # small penalty for very long names

    # Prefer ones that include "instruct"/"chat" if multiple collide (helpful for your usage)
    if "INSTRUCT" in f_up or "CHAT" in f_up:
        score += 10

    return score


def pick_gguf_file(files: List[str], quant_type: str) -> Tuple[Optional[str], List[str]]:
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        return None, []

    target = _norm_quant(quant_type)
    # First pass: exact quant match
    exact = [f for f in ggufs if _extract_quant_from_filename(f) == target]
    if exact:
        best = max(exact, key=lambda f: _score_candidate(f, target))
        return best, ggufs

    # Second pass: try relaxed matching like "Q4KM" or "Q4-K-M" variations
    relaxed = re.sub(r"[^A-Z0-9]", "", target)  # Q4_K_M -> Q4KM
    rel = []
    for f in ggufs:
        compact = re.sub(r"[^A-Z0-9]", "", f.upper())
        if relaxed in compact:
            rel.append(f)
    if rel:
        best = max(rel, key=lambda f: _score_candidate(f, target))
        return best, ggufs

    # Otherwise: no match; return None but provide list
    return None, ggufs


class GGUFDownloader:
    def __init__(self, output_dir: str = "./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_id: str, quant_type: str = "Q8_0") -> Path:
        print(f"\n{'='*70}")
        print(f"Repo:  {model_id}")
        print(f"Quant: {_norm_quant(quant_type)}")
        print(f"{'='*70}")

        try:
            files = list(list_repo_files(model_id))
        except Exception as e:
            print(f"[error] Could not list repo files for {model_id}: {e}")
            print("Tip: ensure the repo exists and you have access. For gated models, login with huggingface-cli.")
            sys.exit(1)

        chosen, gguf_files = pick_gguf_file(files, quant_type)
        if not gguf_files:
            print(f"[error] No .gguf files found in {model_id}.")
            sys.exit(1)

        if not chosen:
            print(f"[error] No GGUF file matched quant={_norm_quant(quant_type)}.")
            print("\nAvailable GGUF files:")
            for f in gguf_files:
                print(f"  - {f}")
            print("\nTry one of these quants that appear in filenames:")
            quants = sorted({q for q in (_extract_quant_from_filename(f) for f in gguf_files) if q})
            if quants:
                for q in quants:
                    print(f"  python download_gguf.py --model_id {model_id} --quant_type {q}")
            sys.exit(1)

        # Download destination folder
        model_dir = self.output_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[info] Selected: {chosen}")
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=chosen,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        p = Path(downloaded_path)
        size_mb = p.stat().st_size / (1024**2)

        print(f"\n✅ Download complete: {p.name}")
        print(f"Path: {p}")
        print(f"Size: {size_mb:.2f} MB")
        return p


def main():
    ap = argparse.ArgumentParser(description="Model-agnostic GGUF downloader from Hugging Face")
    ap.add_argument("--model_id", required=True, help="Hugging Face repo id containing GGUF files")
    ap.add_argument("--quant_type", default="Q8_0", help="Quantization type (e.g., Q4_K_M, Q5_K_M, Q8_0, F16)")
    ap.add_argument("--output_dir", default="./models", help="Output directory")
    args = ap.parse_args()

    d = GGUFDownloader(output_dir=args.output_dir)
    d.download(args.model_id, args.quant_type)


if __name__ == "__main__":
    main()
