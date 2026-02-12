#!/usr/bin/env python3
"""
Analyze batch.jsonl and generate graphs for latency + memory.

Usage:
  python analyze_batch_jsonl.py --in profiles/batch.jsonl --outdir profiles/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # skip malformed line but keep a breadcrumb
                rows.append({"event": "parse_error", "line_no": line_no, "error": str(e)})
    return rows


def quantile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    i = int(q * (len(sorted_vals) - 1))
    return sorted_vals[i]


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def collect_metrics(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Returns lists aligned by row order (sorted by row_id when available).
    """
    samples = [r for r in rows if r.get("event") == "csv_row_result"]

    # Sort by row_id if present
    def key_fn(r: Dict[str, Any]) -> Tuple[int, int]:
        rid = r.get("row_id")
        try:
            return (0, int(rid))
        except Exception:
            return (1, 10**9)

    samples.sort(key=key_fn)

    row_ids: List[float] = []
    latency_one: List[float] = []
    rss_before: List[float] = []
    rss_after: List[float] = []
    rss_delta: List[float] = []
    passed: List[float] = []

    for r in samples:
        rid = r.get("row_id")
        try:
            row_ids.append(float(int(rid)))
        except Exception:
            row_ids.append(float(len(row_ids) + 1))

        latency_one.append(safe_float(r.get("latency_one_q_ms")) or float("nan"))
        rss_before.append(safe_float(r.get("rss_mb_before")) or float("nan"))
        rss_after.append(safe_float(r.get("rss_mb_after")) or float("nan"))
        rss_delta.append(safe_float(r.get("rss_delta_mb")) or float("nan"))

        passed.append(1.0 if r.get("pass") is True else 0.0)

    return {
        "row_id": row_ids,
        "latency_one_q_ms": latency_one,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_delta_mb": rss_delta,
        "pass": passed,
    }


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_latency_trend(row_id: List[float], latency: List[float], out: Path) -> None:
    plt.figure()
    plt.plot(row_id, latency, marker="o", linewidth=1)
    plt.xlabel("Row ID")
    plt.ylabel("Latency (1-question) ms")
    plt.title("Latency per row (1-question request)")
    save_fig(out)


def plot_latency_hist(latency: List[float], out: Path) -> None:
    clean = [x for x in latency if x == x]  # drop NaN
    plt.figure()
    plt.hist(clean, bins=20)
    plt.xlabel("Latency (1-question) ms")
    plt.ylabel("Count")
    plt.title("Latency distribution (1-question request)")
    save_fig(out)


def plot_latency_box(latency: List[float], out: Path) -> None:
    clean = [x for x in latency if x == x]
    plt.figure()
    plt.boxplot(clean, vert=True)
    plt.ylabel("Latency (1-question) ms")
    plt.title("Latency boxplot (1-question request)")
    save_fig(out)


def plot_rss_before_after(row_id: List[float], rss_before: List[float], rss_after: List[float], out: Path) -> None:
    plt.figure()
    plt.plot(row_id, rss_before, marker="o", linewidth=1, label="RSS before")
    plt.plot(row_id, rss_after, marker="o", linewidth=1, label="RSS after")
    plt.xlabel("Row ID")
    plt.ylabel("RSS (MB)")
    plt.title("RSS before vs after (per row)")
    plt.legend()
    save_fig(out)


def plot_rss_delta_trend(row_id: List[float], rss_delta: List[float], out: Path) -> None:
    plt.figure()
    plt.plot(row_id, rss_delta, marker="o", linewidth=1)
    plt.xlabel("Row ID")
    plt.ylabel("RSS Δ (MB)")
    plt.title("RSS delta per row (after - before)")
    save_fig(out)


def plot_rss_delta_hist(rss_delta: List[float], out: Path) -> None:
    clean = [x for x in rss_delta if x == x]
    plt.figure()
    plt.hist(clean, bins=20)
    plt.xlabel("RSS Δ (MB)")
    plt.ylabel("Count")
    plt.title("RSS delta distribution")
    save_fig(out)


def plot_latency_vs_rss_delta(latency: List[float], rss_delta: List[float], out: Path) -> None:
    xs, ys = [], []
    for x, y in zip(latency, rss_delta):
        if (x == x) and (y == y):  # not NaN
            xs.append(x)
            ys.append(y)

    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Latency (1-question) ms")
    plt.ylabel("RSS Δ (MB)")
    plt.title("Latency vs RSS delta")
    save_fig(out)


def print_summary(latency: List[float], rss_delta: List[float]) -> None:
    lat = sorted([x for x in latency if x == x])
    rd = sorted([x for x in rss_delta if x == x])

    def fmt(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    print("\nSummary:")
    print(f"Latency (ms):  count={len(lat)}  p50={fmt(quantile(lat, 0.50))}  p95={fmt(quantile(lat, 0.95))}  max={fmt(quantile(lat, 1.0))}")
    print(f"RSS Δ (MB):    count={len(rd)}  p50={fmt(quantile(rd, 0.50))}  p95={fmt(quantile(rd, 0.95))}  max={fmt(quantile(rd, 1.0))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to batch.jsonl")
    ap.add_argument("--outdir", required=True, help="Directory to write PNGs")
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    m = collect_metrics(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Latency plots
    plot_latency_trend(m["row_id"], m["latency_one_q_ms"], outdir / "latency_one_q_trend.png")
    plot_latency_hist(m["latency_one_q_ms"], outdir / "latency_one_q_hist.png")
    plot_latency_box(m["latency_one_q_ms"], outdir / "latency_one_q_box.png")

    # Memory plots
    plot_rss_before_after(m["row_id"], m["rss_mb_before"], m["rss_mb_after"], outdir / "rss_before_after.png")
    plot_rss_delta_trend(m["row_id"], m["rss_delta_mb"], outdir / "rss_delta_trend.png")
    plot_rss_delta_hist(m["rss_delta_mb"], outdir / "rss_delta_hist.png")

    # Relationship plot
    plot_latency_vs_rss_delta(m["latency_one_q_ms"], m["rss_delta_mb"], outdir / "latency_vs_rss_delta.png")

    print(f"\nWrote plots to: {outdir.resolve()}")
    print_summary(m["latency_one_q_ms"], m["rss_delta_mb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
