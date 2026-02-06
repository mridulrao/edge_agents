#!/usr/bin/env python3
"""
export_smollm2_to_onnx.py

Export HuggingFaceTB/SmolLM2-360M-Instruct to ONNX for browser usage (onnxruntime-web + WebGPU),
optionally applying post-export quantization.

What it does:
- Exports a decoder-only text-generation model with KV cache ("text-generation-with-past") via Optimum
- Saves tokenizer + config into the same output directory
- Optionally quantizes ONNX weights (INT8 / dynamic INT8 / fp16)

Usage:
  # FP32 export (default)
  python export_smollm2_to_onnx.py --out ./smollm2_onnx

  # Dynamic INT8 (recommended for most CPU use-cases)
  python export_smollm2_to_onnx.py --out ./smollm2_onnx --quantize int8-dynamic

  # Static INT8 (needs calibration data)
  python export_smollm2_to_onnx.py --out ./smollm2_onnx --quantize int8-static --calib-text-file calib.txt

  # FP16 weights (often good for WebGPU)
  python export_smollm2_to_onnx.py --out ./smollm2_onnx --quantize fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def _die(msg: str) -> None:
    raise SystemExit(f"[export] {msg}")


# ----------------------------
# Quantization helpers
# ----------------------------
def _collect_onnx_files(out_dir: Path) -> List[Path]:
    # Keep it flexible: some exporters put models in out_dir, some in out_dir/onnx
    candidates = []
    candidates += sorted(out_dir.glob("*.onnx"))
    candidates += sorted((out_dir / "onnx").glob("*.onnx")) if (out_dir / "onnx").exists() else []
    return candidates


def _quantize_dynamic_int8(model_path: Path, out_path: Path, per_channel: bool, reduce_range: bool) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        _die(
            f"Missing onnxruntime quantization tools. Install: pip install -U onnxruntime onnxruntime-tools. Details: {e}"
        )

    print(f"[quant] dynamic int8 -> {out_path.name}")
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,   # signed int8 weights
        per_channel=per_channel,
        reduce_range=reduce_range,
    )


def _quantize_static_int8(
    model_path: Path,
    out_path: Path,
    calib_texts: List[str],
    model_id: str,
    trust_remote_code: bool,
    max_length: int,
    num_samples: int,
) -> None:
    """
    Static INT8 requires calibration with representative inputs.

    For decoder-only LMs, calibrating *all* dynamic inputs robustly is tricky; this
    is a best-effort baseline that feeds tokenized inputs.
    """
    try:
        import numpy as np
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantFormat,
            QuantType,
        )
    except Exception as e:
        _die(
            f"Missing onnxruntime quantization tools. Install: pip install -U onnxruntime onnxruntime-tools numpy. Details: {e}"
        )

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        _die(f"Missing transformers. Install: pip install -U transformers. Details: {e}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    # Many decoder-only exports expect input_ids + attention_mask; some include position_ids.
    # We'll provide input_ids + attention_mask. If your exported graph requires more, ORT will error.
    class _TextCalibReader(CalibrationDataReader):
        def __init__(self, texts: List[str]):
            self.texts = texts[:num_samples]
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.texts):
                return None
            t = self.texts[self.idx]
            self.idx += 1
            enc = tok(
                t,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            # Ensure int64 dtype for ids/masks (common for ONNX NLP models)
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            return feeds

    reader = _TextCalibReader(calib_texts)

    print(f"[quant] static int8 (calibration) -> {out_path.name}")
    quantize_static(
        model_input=str(model_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )


def _convert_fp16(model_path: Path, out_path: Path) -> None:
    """
    Convert ONNX initializers to FP16 (weights). Keeps model ops in FP16 where possible.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        _die(
            f"Missing fp16 conversion deps. Install: pip install -U onnx onnxconverter-common. Details: {e}"
        )

    print(f"[quant] fp16 weights -> {out_path.name}")
    m = onnx.load(str(model_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct", help="HF model id or local path")
    ap.add_argument("--out", default="./onnx_adaption_web/smollm2_onnx", help="Output directory")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (use 17 by default for broad compatibility)")
    ap.add_argument(
        "--task",
        default="text-generation-with-past",
        choices=["text-generation", "text-generation-with-past"],
        help="text-generation-with-past enables KV cache (recommended).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device (cpu recommended for portability)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True if the model requires it.")
    ap.add_argument("--use-external-data-format", action="store_true", help="Store large weights in external data file(s) if needed.")

    # NEW: quantization
    ap.add_argument(
        "--quantize",
        default="none",
        choices=["none", "int8-dynamic", "int8-static", "fp16"],
        help="Post-export quantization. "
             "int8-dynamic is easiest; int8-static needs calibration data; fp16 is often best for WebGPU.",
    )
    ap.add_argument(
        "--quantize-output-suffix",
        default=None,
        help="Optional suffix for quantized model files (default based on --quantize).",
    )
    ap.add_argument("--per-channel", action="store_true", help="(int8-dynamic) Use per-channel weight quantization.")
    ap.add_argument("--reduce-range", action="store_true", help="(int8-dynamic) Reduce quant range for some CPUs/backends.")

    # Static INT8 calibration inputs
    ap.add_argument("--calib-text-file", default=None, help="(int8-static) Path to a text file with one calibration prompt per line.")
    ap.add_argument("--calib-max-length", type=int, default=128, help="(int8-static) Token length for calibration.")
    ap.add_argument("--calib-num-samples", type=int, default=64, help="(int8-static) Number of calibration lines to use.")

    # Optionally keep originals
    ap.add_argument("--keep-fp32", action="store_true", help="Keep the original FP32 ONNX files alongside quantized ones.")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save tokenizer + config alongside ONNX (useful for your web app).
    try:
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    except Exception as e:
        _die(f"Missing transformers. Install: pip install -U transformers. Details: {e}")

    print("[export] downloading/saving tokenizer + config...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tok.save_pretrained(out_dir)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    cfg.save_pretrained(out_dir)

    # generation_config (best-effort)
    gen_cfg = out_dir / "generation_config.json"
    if not gen_cfg.exists():
        try:
            gcfg = GenerationConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
            gcfg.save_pretrained(out_dir)
            print("[export] saved generation_config.json")
        except Exception:
            pass

    # 2) Export to ONNX via Optimum exporter.
    try:
        from optimum.exporters.onnx import main_export
    except Exception as e:
        _die(f"Missing optimum exporter. Install: pip install -U 'optimum[onnxruntime]'. Details: {e}")

    print(f"[export] exporting ONNX (task={args.task}, opset={args.opset}) -> {out_dir}")
    export_dtype = None
    if args.quantize == "fp16":
        export_dtype = "fp16"   # Export FP16 directly (recommended)

    main_export(
        model_name_or_path=args.model,
        output=out_dir,
        task=args.task,
        opset=args.opset,
        device=args.device,
        dtype=export_dtype,  
        trust_remote_code=args.trust_remote_code,
        use_external_data_format=args.use_external_data_format,
    )


    # 3) Summarize produced ONNX files
    onnx_files = _collect_onnx_files(out_dir)
    if not onnx_files:
        _die(f"No .onnx files were produced in {out_dir}. Check logs above.")
    print("[export] produced ONNX files:")
    for p in onnx_files:
        print("  -", p.relative_to(out_dir))

    # 4) Optional post-export quantization
    if args.quantize != "none":
        suffix = args.quantize_output_suffix
        if not suffix:
            suffix = {
                "int8-dynamic": "int8",
                "int8-static": "int8s",
                "fp16": "fp16",
            }[args.quantize]

        if args.quantize == "int8-static":
            if not args.calib_text_file:
                _die("--quantize int8-static requires --calib-text-file with representative prompts (one per line).")
            calib_path = Path(args.calib_text_file)
            if not calib_path.exists():
                _die(f"Calibration file not found: {calib_path}")

            calib_lines = [ln.strip() for ln in calib_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not calib_lines:
                _die(f"Calibration file is empty: {calib_path}")

        print(f"[quant] applying {args.quantize} to exported ONNX files...")
        for src in onnx_files:
            # Write quantized model next to src
            dst = src.with_name(f"{src.stem}.{suffix}{src.suffix}")

            if args.quantize == "int8-dynamic":
                _quantize_dynamic_int8(src, dst, per_channel=args.per_channel, reduce_range=args.reduce_range)

            elif args.quantize == "int8-static":
                _quantize_static_int8(
                    src,
                    dst,
                    calib_texts=calib_lines,
                    model_id=args.model,
                    trust_remote_code=args.trust_remote_code,
                    max_length=args.calib_max_length,
                    num_samples=args.calib_num_samples,
                )

            elif args.quantize == "fp16":
                continue

            else:
                _die(f"Unknown quantization mode: {args.quantize}")

            if not args.keep_fp32:
                # Replace original with quantized to keep downstream paths stable
                src.unlink(missing_ok=True)
                dst.rename(src)
                print(f"[quant] replaced {src.name} with quantized version")

    print("\n[export] done.")
    print(f"[export] output dir: {out_dir}")
    print("[export] next: load these .onnx + tokenizer/config in your web app (onnxruntime-web WebGPU EP).")
    if args.quantize != "none":
        print(f"[export] quantization applied: {args.quantize}")


if __name__ == "__main__":
    main()
