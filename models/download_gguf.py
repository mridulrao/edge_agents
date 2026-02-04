#!/usr/bin/env python3
"""
Direct GGUF Download Script
Downloads pre-quantized GGUF models directly from HuggingFace.

This bypasses the conversion step for models that already have GGUF versions available.

Usage:
    python download_gguf.py --model_id bartowski/gemma-3-270m-it-GGUF --quant_type Q8_0
    python download_gguf.py --model_id bartowski/Qwen2.5-0.5B-Instruct-GGUF --quant_type Q8_0
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


class GGUFDownloader:
    def __init__(self, output_dir="./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_gguf(self, model_id: str, quant_type: str = "Q8_0"):
        """Download a pre-quantized GGUF model from HuggingFace"""
        
        print(f"\n{'='*60}")
        print(f"Downloading GGUF model: {model_id}")
        print(f"Quantization: {quant_type}")
        print(f"{'='*60}")
        
        # Common filename patterns for GGUF models
        # Most GGUF repos follow the pattern: model-name-Q8_0.gguf or model-name.Q8_0.gguf
        possible_filenames = [
            f"{quant_type}.gguf",
            f"{quant_type.lower()}.gguf",
            f"model-{quant_type}.gguf",
            f"model-{quant_type.lower()}.gguf",
            f"ggml-model-{quant_type}.gguf",
            f"ggml-model-{quant_type.lower()}.gguf",
        ]
        
        # Extract base model name for additional patterns
        base_name = model_id.split('/')[-1].replace('-GGUF', '').replace('-gguf', '')
        possible_filenames.extend([
            f"{base_name}-{quant_type}.gguf",
            f"{base_name}-{quant_type.lower()}.gguf",
            f"{base_name}.{quant_type}.gguf",
            f"{base_name}.{quant_type.lower()}.gguf",
        ])
        
        # Create output directory
        model_name = model_id.replace("/", "_")
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to download with each possible filename
        downloaded_file = None
        for filename in possible_filenames:
            try:
                print(f"Trying filename: {filename}")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_file = Path(downloaded_path)
                print(f"✓ Successfully downloaded: {filename}")
                break
            except Exception as e:
                continue
        
        if not downloaded_file:
            print(f"\n✗ Could not find GGUF file with quantization {quant_type}")
            print(f"\nTrying to list available files in the repository...")
            
            try:
                from huggingface_hub import list_repo_files
                files = list(list_repo_files(model_id))
                gguf_files = [f for f in files if f.endswith('.gguf')]
                
                if gguf_files:
                    print(f"\nAvailable GGUF files in {model_id}:")
                    for f in gguf_files:
                        size_info = ""
                        print(f"  - {f}")
                    
                    print(f"\nTry one of these quantization types:")
                    quant_types = set()
                    for f in gguf_files:
                        # Extract quantization type from filename
                        for qt in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16"]:
                            if qt in f.upper():
                                quant_types.add(qt)
                    
                    for qt in sorted(quant_types):
                        print(f"  python download_gguf.py --model_id {model_id} --quant_type {qt}")
                else:
                    print(f"No GGUF files found in repository {model_id}")
                    print("This repository might not contain pre-quantized models.")
            except Exception as e:
                print(f"Error listing repository files: {e}")
            
            sys.exit(1)
        
        # Show file info
        file_size_mb = downloaded_file.stat().st_size / (1024**2)
        
        print(f"\n{'='*60}")
        print(f"✅ Download complete!")
        print(f"{'='*60}")
        print(f"File: {downloaded_file}")
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"\nYou can now test this model with:")
        print(f"  python test_inference.py --model_path {downloaded_file}")
        
        return downloaded_file


# Popular pre-quantized GGUF model repositories
RECOMMENDED_GGUF_REPOS = {
    "gemma-3-270m": "bartowski/gemma-3-270m-it-GGUF",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-quantized GGUF models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Download Gemma-3-270M (Q8_0)
  python download_gguf.py --model_id bartowski/gemma-3-270m-it-GGUF --quant_type Q8_0
  
  # Download Qwen2.5-0.5B (Q8_0)
  python download_gguf.py --model_id Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant_type Q8_0
  
  # Download SmolLM2-135M (Q8_0)
  python download_gguf.py --model_id HuggingFaceTB/SmolLM2-135M-Instruct-GGUF --quant_type Q8_0
  
  # Try different quantization
  python download_gguf.py --model_id bartowski/gemma-3-270m-it-GGUF --quant_type Q4_K_M

Recommended GGUF repositories:
{chr(10).join(f"  {k}: {v}" for k, v in RECOMMENDED_GGUF_REPOS.items())}

Common quantization types available:
  Q4_K_M - 4-bit (smaller, faster)
  Q5_K_M - 5-bit (medium)
  Q8_0   - 8-bit (recommended for edge)
  F16    - 16-bit (largest, highest quality)
        """
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace GGUF model repository ID"
    )
    
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q8_0",
        help="Quantization type (e.g., Q8_0, Q4_K_M, Q5_K_M, F16)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Output directory for models (default: ./models)"
    )
    
    parser.add_argument(
        "--list-recommended",
        action="store_true",
        help="List recommended GGUF repositories"
    )
    
    args = parser.parse_args()
    
    if args.list_recommended:
        print("\n🎯 Recommended Pre-Quantized GGUF Repositories:\n")
        for name, repo_id in RECOMMENDED_GGUF_REPOS.items():
            print(f"  {name:20} → {repo_id}")
        print("\nUsage:")
        print(f"  python download_gguf.py --model_id <repo_id> --quant_type Q8_0")
        return
    
    downloader = GGUFDownloader(output_dir=args.output_dir)
    downloader.download_gguf(
        model_id=args.model_id,
        quant_type=args.quant_type
    )


if __name__ == "__main__":
    main()