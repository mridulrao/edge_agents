#!/usr/bin/env python3
"""
Model Preparation Script
Downloads HuggingFace models and quantizes them to GGUF format for edge deployment.

Usage:
    python prepare_model.py --model_id Qwen/Qwen2.5-0.5B-Instruct --quant_type q8_0
    python prepare_model.py --model_id google/gemma-3-270m-it --quant_type q4_k_m
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


class ModelPreparer:
    def __init__(self, output_dir="./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llama_cpp_path = self._find_llama_cpp()
        
    def _find_llama_cpp(self):
        """Find llama.cpp installation directory"""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.resolve()
        
        # Try common locations
        possible_paths = [
            script_dir / "llama.cpp",              # Same directory as script (PRIORITY)
            Path.cwd() / "llama.cpp",              # Current working directory
            Path.home() / "llama.cpp",             # Home directory
            Path("/opt/homebrew/opt/llama.cpp"),   # Homebrew (Apple Silicon)
            Path("/usr/local/opt/llama.cpp"),      # Homebrew (Intel)
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"✓ Found llama.cpp at: {path}")
                return path
        
        print("✗ llama.cpp not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease either:")
        print("  1. Clone llama.cpp in the same directory as this script")
        print("  2. Use --llama_cpp_path to specify the location")
        print("  3. Clone to ~/llama.cpp")
        return None
    
    def download_model(self, model_id, force_download=False):
        """Download model from HuggingFace"""
        print(f"\n{'='*60}")
        print(f"Downloading model: {model_id}")
        print(f"{'='*60}")
        
        # Create safe directory name
        model_name = model_id.replace("/", "_")
        model_path = self.output_dir / model_name
        
        if model_path.exists() and not force_download:
            print(f"✓ Model already exists at: {model_path}")
            print("  Use --force to re-download")
            return model_path
        
        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✓ Model downloaded to: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            sys.exit(1)
    
    def _install_gguf_requirements(self):
        """Install the correct gguf package version from llama.cpp requirements"""
        print("\n🔧 Installing gguf package dependencies...")
        
        # Try to find requirements.txt in llama.cpp
        requirements_locations = [
            self.llama_cpp_path / "requirements.txt",
            self.llama_cpp_path / "libexec" / "requirements.txt",
            self.llama_cpp_path / ".." / ".." / "Cellar" / "llama.cpp" / "*" / "libexec" / "requirements.txt",
        ]
        
        requirements_file = None
        for req_path in requirements_locations:
            # Handle glob pattern for Cellar
            if "*" in str(req_path):
                import glob
                matches = glob.glob(str(req_path))
                if matches:
                    requirements_file = Path(matches[0])
                    break
            elif req_path.exists():
                requirements_file = req_path
                break
        
        if requirements_file and requirements_file.exists():
            try:
                print(f"  Found requirements at: {requirements_file}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_file)],
                    check=True
                )
                print("  ✓ Dependencies installed")
                return True
            except subprocess.CalledProcessError:
                print("  ⚠ Failed to install from requirements.txt, trying manual install...")
        
        # Fallback: install specific version known to work with llama.cpp
        try:
            print("  Installing gguf package...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf>=0.10.0"],
                check=True
            )
            print("  ✓ gguf package installed")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to install gguf package")
            return False
    
    def convert_to_gguf(self, model_path, output_type="f16"):
        """Convert HuggingFace model to GGUF format"""
        print(f"\n{'='*60}")
        print(f"Converting to GGUF format (output_type: {output_type})")
        print(f"{'='*60}")
        
        if not self.llama_cpp_path:
            print("✗ llama.cpp path not set. Cannot convert.")
            sys.exit(1)
        
        # Try multiple possible locations for the conversion script
        possible_script_locations = [
            self.llama_cpp_path / "convert_hf_to_gguf.py",           # Source build - root
            self.llama_cpp_path / "convert-hf-to-gguf.py",           # Alternative name
            self.llama_cpp_path / "libexec" / "convert_hf_to_gguf.py",  # Homebrew
            self.llama_cpp_path / "bin" / "convert_hf_to_gguf.py",   # Some builds
        ]
        
        convert_script = None
        for script_path in possible_script_locations:
            if script_path.exists():
                convert_script = script_path
                print(f"✓ Found conversion script: {convert_script}")
                break
        
        if not convert_script:
            print(f"✗ Conversion script not found in any of these locations:")
            for loc in possible_script_locations:
                print(f"  - {loc}")
            print("\nMake sure you have the Python scripts in your llama.cpp installation.")
            print("If you built from source, they should be in the root directory.")
            sys.exit(1)
        
        output_file = model_path / f"ggml-model-{output_type}.gguf"
        
        # Try conversion, and if it fails due to import errors, install dependencies
        max_retries = 2
        for attempt in range(max_retries):
            try:
                cmd = [
                    sys.executable,
                    str(convert_script),
                    str(model_path),
                    "--outtype", output_type,
                    "--outfile", str(output_file)
                ]
                
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                print(f"✓ Converted to: {output_file}")
                return output_file
                
            except subprocess.CalledProcessError as e:
                error_output = e.stderr
                
                # Check if it's a missing module or import error
                if ("ModuleNotFoundError" in error_output or 
                    "ImportError" in error_output or 
                    "No module named" in error_output) and attempt == 0:
                    print(f"⚠ Missing dependencies detected. Installing...")
                    if self._install_gguf_requirements():
                        print("  Retrying conversion...")
                        continue
                
                print(f"✗ Conversion failed: {e}")
                print(error_output)
                
                # Provide helpful error message
                if "ImportError" in error_output or "ModuleNotFoundError" in error_output:
                    print("\n💡 Troubleshooting:")
                    print("  The llama.cpp conversion script needs specific Python packages.")
                    print("  Try manually installing the requirements:")
                    print(f"    pip install gguf numpy sentencepiece protobuf")
                    print("\n  Or clone llama.cpp and use its requirements:")
                    print("    git clone https://github.com/ggerganov/llama.cpp")
                    print("    cd llama.cpp")
                    print("    pip install -r requirements.txt")
                
                sys.exit(1)
    
    def quantize_model(self, input_file, quant_type="q8_0"):
        """Quantize GGUF model to specified quantization type
        
        Common quantization types:
        - q4_0: 4-bit, small, lower quality
        - q4_k_m: 4-bit, medium quality (good balance)
        - q5_0: 5-bit, medium quality
        - q5_k_m: 5-bit, higher quality
        - q8_0: 8-bit, high quality, good for edge (RECOMMENDED)
        - f16: 16-bit float (no quantization)
        """
        print(f"\n{'='*60}")
        print(f"Quantizing to {quant_type}")
        print(f"{'='*60}")
        
        if not self.llama_cpp_path:
            print("✗ llama.cpp path not set. Cannot quantize.")
            sys.exit(1)
        
        # Try multiple possible locations for the quantize binary
        possible_binary_locations = [
            self.llama_cpp_path / "llama-quantize",                    # Source build - root (old Makefile)
            self.llama_cpp_path / "build" / "bin" / "llama-quantize",  # CMake build (NEW)
            self.llama_cpp_path / "quantize",                          # Alternative name
            self.llama_cpp_path / "bin" / "llama-quantize",            # Some builds
            self.llama_cpp_path / "libexec" / "llama-quantize",        # Homebrew
        ]
        
        quantize_bin = None
        for bin_path in possible_binary_locations:
            if bin_path.exists() and os.access(bin_path, os.X_OK):
                quantize_bin = bin_path
                print(f"✓ Found quantization binary: {quantize_bin}")
                break
        
        if not quantize_bin:
            print(f"✗ Quantization binary not found in any of these locations:")
            for loc in possible_binary_locations:
                print(f"  - {loc}")
            print("\nMake sure llama.cpp is compiled with CMake:")
            print("  cd llama.cpp")
            print("  mkdir build && cd build")
            print("  cmake .. -DGGML_METAL=ON")
            print("  cmake --build . --config Release -j")
            print("  make LLAMA_METAL=1")
            sys.exit(1)
        
        output_file = input_file.parent / f"ggml-model-{quant_type}.gguf"
        
        try:
            cmd = [
                str(quantize_bin),
                str(input_file),
                str(output_file),
                quant_type.upper()
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print(f"✓ Quantized model saved to: {output_file}")
            
            # Show file sizes
            orig_size = input_file.stat().st_size / (1024**2)
            quant_size = output_file.stat().st_size / (1024**2)
            reduction = (1 - quant_size/orig_size) * 100
            
            print(f"\n📊 Size Comparison:")
            print(f"  Original ({input_file.name}): {orig_size:.2f} MB")
            print(f"  Quantized ({output_file.name}): {quant_size:.2f} MB")
            print(f"  Reduction: {reduction:.1f}%")
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Quantization failed: {e}")
            print(e.stderr)
            sys.exit(1)
    
    def prepare_model(self, model_id, quant_type="q8_0", force_download=False):
        """Complete pipeline: download -> convert -> quantize"""
        print(f"\n🚀 Starting model preparation pipeline")
        print(f"   Model: {model_id}")
        print(f"   Quantization: {quant_type}")
        
        # Step 1: Download
        model_path = self.download_model(model_id, force_download)
        
        # Step 2: Convert to GGUF (f16 intermediate format)
        gguf_f16 = self.convert_to_gguf(model_path, output_type="f16")
        
        # Step 3: Quantize
        if quant_type.lower() == "f16":
            print(f"\n✓ Skipping quantization (f16 requested)")
            final_model = gguf_f16
        else:
            final_model = self.quantize_model(gguf_f16, quant_type)
        
        print(f"\n{'='*60}")
        print(f"✅ Model preparation complete!")
        print(f"{'='*60}")
        print(f"Final model: {final_model}")
        print(f"Size: {final_model.stat().st_size / (1024**2):.2f} MB")
        
        return final_model


def main():
    parser = argparse.ArgumentParser(
        description="Download and quantize HuggingFace models for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and quantize to INT8
  python prepare_model.py --model_id Qwen/Qwen2.5-0.5B-Instruct --quant_type q8_0
  
  # Use 4-bit quantization for smaller size
  python prepare_model.py --model_id google/gemma-3-270m-it --quant_type q4_k_m
  
  # Keep FP16 (no quantization)
  python prepare_model.py --model_id HuggingFaceTB/SmolLM2-135M --quant_type f16
  
  # Force re-download
  python prepare_model.py --model_id Qwen/Qwen3-0.6B --quant_type q8_0 --force

Quantization types (from smallest to largest):
  q4_0, q4_k_m  - 4-bit (very small, lower quality)
  q5_0, q5_k_m  - 5-bit (medium quality)
  q8_0          - 8-bit (recommended for edge - good quality/size balance)
  f16           - 16-bit (no quantization, largest)
        """
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )
    
    parser.add_argument(
        "--quant_type",
        type=str,
        default="q8_0",
        choices=["q4_0", "q4_k_m", "q5_0", "q5_k_m", "q8_0", "f16"],
        help="Quantization type (default: q8_0)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Output directory for models (default: ./models)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    parser.add_argument(
        "--llama_cpp_path",
        type=str,
        help="Path to llama.cpp directory (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Create preparer
    preparer = ModelPreparer(output_dir=args.output_dir)
    
    # Override llama.cpp path if provided
    if args.llama_cpp_path:
        preparer.llama_cpp_path = Path(args.llama_cpp_path)
    
    # Run pipeline
    preparer.prepare_model(
        model_id=args.model_id,
        quant_type=args.quant_type,
        force_download=args.force
    )


if __name__ == "__main__":
    main()