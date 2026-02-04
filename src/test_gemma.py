#!/usr/bin/env python3
"""
Test script with verbose debugging
"""

import asyncio
import sys
import traceback
from pathlib import Path

print("Starting test script...")
print(f"Python version: {sys.version}")
print(f"Working directory: {Path.cwd()}")

# Test imports
print("\nTesting imports...")
try:
    from pipeline_types import ArticleInput, GenerationConfig
    print("✓ Imported pipeline_types")
except ImportError as e:
    print(f"✗ Failed to import pipeline_types: {e}")
    sys.exit(1)

try:
    from pipeline import QuestionGenerationPipeline
    print("✓ Imported pipeline")
except ImportError as e:
    print(f"✗ Failed to import pipeline: {e}")
    sys.exit(1)

try:
    from gemma_adapter import GemmaAdapter
    print("✓ Imported gemma_adapter")
except ImportError as e:
    print(f"✗ Failed to import gemma_adapter: {e}")
    sys.exit(1)

# Find paths
print("\nFinding paths...")
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR_GEMMA = MODEL_DIR / "models" / "bartowski_google_gemma-3-270m-it-GGUF"

print(f"Script dir: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Model dir: {MODEL_DIR}")
print(f"Model dir gemma: {MODEL_DIR_GEMMA}")

# Find model
print("\nSearching for GGUF models...")
gguf_files = list(MODEL_DIR_GEMMA.rglob("*.gguf"))
print(f"Found {len(gguf_files)} GGUF files")

if not gguf_files:
    print("✗ No GGUF models found!")
    sys.exit(1)

model_path = gguf_files[0]
print(f"✓ Using model: {model_path}")
print(f"  Size: {model_path.stat().st_size / (1024**2):.1f}MB")

# Find llama.cpp
print("\nSearching for llama.cpp...")
llama_cpp_path = MODEL_DIR / "llama.cpp"

if not llama_cpp_path.exists():
    print(f"✗ llama.cpp not found at: {llama_cpp_path}")
    sys.exit(1)

print(f"✓ Found llama.cpp at: {llama_cpp_path}")

# Check for binary
binary_path = llama_cpp_path / "build" / "bin" / "llama-cli"
if not binary_path.exists():
    print(f"✗ llama-cli binary not found at: {binary_path}")
    sys.exit(1)

print(f"✓ Found llama-cli at: {binary_path}")

# Test adapter initialization
async def test_adapter():
    print("\n" + "="*70)
    print("Testing Adapter Initialization")
    print("="*70)
    
    try:
        print("\nCreating adapter...")
        adapter = GemmaAdapter(
            model_path=str(model_path),
            llama_cpp_path=str(llama_cpp_path),
            n_threads=4,
            use_gpu=True,
            timeout_seconds=60.0  # Longer timeout
        )
        print("✓ Adapter created")
        
        print("\nInitializing model (this may take a moment)...")
        await adapter.initialize()
        print("✓ Model initialized")
        
        print("\nAdapter info:")
        info = adapter.get_info()
        print(f"  Ready: {info['ready']}")
        print(f"  Model: {info['model_name']}")
        print(f"  GPU: {info['config']['use_gpu']}")
        
        return adapter
        
    except Exception as e:
        print(f"\n✗ Adapter initialization failed:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None

# Test generation
async def test_generation(adapter):
    print("\n" + "="*70)
    print("Testing Question Generation")
    print("="*70)
    
    try:
        print("\nCreating pipeline...")
        pipeline = QuestionGenerationPipeline(adapter)
        print("✓ Pipeline created")
        
        print("\nGenerating questions (this may take 10-30 seconds)...")
        article = ArticleInput(
            text="""
            Conversion tracking allows you to measure user actions after clicking your ads.
            To enable conversion tracking, navigate to Tools > Conversions in your account.
            Click the + button and select the conversion type you want to track.
            You can track web conversions, app installs, or phone calls.
            """,
            desired_questions=2  # Start with just 2 to test
        )
        
        result = await pipeline.generate_with_metrics(article)
        
        print(f"\n✓ Generated {len(result.questions)} questions:")
        for i, q in enumerate(result.questions, 1):
            print(f"\n{i}. {q.question}")
            print(f"   Type: {q.type}")
            print(f"   Confidence: {q.confidence_score:.2f}")
        
        print(f"\n📊 Metrics:")
        print(f"  Latency: {result.metrics.latency_ms:.0f}ms")
        print(f"  Memory: {result.metrics.memory_peak_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Generation failed:")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False

# Main
async def main():
    print("\n" + "="*70)
    print("GEMMA ADAPTER TEST")
    print("="*70)
    
    adapter = await test_adapter()
    
    if adapter:
        success = await test_generation(adapter)
        if success:
            print("\n" + "="*70)
            print("✓ ALL TESTS PASSED!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ Generation test failed")
            print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Initialization test failed")
        print("="*70)

if __name__ == "__main__":
    try:
        print("\nRunning async main...")
        asyncio.run(main())
        print("\nScript completed.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)