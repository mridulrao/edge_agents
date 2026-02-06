#!/usr/bin/env python3
"""
Main entry point for Edge Question Generation with ONNX Runtime
Quick test/demo script - UPDATED VERSION
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.onnx_adaption.onnx_adapter import create_onnx_adapter
from src.pipeline import QuestionGenerationPipeline
from src.pipeline_types import ArticleInput, GenerationConfig


async def main():
    """Run a simple test of the question generation pipeline"""
    
    print("=" * 60)
    print("Edge Question Generation - ONNX Test")
    print("=" * 60)
    
    # Sample article for testing
    sample_article = ArticleInput(
        text="""
        To enable conversion tracking in your ad platform, start by navigating to 
        Tools > Conversions in the main menu. Click the + button to create a new 
        conversion action.
        
        You can track three main types of conversions: web conversions, app installs, 
        or phone calls. Each type requires different setup steps and configurations.
        
        For web conversions, you'll need to install a tracking tag on your website. 
        Copy the provided tracking code snippet and paste it into your website's 
        <head> section. The tag should fire when a user completes your desired action, 
        such as making a purchase or submitting a lead form.
        
        To configure app install tracking, integrate with your mobile measurement 
        partner (MMP). Select your MMP from the dropdown menu - options include 
        AppsFlyer, Adjust, or Firebase. You'll need to provide your app ID and 
        API credentials from your MMP dashboard.
        
        For phone call tracking, you can set up call extensions or use a forwarding 
        number. The platform will track calls that come through your ads and attribute 
        them to the appropriate campaigns.
        
        After setting up tracking, you can use conversion data to optimize your 
        campaigns. Set bid strategies based on conversion goals, create audiences 
        of converters, and analyze which keywords and ad groups drive the most 
        valuable actions.
        """,
        desired_questions=3
    )
    
    try:
        # Create adapter with auto-managed server
        print("\nStarting ONNX Runtime...")
        
        adapter = create_onnx_adapter(
            model_id="models/onnx_adaption/onnx_models/smollm2_360m_instruct_onnx_int8",
            onnx_filename="model_quantized.onnx",
            providers=["CPUExecutionProvider"],
            timeout_seconds=120.0,
            max_retries=3,
        )
        
        # Wait for readiness
        print("Waiting for model to load...")
        ready = await adapter.wait_until_ready(max_wait_seconds=180)
        
        if not ready:
            print("Model failed to become ready within timeout")
            return 1
        
        print("Model ready!\n")
        
        # Create pipeline with custom config (more tokens)
        print("Creating pipeline...")
        
        # Use more generous token limit for generation
        config = GenerationConfig(
            max_output_tokens=64,   # 32–80 sweet spot
            temperature=0.2,        # 0.0 for JSON
            top_p=1.0,              # or omit
        )
        
        pipeline = QuestionGenerationPipeline(adapter=adapter, config=config)
        
        # Generate questions
        print("Generating questions...") 
        
        result = await pipeline.generate_with_metrics(sample_article)
        
        # Display results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print(f"\nGenerated {len(result.questions)} questions")
        print(f"\nMetrics:")
        print(f"   • Total latency: {result.metrics.latency_ms:.1f}ms")
        print(f"   • Peak memory: {result.metrics.memory_peak_mb:.1f}MB")
        print(f"   • Chunks created: {result.metrics.chunks_created}")
        print(f"   • Candidates generated: {result.metrics.candidates_generated}")
        print(f"   • Validation pass rate: {result.metrics.validation_pass_rate:.1%}")
        print(f"   • Deduplication reduction: {result.metrics.deduplication_reduction:.1%}")
        
        print(f"\nQuestions:\n")    
        for i, q in enumerate(result.questions, 1):
            print(f"{i}. {q.question}")
            print()
        
        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if 'adapter' in locals():
            print("\nCleaning up...")
            await adapter.shutdown()
            print("Server stopped")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)