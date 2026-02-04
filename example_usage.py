"""
Example: Using the Question Generation Pipeline

Demonstrates end-to-end usage with the MockAdapter.
"""

import asyncio
from src.pipeline import QuestionGenerationPipeline
from src.model_adapter import MockAdapter
from src.types import GenerationConfig, ArticleInput




# Sample ad tech article
SAMPLE_ARTICLE = """
To set up conversion tracking in Campaign Manager, navigate to the Tools menu and select Conversions. 
Click the + button to create a new conversion action. You'll need to choose the conversion type: 
web conversions, app installs, or phone calls. Each type requires different setup steps.

For web conversions, you'll need to install a tracking tag on your website. Copy the global site tag 
and place it in the header of every page. Then create conversion-specific tags for actions you want 
to track, such as purchases, sign-ups, or downloads. Place these tags on the relevant thank-you or 
confirmation pages.

To configure audience targeting, go to Campaign Settings and select Audiences. You can create custom 
audiences based on demographics, interests, behaviors, or website visitors. Use the audience builder 
to combine multiple criteria. For example, you might target users aged 25-45 who are interested in 
technology and have visited your product pages in the last 30 days.

Budget management is available under the Campaign Budget section. Set your daily or monthly spend 
limits and choose between standard or accelerated delivery. Standard delivery spreads your budget 
evenly throughout the day, while accelerated delivery shows ads as quickly as possible until your 
budget is exhausted. You can also set bid adjustments for different devices, locations, or time periods.

For troubleshooting conversion tracking issues, first verify that your tags are properly installed 
using the Tag Assistant browser extension. Check that the conversion window settings match your 
business needs - the default is 30 days for clicks and 1 day for views. If conversions aren't being 
recorded, ensure your conversion action is set to "Enabled" status and that the tag fires on the 
correct pages.
"""


async def main():
    """Run example pipeline execution"""
    
    print("=" * 80)
    print("Question Generation Pipeline - Example Usage")
    print("=" * 80)
    print()
    
    # Initialize adapter (using mock for demonstration)
    print("1. Initializing MockAdapter...")
    adapter = MockAdapter(latency_ms=150)
    
    # Check readiness
    is_ready = await adapter.is_ready()
    print(f"   Model ready: {is_ready}")
    print()
    
    # Create pipeline with custom config
    print("2. Creating pipeline...")
    config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=200,
        top_p=0.9
    )
    pipeline = QuestionGenerationPipeline(adapter, config)
    print(f"   Config: temp={config.temperature}, max_tokens={config.max_output_tokens}")
    print()
    
    # Prepare article input
    print("3. Preparing article...")
    article = ArticleInput(
        text=SAMPLE_ARTICLE,
        desired_questions=4
    )
    print(f"   Article length: {len(article.text.split())} words")
    print(f"   Target questions: {article.desired_questions}")
    print()
    
    # Generate questions with metrics
    print("4. Running pipeline...")
    print()
    
    try:
        result = await pipeline.generate_with_metrics(article)
        
        # Display results
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        print(f"Generated {len(result.questions)} questions:")
        print()
        
        for i, question in enumerate(result.questions, 1):
            print(f"Question {i}:")
            print(f"  Text: {question.question}")
            print(f"  Type: {question.type}")
            print(f"  Source Chunk: {question.source_chunk_id}")
            print(f"  Confidence: {question.confidence_score:.3f}")
            print()
        
        # Display metrics
        print("=" * 80)
        print("METRICS")
        print("=" * 80)
        print()
        
        metrics = result.metrics
        print(f"Chunks created:           {metrics.chunks_created}")
        print(f"Candidates generated:     {metrics.candidates_generated}")
        print(f"Candidates validated:     {metrics.candidates_validated}")
        print(f"Candidates deduplicated:  {metrics.candidates_deduplicated}")
        print(f"Validation pass rate:     {metrics.validation_pass_rate:.1%}")
        print(f"Dedup reduction:          {metrics.deduplication_reduction:.1%}")
        print(f"Total latency:            {metrics.latency_ms:.1f} ms")
        print(f"Peak memory:              {metrics.memory_peak_mb:.1f} MB")
        print()
        
        if metrics.chunk_processing_times:
            print("Per-chunk processing times:")
            for i, chunk_time in enumerate(metrics.chunk_processing_times):
                print(f"  Chunk {i}: {chunk_time:.1f} ms")
        
        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())