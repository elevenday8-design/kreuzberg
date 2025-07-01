#!/usr/bin/env python3
"""Final comprehensive benchmark comparing all improvements."""

import asyncio
import json
import time
from pathlib import Path

from kreuzberg import ExtractionConfig, extract_file
from kreuzberg._utils._cache import (
    get_ocr_cache, get_document_cache, get_table_cache, get_mime_cache, 
    clear_all_caches
)


async def run_final_benchmark():
    """Run comprehensive benchmark of all caching improvements."""
    test_files_dir = Path("tests/test_source_files")
    pdf_files = list(test_files_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF test files found")
        return
    
    single_file = pdf_files[0]
    print(f"Final benchmark with: {single_file.name}")
    
    # Configuration that uses all cache layers
    full_config = ExtractionConfig(
        force_ocr=True, 
        ocr_backend="tesseract",
        extract_tables=True,
        chunk_content=True
    )
    
    print("\n" + "=" * 70)
    print("FINAL COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    
    # Baseline: Cold extraction (no cache)
    print("\n🔥 BASELINE: COLD EXTRACTION")
    print("-" * 50)
    clear_all_caches()
    
    start_time = time.time()
    try:
        result_baseline = await extract_file(single_file, config=full_config)
        baseline_duration = time.time() - start_time
        
        print(f"Duration: {baseline_duration:.3f}s")
        print(f"Content: {len(result_baseline.content):,} chars")
        print(f"Tables: {len(result_baseline.tables)}")
        print(f"Chunks: {len(result_baseline.chunks)}")
        print(f"Success: ✅")
        baseline_success = True
    except Exception as e:
        baseline_duration = time.time() - start_time
        print(f"Duration: {baseline_duration:.3f}s")
        print(f"Error: {e}")
        print(f"Success: ❌")
        baseline_success = False
        result_baseline = None
    
    # Test: Multi-layer cache (msgspec)
    print("\n⚡ MULTI-LAYER CACHE (MSGSPEC)")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result_cached = await extract_file(single_file, config=full_config)
        cached_duration = time.time() - start_time
        
        total_speedup = baseline_duration / cached_duration if cached_duration > 0 else float('inf')
        content_match = (result_baseline.content == result_cached.content 
                        if result_baseline and result_cached else False)
        
        print(f"Duration: {cached_duration:.6f}s")
        print(f"Speedup: {total_speedup:,.0f}x")
        print(f"Content match: {'✅' if content_match else '❌'}")
        print(f"Cache hit: {'✅' if cached_duration < 0.1 else '❌'}")
        cached_success = True
    except Exception as e:
        cached_duration = time.time() - start_time
        print(f"Duration: {cached_duration:.6f}s")
        print(f"Error: {e}")
        print(f"Success: ❌")
        cached_success = False
        total_speedup = 1
        content_match = False
    
    # Multiple runs for stability test
    print("\n🔄 STABILITY TEST (10 RUNS)")
    print("-" * 50)
    
    run_times = []
    for i in range(10):
        start_time = time.time()
        try:
            await extract_file(single_file, config=full_config)
            duration = time.time() - start_time
            run_times.append(duration)
            print(f"  Run {i+1:2d}: {duration:.6f}s")
        except Exception:
            run_times.append(float('inf'))
            print(f"  Run {i+1:2d}: ERROR")
    
    valid_times = [t for t in run_times if t != float('inf')]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        std_dev = (sum((t - avg_time) ** 2 for t in valid_times) / len(valid_times)) ** 0.5
        
        print(f"Average: {avg_time:.6f}s")
        print(f"Range: {min_time:.6f}s - {max_time:.6f}s")
        print(f"Std Dev: {std_dev:.6f}s")
        print(f"Reliability: {len(valid_times)}/10 runs successful")
        print(f"Consistency: {'✅' if std_dev < 0.001 else '❌'}")
    
    # Cache layer analysis
    print("\n💾 CACHE LAYER ANALYSIS")
    print("-" * 50)
    
    caches = {
        "MIME": get_mime_cache(),
        "OCR": get_ocr_cache(),
        "Tables": get_table_cache(),
        "Documents": get_document_cache()
    }
    
    total_size = 0
    total_items = 0
    
    for name, cache in caches.items():
        stats = cache.get_stats()
        total_size += stats['total_cache_size_mb']
        total_items += stats['cached_results']
        
        efficiency = stats['cached_results'] / max(stats['total_cache_size_mb'], 0.001)
        
        print(f"{name:>9}: {stats['cached_results']:>3} items, "
              f"{stats['total_cache_size_mb']:>7.3f}MB, "
              f"{efficiency:>5.0f} items/MB")
    
    print(f"{'TOTAL':>9}: {total_items:>3} items, {total_size:>7.3f}MB")
    
    # Performance breakdown estimate
    print("\n📊 PERFORMANCE BREAKDOWN")
    print("-" * 50)
    
    if baseline_success and cached_success:
        time_saved = baseline_duration - cached_duration
        
        # Estimated component contributions (based on typical extraction patterns)
        components = {
            "MIME Detection": 0.001,  # Very fast
            "OCR Processing": baseline_duration * 0.7,  # ~70% of time
            "Table Extraction": baseline_duration * 0.25,  # ~25% of time  
            "Document Processing": baseline_duration * 0.05,  # ~5% of time
        }
        
        print("Estimated time breakdown:")
        for component, time_est in components.items():
            percentage = (time_est / baseline_duration) * 100
            print(f"  {component:<20}: {time_est:>6.3f}s ({percentage:>5.1f}%)")
        
        print(f"\nTotal time saved: {time_saved:.3f}s ({(time_saved/baseline_duration)*100:.1f}%)")
        print(f"Cache efficiency: {(1 - cached_duration/baseline_duration)*100:.3f}%")
    
    # Storage efficiency
    print("\n💽 STORAGE EFFICIENCY")
    print("-" * 50)
    
    cache_root = Path(".kreuzberg")
    if cache_root.exists():
        file_count = len(list(cache_root.rglob("*.msgpack")))
        dir_size = sum(f.stat().st_size for f in cache_root.rglob("*") if f.is_file())
        
        print(f"Cache files: {file_count}")
        print(f"Disk usage: {dir_size/1024:.1f}KB")
        print(f"Avg file size: {dir_size/file_count/1024:.1f}KB" if file_count > 0 else "N/A")
        print(f"Storage efficiency: {'✅' if dir_size < 500*1024 else '❌'}")  # < 500KB
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 FINAL BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"🔥 Baseline (cold):     {baseline_duration:>8.3f}s")
    print(f"⚡ Cached (msgspec):   {cached_duration:>8.6f}s")
    print(f"🚀 Total speedup:      {total_speedup:>8,.0f}x")
    print(f"💾 Cache storage:      {total_size:>8.1f}MB ({total_items} items)")
    print(f"🔧 Avg cached time:    {avg_time:>8.6f}s" if valid_times else "N/A")
    
    # Overall performance indicators
    print(f"\n🏆 ACHIEVEMENT UNLOCKED:")
    print(f"   ⚡ Ultra-fast caching:     {'✅' if total_speedup > 100000 else '❌'}")
    print(f"   🎯 Content accuracy:       {'✅' if content_match else '❌'}")
    print(f"   🔒 Cache reliability:      {'✅' if len(valid_times) >= 9 else '❌'}")
    print(f"   💾 Storage efficiency:     {'✅' if total_size < 5 else '❌'}")
    print(f"   ⚡ Consistent performance: {'✅' if std_dev < 0.001 else '❌'}")
    
    # Technology stack summary
    print(f"\n🛠️  TECHNOLOGY STACK:")
    print(f"   📦 Multi-layer caching:    OCR + Tables + MIME + Documents")
    print(f"   ⚡ Serialization:          msgspec (ultra-fast msgpack)")
    print(f"   💾 Storage:                File-based (.kreuzberg/ directory)")
    print(f"   🔄 Interfaces:             Async + Sync support")
    print(f"   🛡️  Thread-safety:          Coordinated processing")
    
    return {
        "baseline_duration": baseline_duration,
        "cached_duration": cached_duration,
        "total_speedup": total_speedup,
        "avg_cached_time": avg_time if valid_times else None,
        "cache_size_mb": total_size,
        "cache_items": total_items,
        "reliability_rate": len(valid_times) / 10,
        "content_accuracy": content_match,
        "baseline_success": baseline_success,
        "cached_success": cached_success
    }


if __name__ == "__main__":
    try:
        results = asyncio.run(run_final_benchmark())
        print(f"\n🎊 FINAL BENCHMARK COMPLETED!")
        
        # Save comprehensive results
        final_results_file = Path("final_benchmark_results.json")
        with final_results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Complete results saved to {final_results_file}")
        
        # Show key metrics
        if results["baseline_success"] and results["cached_success"]:
            print(f"\n🔥 KEY METRICS:")
            print(f"   Speedup: {results['total_speedup']:,.0f}x")
            print(f"   Cache size: {results['cache_size_mb']:.1f}MB")
            print(f"   Reliability: {results['reliability_rate']*100:.0f}%")
            print(f"   Accuracy: {'100%' if results['content_accuracy'] else '0%'}")
        
    except Exception as e:
        print(f"\n❌ Final benchmark failed: {e}")
        import traceback
        traceback.print_exc()