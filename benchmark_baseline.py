#!/usr/bin/env python3
"""Baseline performance benchmark before implementing multi-layer caching."""

import asyncio
import time
from pathlib import Path

from kreuzberg import ExtractionConfig, batch_extract_file, extract_file_sync
from kreuzberg._utils._document_cache import clear_document_cache, get_document_cache


async def run_baseline_benchmark():
    """Run comprehensive baseline benchmark."""
    test_files_dir = Path("tests/test_source_files")
    test_files = list(test_files_dir.glob("*.pdf"))
    
    if not test_files:
        print("No PDF test files found")
        return
    
    # Select test files
    single_file = test_files[0]
    if len(test_files) >= 3:
        mixed_files = test_files[:3]
    else:
        mixed_files = [single_file] * 3
    
    print("=" * 60)
    print("BASELINE PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Test files: {len(test_files)} PDFs available")
    print(f"Primary test file: {single_file.name} ({single_file.stat().st_size / 1024:.1f} KB)")
    print()
    
    results = {}
    
    # Test 1: Single file extraction (cold)
    print("1. SINGLE FILE EXTRACTION (COLD)")
    print("-" * 40)
    clear_document_cache()
    
    start_time = time.time()
    result = extract_file_sync(single_file)
    cold_duration = time.time() - start_time
    
    print(f"Duration: {cold_duration:.3f}s")
    print(f"Content length: {len(result.content):,} chars")
    print(f"Success: {'✅' if not result.metadata.get('error') else '❌'}")
    
    results["single_file_cold"] = {
        "duration": cold_duration,
        "content_length": len(result.content),
        "success": not result.metadata.get('error')
    }
    
    # Test 2: Single file extraction (cached)
    print("\n2. SINGLE FILE EXTRACTION (CACHED)")
    print("-" * 40)
    
    start_time = time.time()
    cached_result = extract_file_sync(single_file)
    cached_duration = time.time() - start_time
    
    speedup = cold_duration / cached_duration if cached_duration > 0 else float('inf')
    
    print(f"Duration: {cached_duration:.6f}s")
    print(f"Speedup: {speedup:,.0f}x")
    print(f"Cache hit: {'✅' if cached_duration < 0.1 else '❌'}")
    
    results["single_file_cached"] = {
        "duration": cached_duration,
        "speedup": speedup,
        "cache_hit": cached_duration < 0.1
    }
    
    # Test 3: Same file batch (10 copies)
    print("\n3. SAME FILE BATCH (10 COPIES)")
    print("-" * 40)
    clear_document_cache()
    
    same_files = [single_file] * 10
    start_time = time.time()
    same_results = await batch_extract_file(same_files)
    same_duration = time.time() - start_time
    
    same_successes = sum(1 for r in same_results if not r.metadata.get("error"))
    same_failure_rate = ((len(same_results) - same_successes) / len(same_results)) * 100
    
    print(f"Duration: {same_duration:.3f}s")
    print(f"Avg per file: {same_duration / len(same_files):.3f}s")
    print(f"Success rate: {100 - same_failure_rate:.1f}%")
    print(f"Throughput: {len(same_files) / same_duration:.1f} files/sec")
    
    results["same_file_batch"] = {
        "duration": same_duration,
        "avg_per_file": same_duration / len(same_files),
        "success_rate": 100 - same_failure_rate,
        "throughput": len(same_files) / same_duration
    }
    
    # Test 4: Mixed files batch
    print("\n4. MIXED FILES BATCH")
    print("-" * 40)
    clear_document_cache()
    
    start_time = time.time()
    mixed_results = await batch_extract_file(mixed_files)
    mixed_duration = time.time() - start_time
    
    mixed_successes = sum(1 for r in mixed_results if not r.metadata.get("error"))
    mixed_failure_rate = ((len(mixed_results) - mixed_successes) / len(mixed_results)) * 100
    
    print(f"Duration: {mixed_duration:.3f}s")
    print(f"Avg per file: {mixed_duration / len(mixed_files):.3f}s")
    print(f"Success rate: {100 - mixed_failure_rate:.1f}%")
    print(f"Throughput: {len(mixed_files) / mixed_duration:.1f} files/sec")
    
    results["mixed_files_batch"] = {
        "duration": mixed_duration,
        "avg_per_file": mixed_duration / len(mixed_files),
        "success_rate": 100 - mixed_failure_rate,
        "throughput": len(mixed_files) / mixed_duration
    }
    
    # Test 5: OCR-heavy workload
    print("\n5. OCR-HEAVY WORKLOAD (FORCE OCR)")
    print("-" * 40)
    clear_document_cache()
    
    ocr_config = ExtractionConfig(force_ocr=True, ocr_backend="tesseract")
    ocr_files = [single_file] * 3  # Smaller batch for OCR
    
    start_time = time.time()
    ocr_results = await batch_extract_file(ocr_files, ocr_config)
    ocr_duration = time.time() - start_time
    
    ocr_successes = sum(1 for r in ocr_results if not r.metadata.get("error"))
    ocr_failure_rate = ((len(ocr_results) - ocr_successes) / len(ocr_results)) * 100
    
    print(f"Duration: {ocr_duration:.3f}s")
    print(f"Avg per file: {ocr_duration / len(ocr_files):.3f}s")
    print(f"Success rate: {100 - ocr_failure_rate:.1f}%")
    print(f"Throughput: {len(ocr_files) / ocr_duration:.2f} files/sec")
    
    results["ocr_workload"] = {
        "duration": ocr_duration,
        "avg_per_file": ocr_duration / len(ocr_files),
        "success_rate": 100 - ocr_failure_rate,
        "throughput": len(ocr_files) / ocr_duration
    }
    
    # Cache statistics
    cache = get_document_cache()
    cache_stats = cache.get_stats()
    
    print("\n6. CACHE STATISTICS")
    print("-" * 40)
    print(f"Cached documents: {cache_stats['cached_documents']}")
    print(f"Processing documents: {cache_stats['processing_documents']}")
    print(f"Total cache size: {cache_stats['total_cache_size_mb']:.2f} MB")
    
    results["cache_stats"] = cache_stats
    
    # Summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"🔥 Cold extraction: {cold_duration:.3f}s")
    print(f"⚡ Cached extraction: {cached_duration:.6f}s ({speedup:,.0f}x speedup)")
    print(f"📦 Same file batch: {same_duration:.3f}s ({same_results[0] and len(same_results[0].content):,} chars)")
    print(f"🎯 Mixed files batch: {mixed_duration:.3f}s")
    print(f"🔍 OCR workload: {ocr_duration:.3f}s")
    print(f"💾 Cache efficiency: {cache_stats['total_cache_size_mb']:.2f}MB for {cache_stats['cached_documents']} docs")
    
    # Performance indicators
    print(f"\n📊 PERFORMANCE INDICATORS:")
    print(f"   - Document cache working: {'✅' if speedup > 1000 else '❌'}")
    print(f"   - Same-file concurrency: {'✅' if same_failure_rate == 0 else '❌'}")
    print(f"   - Mixed-file concurrency: {'✅' if mixed_failure_rate == 0 else '❌'}")
    print(f"   - OCR stability: {'✅' if ocr_failure_rate == 0 else '❌'}")
    
    return results


if __name__ == "__main__":
    baseline_results = asyncio.run(run_baseline_benchmark())
    
    # Save baseline for comparison
    import json
    baseline_file = Path("baseline_results.json")
    with baseline_file.open("w") as f:
        json.dump(baseline_results, f, indent=2, default=str)
    
    print(f"\n💾 Baseline results saved to {baseline_file}")
    print("Ready to implement multi-layer caching improvements!")