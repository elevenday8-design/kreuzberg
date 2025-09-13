# Kreuzberg Batch Processing Optimization Results

## Optimization Summary

### Worker Count Optimization

We implemented dynamic worker allocation based on batch size:

- **1 task**: 1 worker (avoid overhead)
- **2-3 tasks**: N workers (match task count)
- **4+ tasks**: All CPU cores (14 on test system)

### Key Findings

#### 1. Small Batches (1-3 files)

- Using fewer workers (1-3) is more efficient than always using all cores
- Reduces process startup overhead
- **Improvement**: 4-10% faster for single files

#### 2. Medium Batches (4-20 files)

- All CPU cores provide best throughput
- Parallel processing scales well
- **Per-file time**: ~0.16-0.24s with parallelization

#### 3. Large Batches (20+ files)

- Maximum parallelization is optimal
- Consistent performance at scale
- **Throughput**: ~9,400 chars/second

#### 4. Shared Pool Advantage

- Reusing process pool eliminates initialization overhead
- **Performance gain**: 94.8% improvement for repeated operations
- Ideal for API servers and batch processing systems

## Real-World Performance

### Mixed Document Processing

- **6 office documents**: 14.91s total (2.48s average)
- **4 images**: 2.34s total (0.59s average)
- **5 PDFs**: 16.48s total (3.30s average)
- **5 text files**: 0.01s total (0.001s average)

### Overall Metrics

- **Total files**: 20
- **Total time**: 33.74s
- **Average**: 1.69s per file
- **Throughput**: 9,407 chars/second

## Image Size Impact

### Small Images (640x480)

- Batch 1: 1.46s per image
- Batch 10: 0.24s per image
- **Scaling**: 6x faster with batching

### Medium Images (1024x768)

- Batch 1: 1.51s per image
- Batch 10: 0.24s per image
- **Scaling**: 6.3x faster with batching

### Large Images (1920x1080)

- Batch 1: 1.57s per image
- Batch 10: 0.24s per image
- **Scaling**: 6.5x faster with batching

## Recommendations

1. **For API servers**: Use shared process pool for consistent low latency
1. **For batch scripts**: Use dynamic worker allocation
1. **For single operations**: Minimize workers to reduce overhead
1. **For large batches**: Maximize parallelization

## Code Changes

### Before

```python
# Fixed worker count
with ProcessPoolExecutor(max_workers=cpu_count - 1) as pool:
    ...
```

### After

```python
# Dynamic worker optimization
optimal_workers = get_optimal_worker_count(len(tasks), cpu_intensive=True)
with ProcessPoolExecutor(max_workers=optimal_workers) as pool:
    ...
```

## Final Performance vs Initial Baseline

- **Speed**: 93.6% faster (338.94s → 21.79s)
- **Memory**: 92.6% reduction (7GB → 520MB)
- **Throughput**: ~15x improvement
