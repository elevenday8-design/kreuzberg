# Kreuzberg Benchmarks

Performance benchmarking suite for the Kreuzberg text extraction library, with comprehensive testing capabilities.

## Features

- **Multiple Benchmark Types**: Baseline, statistical, serialization, and Tesseract-specific benchmarks
- **Comprehensive Performance Metrics**: Memory usage, CPU utilization, execution time, cache performance
- **Statistical Analysis**: Multiple trials with proper statistical reporting (mean, stdev, median)
- **Cache Performance Testing**: Cold vs warm cache comparison with speedup calculations
- **Serialization Benchmarking**: JSON vs msgpack performance comparison
- **Tesseract Format Comparison**: Different OCR output formats (text, hOCR, markdown, TSV)
- **Rich CLI Interface**: Beautiful terminal output with progress bars and tables
- **JSON Output**: Structured results for CI/CD integration and historical tracking

## Installation

```bash
cd benchmarks
uv sync
```

## Usage

### Benchmark Commands

```bash
# Run comprehensive benchmarks
python -m benchmarks run

# Run baseline cache performance test
python -m benchmarks baseline --output results/baseline.json

# Run statistical benchmark with multiple trials
python -m benchmarks statistical --trials 10 --output results/stats.json

# Run serialization performance test
python -m benchmarks serialization --output results/serialization.json

# Run Tesseract format comparison
python -m benchmarks tesseract --output-dir results/

# Compare two benchmark results
python -m benchmarks compare results/before.json results/after.json

# Analyze benchmark results
python -m benchmarks analyze results/benchmark.json --quality

# Custom test files directory
kreuzberg-bench run --test-files-dir ../tests/test_source_files
```

### Analysis

```bash
# Analyze benchmark results
kreuzberg-bench analyze results/latest.json

# Compare two benchmark runs
kreuzberg-bench compare results/run1.json results/run2.json

# Save comparison to file
kreuzberg-bench compare results/run1.json results/run2.json --output comparison.json
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "name": "kreuzberg_sync_vs_async",
  "timestamp": "2025-01-01T12:00:00",
  "system_info": {
    "platform": "macOS-15.5-arm64-arm-64bit",
    "python_version": "3.12.10",
    "cpu_count": 14,
    "memory_total_gb": 48.0
  },
  "summary": {
    "total_duration_seconds": 94.129,
    "total_benchmarks": 177,
    "successful_benchmarks": 57,
    "success_rate_percent": 32.2
  },
  "results": [
    {
      "name": "sync_pdf_small_default",
      "success": true,
      "performance": {
        "duration_seconds": 8.022,
        "memory_peak_mb": 27.8,
        "memory_average_mb": 25.1,
        "cpu_percent_average": 75.2,
        "cpu_percent_peak": 90.5,
        "gc_collections": {0: 2, 1: 1, 2: 0}
      },
      "metadata": {
        "file_type": "pdf",
        "config": "default"
      }
    }
  ]
}
```

## CI Integration

### GitHub Actions

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          cd benchmarks
          uv sync

      - name: Run benchmarks
        run: |
          cd benchmarks
          kreuzberg-bench run --output-dir ./results

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmarks/results/

      - name: Compare with baseline
        if: github.event_name == 'pull_request'
        run: |
          # Download baseline results from main branch
          # Compare and comment on PR
```
