# EasyOCR Memory Leak Testing Tools

This directory contains scripts for testing and analyzing memory leaks in EasyOCR's `readtext_batched` function.

## Background

We've observed VRAM (GPU memory) leaks in production when running EasyOCR's `readtext_batched` method. Previous attempts to mitigate this with `del easyocr.Reader` followed by `gc.collect()` and `torch.cuda.empty_cache()` were unsuccessful.

## Scripts

### 1. memory_leak_check.py

This script runs multiple iterations of `readtext_batched` and monitors GPU memory usage to detect potential memory leaks.

#### Features:
- Tracks allocated, reserved, and max memory usage over multiple iterations
- Tests with both persistent Reader object and recreating Reader between iterations
- Generates memory usage plots for visualization
- Supports optional cleanup between iterations

#### Usage:

```bash
python memory_leak_check.py --images examples/english.png examples/french.jpg \
                           --languages en fr \
                           --iterations 20 \
                           --batch-size 2 \
                           --cleanup
```

To test if recreating the Reader object between iterations helps:

```bash
python memory_leak_check.py --images examples/english.png \
                           --languages en \
                           --iterations 10 \
                           --test-recreation
```

### 2. memory_leak_analysis.py

This script provides more advanced analysis tools to identify the source of memory leaks.

#### Features:
- Detailed analysis of Reader object memory usage
- Tracing of memory allocations during execution
- Testing for CUDA memory fragmentation
- Analysis of intermediate tensor allocations

#### Usage:

Basic memory leak test:
```bash
python memory_leak_analysis.py --images examples/english.png
```

Analyze Reader object memory:
```bash
python memory_leak_analysis.py --images examples/english.png --analyze-reader
```

Trace memory allocations:
```bash
python memory_leak_analysis.py --images examples/english.png --trace-memory
```

Test for CUDA memory fragmentation:
```bash
python memory_leak_analysis.py --images examples/english.png --test-fragmentation --iterations 5
```

## Common Options

Both scripts support the following common options:

- `--images`: Paths to test images (required)
- `--languages`: Languages for OCR (default: en)
- `--width`, `--height`: Resize images to these dimensions (optional)
- `--batch-size`: Batch size for readtext_batched (default: 1)

## Requirements

- EasyOCR
- PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)

## Interpreting Results

### Memory Leak Detection

A memory leak is indicated by:
1. Steadily increasing memory usage across iterations
2. Memory not being released after cleanup operations
3. Significant difference between initial memory usage and final usage after cleanup

### Potential Causes of Memory Leaks

1. **Tensor References**: PyTorch tensors not being properly released
2. **CUDA Memory Fragmentation**: Memory fragmentation preventing efficient reuse
3. **Cached Operations**: PyTorch caching mechanisms retaining references
4. **Python Circular References**: Objects with circular references not being garbage collected

## Mitigation Strategies

If a memory leak is confirmed, consider:

1. Implementing a custom wrapper around EasyOCR that recreates the Reader object periodically
2. Modifying the EasyOCR source code to explicitly free resources
3. Using a process-based approach where each OCR operation runs in a separate process
4. Setting environment variables like `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (if applicable)

## Example Workflow

1. Run `memory_leak_check.py` to confirm if there's a memory leak
2. If a leak is detected, use `memory_leak_analysis.py` to identify the source
3. Based on the analysis, implement an appropriate mitigation strategy
