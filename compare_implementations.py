#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare the original EasyOCR implementation with the patched version.

This script runs a side-by-side comparison of the original EasyOCR Reader
and the patched version to demonstrate the memory leak fix.
"""

import gc
import time
import argparse
import numpy as np
import torch
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
from patched_easyocr import create_reader


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0
    
    return torch.cuda.memory_allocated() / (1024 * 1024)


def run_comparison(
    image_path,
    languages=['en'],
    iterations=10,
    batch_size=1,
    n_width=None,
    n_height=None
):
    """
    Run a comparison between original and patched EasyOCR.
    
    Args:
        image_path: Path to test image
        languages: List of languages for OCR
        iterations: Number of iterations to run
        batch_size: Batch size for readtext_batched
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
    """
    # Load test image
    img = np.array(Image.open(image_path))
    if n_width and n_height:
        img = Image.fromarray(img).resize((n_width, n_height))
        img = np.array(img)
    
    # Create readers
    print("Creating original EasyOCR Reader...")
    original_reader = easyocr.Reader(languages, gpu=True)
    
    print("Creating patched EasyOCR Reader...")
    patched_reader = create_reader(languages, isolation_mode='patched', gpu=True)
    
    # Initialize memory tracking
    original_memory = []
    patched_memory = []
    
    # Force initial CUDA allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy = torch.ones((1,)).cuda()
        del dummy
    
    # Run comparison
    print(f"\nRunning comparison for {iterations} iterations...")
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Test original implementation
        print("Testing original implementation:")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Measure memory before
        before_original = get_gpu_memory_usage()
        print(f"  Before OCR: {before_original:.2f} MB")
        
        # Run OCR
        start_time = time.time()
        original_results = original_reader.readtext_batched(
            img, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        original_time = time.time() - start_time
        
        # Measure memory after
        after_original = get_gpu_memory_usage()
        print(f"  After OCR: {after_original:.2f} MB")
        print(f"  Diff: {after_original - before_original:.2f} MB")
        print(f"  Time: {original_time:.2f} seconds")
        
        # Clean up
        del original_results
        gc.collect()
        torch.cuda.empty_cache()
        
        # Measure memory after cleanup
        cleanup_original = get_gpu_memory_usage()
        print(f"  After cleanup: {cleanup_original:.2f} MB")
        print(f"  Leak: {cleanup_original - before_original:.2f} MB")
        
        # Store memory usage
        original_memory.append(cleanup_original - before_original)
        
        # Test patched implementation
        print("\nTesting patched implementation:")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Measure memory before
        before_patched = get_gpu_memory_usage()
        print(f"  Before OCR: {before_patched:.2f} MB")
        
        # Run OCR
        start_time = time.time()
        patched_results = patched_reader.readtext_batched(
            img, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        patched_time = time.time() - start_time
        
        # Measure memory after
        after_patched = get_gpu_memory_usage()
        print(f"  After OCR: {after_patched:.2f} MB")
        print(f"  Diff: {after_patched - before_patched:.2f} MB")
        print(f"  Time: {patched_time:.2f} seconds")
        
        # Clean up
        del patched_results
        gc.collect()
        torch.cuda.empty_cache()
        
        # Measure memory after cleanup
        cleanup_patched = get_gpu_memory_usage()
        print(f"  After cleanup: {cleanup_patched:.2f} MB")
        print(f"  Leak: {cleanup_patched - before_patched:.2f} MB")
        
        # Store memory usage
        patched_memory.append(cleanup_patched - before_patched)
        
        # Sleep a bit to allow for any async operations to complete
        time.sleep(1)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations+1), original_memory, 'r-', label='Original')
    plt.plot(range(1, iterations+1), patched_memory, 'g-', label='Patched')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Leak (MB)')
    plt.title('Memory Leak Comparison: Original vs Patched')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_results.png')
    plt.close()
    
    print(f"\nComparison results saved to 'comparison_results.png'")
    
    # Print summary
    print("\nSummary:")
    print(f"  Original implementation average leak: {sum(original_memory)/len(original_memory):.2f} MB")
    print(f"  Patched implementation average leak: {sum(patched_memory)/len(patched_memory):.2f} MB")
    print(f"  Improvement: {sum(original_memory)/len(original_memory) - sum(patched_memory)/len(patched_memory):.2f} MB")


def run_process_isolation_test(
    image_path,
    languages=['en'],
    iterations=5,
    batch_size=1,
    n_width=None,
    n_height=None
):
    """
    Test the process isolation approach.
    
    Args:
        image_path: Path to test image
        languages: List of languages for OCR
        iterations: Number of iterations to run
        batch_size: Batch size for readtext_batched
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
    """
    # Load test image
    img = np.array(Image.open(image_path))
    if n_width and n_height:
        img = Image.fromarray(img).resize((n_width, n_height))
        img = np.array(img)
    
    # Create reader
    print("Creating process-isolated EasyOCR Reader...")
    process_reader = create_reader(languages, isolation_mode='process', gpu=True)
    
    # Initialize memory tracking
    process_memory = []
    
    # Force initial CUDA allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy = torch.ones((1,)).cuda()
        del dummy
    
    # Run test
    print(f"\nRunning process isolation test for {iterations} iterations...")
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Clean up before test
        torch.cuda.empty_cache()
        gc.collect()
        
        # Measure memory before
        before = get_gpu_memory_usage()
        print(f"  Before OCR: {before:.2f} MB")
        
        # Run OCR
        start_time = time.time()
        results = process_reader.readtext_batched(
            img, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        
        # Measure memory after
        after = get_gpu_memory_usage()
        print(f"  After OCR: {after:.2f} MB")
        print(f"  Diff: {after - before:.2f} MB")
        print(f"  Time: {elapsed:.2f} seconds")
        
        # Clean up
        del results
        gc.collect()
        torch.cuda.empty_cache()
        
        # Measure memory after cleanup
        cleanup = get_gpu_memory_usage()
        print(f"  After cleanup: {cleanup:.2f} MB")
        print(f"  Leak: {cleanup - before:.2f} MB")
        
        # Store memory usage
        process_memory.append(cleanup - before)
        
        # Sleep a bit to allow for any async operations to complete
        time.sleep(1)
    
    # Print summary
    print("\nProcess Isolation Summary:")
    print(f"  Average leak: {sum(process_memory)/len(process_memory):.2f} MB")
    print(f"  Average processing time: {elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Compare original and patched EasyOCR')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for readtext_batched')
    parser.add_argument('--width', type=int, default=None, help='Width to resize images to')
    parser.add_argument('--height', type=int, default=None, help='Height to resize images to')
    parser.add_argument('--test-process', action='store_true', help='Test process isolation approach')
    
    args = parser.parse_args()
    
    if args.test_process:
        run_process_isolation_test(
            args.image,
            args.languages,
            args.iterations,
            args.batch_size,
            args.width,
            args.height
        )
    else:
        run_comparison(
            args.image,
            args.languages,
            args.iterations,
            args.batch_size,
            args.width,
            args.height
        )


if __name__ == "__main__":
    main()
