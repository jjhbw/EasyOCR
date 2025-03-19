#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory leak detection script for EasyOCR's readtext_batched function.

This script runs multiple iterations of readtext_batched and monitors GPU memory
usage to detect potential memory leaks.
"""

import os
import gc
import time
import argparse
import numpy as np
import torch
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional


def get_gpu_memory_usage() -> Tuple[int, int, int]:
    """
    Get current GPU memory usage.
    
    Returns:
        Tuple containing (allocated memory in MB, reserved memory in MB, max memory in MB)
    """
    if not torch.cuda.is_available():
        return (0, 0, 0)
    
    # Get current memory allocation
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    return (allocated, reserved, max_memory)


def print_memory_stats(iteration: int, before: Tuple[int, int, int], after: Tuple[int, int, int]) -> None:
    """Print memory usage statistics before and after an operation."""
    print(f"Iteration {iteration}:")
    print(f"  Before - Allocated: {before[0]:.2f} MB, Reserved: {before[1]:.2f} MB, Max: {before[2]:.2f} MB")
    print(f"  After  - Allocated: {after[0]:.2f} MB, Reserved: {after[1]:.2f} MB, Max: {after[2]:.2f} MB")
    print(f"  Diff   - Allocated: {after[0] - before[0]:.2f} MB, Reserved: {after[1] - before[1]:.2f} MB")
    print()


def load_test_images(image_paths: List[str], n_width: Optional[int] = None, n_height: Optional[int] = None) -> List[np.ndarray]:
    """
    Load test images for OCR processing.
    
    Args:
        image_paths: List of paths to images
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
        
    Returns:
        List of numpy arrays containing the images
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        if n_width and n_height:
            img = img.resize((n_width, n_height))
        images.append(np.array(img))
    return images


def test_readtext_batched(
    reader: easyocr.Reader,
    images: List[np.ndarray],
    iterations: int = 10,
    batch_size: int = 1,
    n_width: Optional[int] = None,
    n_height: Optional[int] = None,
    cleanup_between_iterations: bool = False,
    plot_memory: bool = True
) -> Dict[str, List[float]]:
    """
    Test for memory leaks in readtext_batched.
    
    Args:
        reader: EasyOCR Reader instance
        images: List of images to process
        iterations: Number of iterations to run
        batch_size: Batch size for readtext_batched
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
        cleanup_between_iterations: Whether to run garbage collection between iterations
        plot_memory: Whether to plot memory usage over time
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_stats = {
        'allocated': [],
        'reserved': [],
        'max': []
    }
    
    # Force initial CUDA allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy = torch.ones((1,)).cuda()
        del dummy
    
    for i in range(iterations):
        # Clear cache before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory before
        before = get_gpu_memory_usage()
        memory_stats['allocated'].append(before[0])
        memory_stats['reserved'].append(before[1])
        memory_stats['max'].append(before[2])
        
        # Run OCR
        start_time = time.time()
        results = reader.readtext_batched(
            images, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        
        # Measure memory after
        after = get_gpu_memory_usage()
        
        # Print stats
        print_memory_stats(i+1, before, after)
        print(f"  Processing time: {elapsed:.2f} seconds")
        print(f"  Results: {len(results)} images processed")
        
        # Try to clean up
        if cleanup_between_iterations:
            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  Cleanup performed")
        
        # Sleep a bit to allow for any async operations to complete
        time.sleep(1)
    
    if plot_memory and iterations > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, iterations+1), memory_stats['allocated'], 'b-', label='Allocated')
        plt.plot(range(1, iterations+1), memory_stats['reserved'], 'r-', label='Reserved')
        plt.plot(range(1, iterations+1), memory_stats['max'], 'g--', label='Max Allocated')
        plt.xlabel('Iteration')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig('memory_usage.png')
        plt.close()
        print(f"Memory usage plot saved to 'memory_usage.png'")
    
    return memory_stats


def test_reader_recreation(
    lang_list: List[str],
    images: List[np.ndarray],
    iterations: int = 10,
    batch_size: int = 1,
    n_width: Optional[int] = None,
    n_height: Optional[int] = None,
    plot_memory: bool = True
) -> Dict[str, List[float]]:
    """
    Test if recreating the Reader object between iterations helps with memory leaks.
    
    Args:
        lang_list: List of languages for the Reader
        images: List of images to process
        iterations: Number of iterations to run
        batch_size: Batch size for readtext_batched
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
        plot_memory: Whether to plot memory usage over time
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_stats = {
        'allocated': [],
        'reserved': [],
        'max': []
    }
    
    # Force initial CUDA allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy = torch.ones((1,)).cuda()
        del dummy
    
    for i in range(iterations):
        # Clear cache before measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory before
        before = get_gpu_memory_usage()
        memory_stats['allocated'].append(before[0])
        memory_stats['reserved'].append(before[1])
        memory_stats['max'].append(before[2])
        
        # Create reader
        reader = easyocr.Reader(lang_list, gpu=True)
        
        # Run OCR
        start_time = time.time()
        results = reader.readtext_batched(
            images, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        
        # Explicitly delete reader
        del reader
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory after
        after = get_gpu_memory_usage()
        
        # Print stats
        print(f"Iteration {i+1} (with reader recreation):")
        print(f"  Before - Allocated: {before[0]:.2f} MB, Reserved: {before[1]:.2f} MB, Max: {before[2]:.2f} MB")
        print(f"  After  - Allocated: {after[0]:.2f} MB, Reserved: {after[1]:.2f} MB, Max: {after[2]:.2f} MB")
        print(f"  Diff   - Allocated: {after[0] - before[0]:.2f} MB, Reserved: {after[1] - before[1]:.2f} MB")
        print(f"  Processing time: {elapsed:.2f} seconds")
        print()
        
        # Sleep a bit to allow for any async operations to complete
        time.sleep(1)
    
    if plot_memory and iterations > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, iterations+1), memory_stats['allocated'], 'b-', label='Allocated')
        plt.plot(range(1, iterations+1), memory_stats['reserved'], 'r-', label='Reserved')
        plt.plot(range(1, iterations+1), memory_stats['max'], 'g--', label='Max Allocated')
        plt.xlabel('Iteration')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage Over Iterations (With Reader Recreation)')
        plt.legend()
        plt.grid(True)
        plt.savefig('memory_usage_recreation.png')
        plt.close()
        print(f"Memory usage plot saved to 'memory_usage_recreation.png'")
    
    return memory_stats


def main():
    parser = argparse.ArgumentParser(description='Test EasyOCR for memory leaks')
    parser.add_argument('--images', nargs='+', required=True, help='Paths to test images')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for readtext_batched')
    parser.add_argument('--width', type=int, default=None, help='Width to resize images to')
    parser.add_argument('--height', type=int, default=None, help='Height to resize images to')
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup between iterations')
    parser.add_argument('--test-recreation', action='store_true', help='Test recreating Reader between iterations')
    parser.add_argument('--no-plot', action='store_true', help='Disable memory usage plotting')
    
    args = parser.parse_args()
    
    # Load test images
    print(f"Loading {len(args.images)} test images...")
    images = load_test_images(args.images, args.width, args.height)
    print(f"Images loaded. Shapes: {[img.shape for img in images]}")
    
    if args.test_recreation:
        print("\n=== Testing with Reader recreation between iterations ===\n")
        test_reader_recreation(
            args.languages,
            images,
            iterations=args.iterations,
            batch_size=args.batch_size,
            n_width=args.width,
            n_height=args.height,
            plot_memory=not args.no_plot
        )
    else:
        print("\n=== Testing with persistent Reader ===\n")
        # Create reader
        print(f"Creating EasyOCR Reader with languages: {args.languages}")
        reader = easyocr.Reader(args.languages, gpu=True)
        
        # Run test
        test_readtext_batched(
            reader,
            images,
            iterations=args.iterations,
            batch_size=args.batch_size,
            n_width=args.width,
            n_height=args.height,
            cleanup_between_iterations=args.cleanup,
            plot_memory=not args.no_plot
        )


if __name__ == "__main__":
    main()
