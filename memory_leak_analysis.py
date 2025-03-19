#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced memory leak analysis for EasyOCR's readtext_batched function.

This script performs a detailed analysis of memory usage patterns and object references
to identify the source of memory leaks in EasyOCR.
"""

import os
import gc
import sys
import time
import argparse
import numpy as np
import torch
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import tracemalloc
from typing import List, Tuple, Dict, Any, Optional, Set


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get detailed GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "max_allocated": 0,
            "max_reserved": 0
        }
    
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),  # MB
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),    # MB
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 * 1024),  # MB
        "max_reserved": torch.cuda.max_memory_reserved() / (1024 * 1024)     # MB
    }


def analyze_tensor_memory(prefix="", obj=None, visited=None, depth=0, max_depth=3):
    """
    Recursively analyze an object to find PyTorch tensors and their memory usage.
    
    Args:
        prefix: String prefix for output
        obj: Object to analyze
        visited: Set of already visited object ids
        depth: Current recursion depth
        max_depth: Maximum recursion depth
        
    Returns:
        Total memory usage in bytes
    """
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return 0
    
    if id(obj) in visited:
        return 0
    
    visited.add(id(obj))
    total_memory = 0
    
    if isinstance(obj, torch.Tensor):
        memory = obj.element_size() * obj.nelement()
        device = obj.device
        shape = list(obj.shape)
        dtype = obj.dtype
        requires_grad = obj.requires_grad
        
        print(f"{prefix}Tensor: shape={shape}, dtype={dtype}, device={device}, "
              f"requires_grad={requires_grad}, memory={memory/1024/1024:.2f} MB")
        
        total_memory += memory
        return total_memory
    
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            total_memory += analyze_tensor_memory(
                f"{prefix}[{i}].", item, visited, depth + 1, max_depth
            )
        return total_memory
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            total_memory += analyze_tensor_memory(
                f"{prefix}{k}.", v, visited, depth + 1, max_depth
            )
        return total_memory
    
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):  # Skip private attributes
                total_memory += analyze_tensor_memory(
                    f"{prefix}{k}.", v, visited, depth + 1, max_depth
                )
    
    return total_memory


def analyze_reader_memory(reader):
    """Analyze memory usage of an EasyOCR Reader object."""
    print("\n=== EasyOCR Reader Memory Analysis ===")
    
    # Analyze detector
    if hasattr(reader, "detector"):
        print("\n--- Detector ---")
        detector_memory = analyze_tensor_memory("detector.", reader.detector)
        print(f"Total detector memory: {detector_memory/1024/1024:.2f} MB")
    
    # Analyze recognizer
    if hasattr(reader, "recognizer"):
        print("\n--- Recognizer ---")
        recognizer_memory = analyze_tensor_memory("recognizer.", reader.recognizer)
        print(f"Total recognizer memory: {recognizer_memory/1024/1024:.2f} MB")
    
    # Analyze converter
    if hasattr(reader, "converter"):
        print("\n--- Converter ---")
        converter_memory = analyze_tensor_memory("converter.", reader.converter)
        print(f"Total converter memory: {converter_memory/1024/1024:.2f} MB")


def trace_readtext_batched_memory(
    reader: easyocr.Reader,
    images: List[np.ndarray],
    n_width: Optional[int] = None,
    n_height: Optional[int] = None,
    batch_size: int = 1
):
    """
    Trace memory allocations during readtext_batched execution.
    
    Args:
        reader: EasyOCR Reader instance
        images: List of images to process
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
        batch_size: Batch size for readtext_batched
    """
    # Start tracing memory allocations
    tracemalloc.start()
    
    # Get current memory snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run OCR
    results = reader.readtext_batched(
        images, 
        n_width=n_width,
        n_height=n_height,
        batch_size=batch_size
    )
    
    # Get memory snapshot after OCR
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("\n=== Top 20 memory allocations during readtext_batched ===")
    for stat in top_stats[:20]:
        print(stat)
    
    # Stop tracing
    tracemalloc.stop()
    
    return results


def test_for_cuda_memory_fragmentation(
    reader: easyocr.Reader,
    images: List[np.ndarray],
    iterations: int = 10,
    batch_size: int = 1,
    n_width: Optional[int] = None,
    n_height: Optional[int] = None
):
    """
    Test for CUDA memory fragmentation by allocating large tensors between OCR calls.
    
    Args:
        reader: EasyOCR Reader instance
        images: List of images to process
        iterations: Number of iterations to run
        batch_size: Batch size for readtext_batched
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping fragmentation test")
        return
    
    print("\n=== Testing for CUDA memory fragmentation ===")
    
    # Get initial memory state
    torch.cuda.empty_cache()
    initial_memory = get_gpu_memory_usage()
    print(f"Initial memory state: {initial_memory}")
    
    # Run OCR iterations with large tensor allocations in between
    for i in range(iterations):
        print(f"\nIteration {i+1}:")
        
        # Run OCR
        results = reader.readtext_batched(
            images, 
            n_width=n_width,
            n_height=n_height,
            batch_size=batch_size
        )
        
        # Check memory after OCR
        after_ocr = get_gpu_memory_usage()
        print(f"After OCR: {after_ocr}")
        
        # Try to allocate a large tensor
        try:
            # Calculate size to use 75% of available memory
            available = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            size = int(np.sqrt(0.75 * available * 1024 * 1024 / 4))  # 4 bytes per float32
            
            print(f"Attempting to allocate a {size}x{size} tensor ({size*size*4/1024/1024:.2f} MB)")
            large_tensor = torch.zeros((size, size), device='cuda')
            
            # Check memory after allocation
            after_alloc = get_gpu_memory_usage()
            print(f"After large allocation: {after_alloc}")
            
            # Free the tensor
            del large_tensor
            torch.cuda.empty_cache()
            
            # Check memory after freeing
            after_free = get_gpu_memory_usage()
            print(f"After freeing large tensor: {after_free}")
            
        except RuntimeError as e:
            print(f"Failed to allocate large tensor: {e}")
            
            # Try to recover
            torch.cuda.empty_cache()
            after_recovery = get_gpu_memory_usage()
            print(f"After recovery attempt: {after_recovery}")
        
        # Clean up OCR results
        del results
        gc.collect()
        torch.cuda.empty_cache()


def analyze_intermediate_tensors(
    reader: easyocr.Reader,
    images: List[np.ndarray],
    n_width: Optional[int] = None,
    n_height: Optional[int] = None
):
    """
    Analyze intermediate tensors created during readtext_batched.
    
    This function monkey patches torch.Tensor creation to track large allocations.
    
    Args:
        reader: EasyOCR Reader instance
        images: List of images to process
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
    """
    print("\n=== Analyzing intermediate tensor allocations ===")
    
    # Store original tensor constructor
    original_tensor = torch.Tensor
    tensor_sizes = []
    
    # Define a wrapper to track tensor creation
    def tensor_tracking_wrapper(*args, **kwargs):
        tensor = original_tensor(*args, **kwargs)
        
        # Only track tensors above a certain size (1MB)
        size_bytes = tensor.element_size() * tensor.nelement()
        if size_bytes > 1024 * 1024:
            tensor_sizes.append({
                'shape': tensor.shape,
                'size_mb': size_bytes / (1024 * 1024),
                'device': tensor.device,
                'dtype': tensor.dtype,
                'stack': ''.join(traceback.format_stack()[-10:-1])  # Get partial stack trace
            })
        
        return tensor
    
    # Monkey patch tensor constructor
    torch.Tensor = tensor_tracking_wrapper
    
    try:
        # Run OCR
        reader.readtext_batched(
            images, 
            n_width=n_width,
            n_height=n_height,
            batch_size=1
        )
    finally:
        # Restore original tensor constructor
        torch.Tensor = original_tensor
    
    # Sort tensors by size
    tensor_sizes.sort(key=lambda x: x['size_mb'], reverse=True)
    
    # Print top 20 largest tensors
    print("\nTop 20 largest tensor allocations:")
    for i, tensor_info in enumerate(tensor_sizes[:20]):
        print(f"{i+1}. Shape: {tensor_info['shape']}, "
              f"Size: {tensor_info['size_mb']:.2f} MB, "
              f"Device: {tensor_info['device']}, "
              f"Dtype: {tensor_info['dtype']}")
        # Uncomment to see stack traces
        # print(f"Stack trace:\n{tensor_info['stack']}\n")


def main():
    parser = argparse.ArgumentParser(description='Advanced memory leak analysis for EasyOCR')
    parser.add_argument('--images', nargs='+', required=True, help='Paths to test images')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--width', type=int, default=None, help='Width to resize images to')
    parser.add_argument('--height', type=int, default=None, help='Height to resize images to')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for readtext_batched')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for fragmentation test')
    parser.add_argument('--analyze-reader', action='store_true', help='Analyze Reader object memory usage')
    parser.add_argument('--trace-memory', action='store_true', help='Trace memory allocations during execution')
    parser.add_argument('--test-fragmentation', action='store_true', help='Test for CUDA memory fragmentation')
    parser.add_argument('--analyze-tensors', action='store_true', help='Analyze intermediate tensor allocations')
    
    args = parser.parse_args()
    
    # Load test images
    print(f"Loading {len(args.images)} test images...")
    images = []
    for path in args.images:
        img = Image.open(path)
        if args.width and args.height:
            img = img.resize((args.width, args.height))
        images.append(np.array(img))
    print(f"Images loaded. Shapes: {[img.shape for img in images]}")
    
    # Create reader
    print(f"Creating EasyOCR Reader with languages: {args.languages}")
    reader = easyocr.Reader(args.languages, gpu=torch.cuda.is_available())
    
    # Run selected analyses
    if args.analyze_reader:
        analyze_reader_memory(reader)
    
    if args.trace_memory:
        trace_readtext_batched_memory(
            reader,
            images,
            n_width=args.width,
            n_height=args.height,
            batch_size=args.batch_size
        )
    
    if args.test_fragmentation:
        test_for_cuda_memory_fragmentation(
            reader,
            images,
            iterations=args.iterations,
            batch_size=args.batch_size,
            n_width=args.width,
            n_height=args.height
        )
    
    if args.analyze_tensors:
        analyze_intermediate_tensors(
            reader,
            images,
            n_width=args.width,
            n_height=args.height
        )
    
    # If no specific analysis was selected, run a basic test
    if not any([args.analyze_reader, args.trace_memory, args.test_fragmentation, args.analyze_tensors]):
        print("\n=== Running basic memory leak test ===")
        
        # Get initial memory state
        torch.cuda.empty_cache()
        initial = get_gpu_memory_usage()
        print(f"Initial memory state: {initial}")
        
        # Run OCR
        results = reader.readtext_batched(
            images, 
            n_width=args.width,
            n_height=args.height,
            batch_size=args.batch_size
        )
        
        # Check memory after OCR
        after_ocr = get_gpu_memory_usage()
        print(f"After OCR: {after_ocr}")
        
        # Clean up and check memory again
        del results
        gc.collect()
        torch.cuda.empty_cache()
        after_cleanup = get_gpu_memory_usage()
        print(f"After cleanup: {after_cleanup}")
        
        # Calculate memory leak
        leak = after_cleanup["allocated"] - initial["allocated"]
        print(f"\nPotential memory leak: {leak:.2f} MB")
        
        if leak > 1.0:  # More than 1MB leak
            print("WARNING: Potential memory leak detected!")
        else:
            print("No significant memory leak detected in this single run.")
            print("Run memory_leak_check.py for a more thorough test with multiple iterations.")


if __name__ == "__main__":
    main()
