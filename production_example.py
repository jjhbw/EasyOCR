#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of using the patched EasyOCR in a production environment.

This script demonstrates how to use the patched EasyOCR to process
a large number of images without memory leaks.
"""

import os
import gc
import time
import argparse
import numpy as np
import torch
from PIL import Image
from patched_easyocr import create_reader


def process_images_batch(reader, image_paths, batch_size=10, n_width=None, n_height=None):
    """
    Process a batch of images with the patched EasyOCR reader.
    
    Args:
        reader: EasyOCR reader instance
        image_paths: List of paths to images
        batch_size: Number of images to process at once
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
        
    Returns:
        List of OCR results for each image
    """
    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        # Load images
        batch_images = []
        for path in batch_paths:
            img = Image.open(path)
            if n_width and n_height:
                img = img.resize((n_width, n_height))
            batch_images.append(np.array(img))
        
        # Process batch
        start_time = time.time()
        batch_results = reader.readtext_batched(batch_images, n_width=n_width, n_height=n_height)
        elapsed = time.time() - start_time
        
        # Store results
        results.extend(batch_results)
        
        print(f"  Processed {len(batch_paths)} images in {elapsed:.2f} seconds")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  GPU memory usage: {memory:.2f} MB")
        
        # Clean up
        del batch_images
        del batch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def process_directory(reader, directory, output_file=None, batch_size=10, n_width=None, n_height=None):
    """
    Process all images in a directory.
    
    Args:
        reader: EasyOCR reader instance
        directory: Directory containing images
        output_file: Optional file to write results to
        batch_size: Number of images to process at once
        n_width: Optional width to resize images to
        n_height: Optional height to resize images to
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images in {directory}")
    
    # Process images
    results = process_images_batch(reader, image_paths, batch_size, n_width, n_height)
    
    # Write results to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (path, result) in enumerate(zip(image_paths, results)):
                f.write(f"File: {path}\n")
                for detection in result:
                    bbox, text, confidence = detection
                    f.write(f"  Text: {text}, Confidence: {confidence:.2f}, BBox: {bbox}\n")
                f.write("\n")
        
        print(f"Results written to {output_file}")
    
    return results


def monitor_memory_usage(interval=1.0, duration=60.0):
    """
    Monitor GPU memory usage over time.
    
    Args:
        interval: Time between measurements in seconds
        duration: Total monitoring duration in seconds
    """
    if not torch.cuda.is_available():
        print("CUDA not available, cannot monitor GPU memory")
        return
    
    import matplotlib.pyplot as plt
    
    # Initialize data structures
    timestamps = []
    allocated = []
    reserved = []
    
    start_time = time.time()
    end_time = start_time + duration
    
    print(f"Monitoring GPU memory usage for {duration} seconds...")
    
    # Collect data
    while time.time() < end_time:
        current_time = time.time() - start_time
        timestamps.append(current_time)
        
        allocated.append(torch.cuda.memory_allocated() / (1024 * 1024))
        reserved.append(torch.cuda.memory_reserved() / (1024 * 1024))
        
        print(f"Time: {current_time:.1f}s, Allocated: {allocated[-1]:.2f} MB, Reserved: {reserved[-1]:.2f} MB")
        
        time.sleep(interval)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, allocated, 'b-', label='Allocated')
    plt.plot(timestamps, reserved, 'r-', label='Reserved')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (MB)')
    plt.title('GPU Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_monitoring.png')
    plt.close()
    
    print(f"Memory usage plot saved to 'memory_monitoring.png'")


def main():
    parser = argparse.ArgumentParser(description='Production example for patched EasyOCR')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--mode', choices=['patched', 'process', 'none'], default='patched',
                        help='Isolation mode')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--width', type=int, default=None, help='Width to resize images to')
    parser.add_argument('--height', type=int, default=None, help='Height to resize images to')
    parser.add_argument('--monitor', action='store_true', help='Monitor memory usage after processing')
    parser.add_argument('--monitor-duration', type=float, default=60.0, 
                        help='Duration to monitor memory usage (seconds)')
    
    args = parser.parse_args()
    
    # Create reader
    print(f"Creating EasyOCR Reader with languages: {args.languages}, mode: {args.mode}")
    reader = create_reader(args.languages, isolation_mode=args.mode, gpu=True)
    
    # Process input
    if os.path.isdir(args.input):
        process_directory(
            reader, 
            args.input, 
            args.output, 
            args.batch_size, 
            args.width, 
            args.height
        )
    else:
        # Single image
        img = np.array(Image.open(args.input))
        if args.width and args.height:
            img = Image.fromarray(img).resize((args.width, args.height))
            img = np.array(img)
        
        # Process image
        results = reader.readtext_batched([img])
        
        # Print results
        print(f"\nResults for {args.input}:")
        for detection in results[0]:
            bbox, text, confidence = detection
            print(f"  Text: {text}, Confidence: {confidence:.2f}, BBox: {bbox}")
        
        # Write results to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"File: {args.input}\n")
                for detection in results[0]:
                    bbox, text, confidence = detection
                    f.write(f"  Text: {text}, Confidence: {confidence:.2f}, BBox: {bbox}\n")
            
            print(f"Results written to {args.output}")
    
    # Monitor memory usage if requested
    if args.monitor:
        monitor_memory_usage(interval=1.0, duration=args.monitor_duration)


if __name__ == "__main__":
    main()
