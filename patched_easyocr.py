#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patched version of EasyOCR with memory leak fixes for readtext_batched.

This module provides a wrapper around EasyOCR that implements fixes for
the memory leak in the readtext_batched method.
"""

import gc
import torch
import easyocr
import numpy as np
from typing import List, Optional, Union, Dict, Any


class PatchedReader:
    """
    A wrapper around EasyOCR's Reader class that implements memory leak fixes.
    
    This class provides the same interface as EasyOCR's Reader but with
    additional memory management to prevent VRAM leaks.
    """
    
    def __init__(self, lang_list, **kwargs):
        """
        Initialize the PatchedReader with the same parameters as EasyOCR's Reader.
        
        Args:
            lang_list: List of language codes
            **kwargs: Additional arguments to pass to EasyOCR's Reader
        """
        self.lang_list = lang_list
        self.kwargs = kwargs
        self._reader = None
        self._create_reader()
    
    def _create_reader(self):
        """Create a new EasyOCR Reader instance."""
        # Clean up old reader if it exists
        if self._reader is not None:
            # Explicitly delete components that might hold references
            if hasattr(self._reader, 'detector'):
                del self._reader.detector
            if hasattr(self._reader, 'recognizer'):
                del self._reader.recognizer
            if hasattr(self._reader, 'converter'):
                del self._reader.converter
            
            # Delete the reader itself
            del self._reader
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create new reader
        self._reader = easyocr.Reader(self.lang_list, **self.kwargs)
    
    def _cleanup_tensors(self, *tensors):
        """
        Clean up PyTorch tensors to prevent memory leaks.
        
        Args:
            *tensors: PyTorch tensors to clean up
        """
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                # Move tensor to CPU if it's on GPU
                if tensor.device.type != 'cpu':
                    tensor = tensor.cpu()
                # Delete the tensor
                del tensor
    
    def readtext_batched(self, image, n_width=None, n_height=None, **kwargs):
        """
        Patched version of readtext_batched that prevents memory leaks.
        
        Args:
            image: Input image or list of images
            n_width: Optional width to resize images to
            n_height: Optional height to resize images to
            **kwargs: Additional arguments to pass to readtext_batched
            
        Returns:
            List of OCR results
        """
        try:
            # Run OCR
            results = self._reader.readtext_batched(
                image, 
                n_width=n_width, 
                n_height=n_height, 
                **kwargs
            )
            
            # Make a deep copy of the results to ensure we don't have references to tensors
            copied_results = []
            for img_result in results:
                img_copied = []
                for detection in img_result:
                    # Each detection is typically [bbox, text, confidence]
                    # Make sure we copy the bbox which might be a tensor or numpy array
                    if isinstance(detection[0], (torch.Tensor, np.ndarray)):
                        bbox_copy = detection[0].copy() if isinstance(detection[0], np.ndarray) else detection[0].clone().cpu()
                    else:
                        # If it's already a list/tuple, just copy it
                        bbox_copy = [coord for coord in detection[0]]
                    
                    # Create a new detection with the copied bbox
                    img_copied.append([bbox_copy, detection[1], detection[2]])
                
                copied_results.append(img_copied)
            
            return copied_results
            
        except Exception as e:
            # If an error occurs, recreate the reader to avoid leaving it in a bad state
            self._create_reader()
            raise e
        
        finally:
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying Reader instance.
        
        This allows the PatchedReader to be used as a drop-in replacement
        for EasyOCR's Reader.
        
        Args:
            name: Name of the attribute to access
            
        Returns:
            The requested attribute from the underlying Reader
        """
        # Only delegate if the attribute exists on the reader
        if self._reader is not None and hasattr(self._reader, name):
            return getattr(self._reader, name)
        
        # Otherwise, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ProcessIsolatedReader:
    """
    A version of EasyOCR Reader that runs in a separate process to prevent memory leaks.
    
    This approach completely isolates the OCR process, ensuring that any memory leaks
    are cleaned up when the process terminates.
    """
    
    def __init__(self, lang_list, **kwargs):
        """
        Initialize the ProcessIsolatedReader.
        
        Args:
            lang_list: List of language codes
            **kwargs: Additional arguments to pass to EasyOCR's Reader
        """
        import multiprocessing as mp
        
        self.lang_list = lang_list
        self.kwargs = kwargs
        self.mp_context = mp.get_context('spawn')  # Use spawn for better compatibility
    
    def _worker_init(self, lang_list, kwargs):
        """
        Initialize the EasyOCR Reader in the worker process.
        
        Args:
            lang_list: List of language codes
            kwargs: Additional arguments for the Reader
            
        Returns:
            Initialized EasyOCR Reader
        """
        return easyocr.Reader(lang_list, **kwargs)
    
    def _worker_readtext_batched(self, reader, image, n_width, n_height, kwargs):
        """
        Run readtext_batched in the worker process.
        
        Args:
            reader: EasyOCR Reader instance
            image: Input image or list of images
            n_width: Optional width to resize images to
            n_height: Optional height to resize images to
            kwargs: Additional arguments for readtext_batched
            
        Returns:
            OCR results
        """
        try:
            return reader.readtext_batched(image, n_width=n_width, n_height=n_height, **kwargs)
        finally:
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def readtext_batched(self, image, n_width=None, n_height=None, **kwargs):
        """
        Run readtext_batched in a separate process.
        
        Args:
            image: Input image or list of images
            n_width: Optional width to resize images to
            n_height: Optional height to resize images to
            **kwargs: Additional arguments for readtext_batched
            
        Returns:
            OCR results
        """
        with self.mp_context.Pool(1) as pool:
            # Initialize reader in worker process
            reader = pool.apply(self._worker_init, (self.lang_list, self.kwargs))
            
            # Run OCR in worker process
            results = pool.apply(
                self._worker_readtext_batched, 
                (reader, image, n_width, n_height, kwargs)
            )
            
            return results


def create_reader(lang_list, isolation_mode='patched', **kwargs):
    """
    Factory function to create an appropriate Reader based on the isolation mode.
    
    Args:
        lang_list: List of language codes
        isolation_mode: Mode of isolation ('patched', 'process', or 'none')
        **kwargs: Additional arguments for the Reader
        
    Returns:
        Reader instance
    """
    if isolation_mode == 'patched':
        return PatchedReader(lang_list, **kwargs)
    elif isolation_mode == 'process':
        return ProcessIsolatedReader(lang_list, **kwargs)
    elif isolation_mode == 'none':
        return easyocr.Reader(lang_list, **kwargs)
    else:
        raise ValueError(f"Unknown isolation mode: {isolation_mode}")


# Example usage
if __name__ == "__main__":
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description='Test patched EasyOCR')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--languages', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--mode', choices=['patched', 'process', 'none'], default='patched',
                        help='Isolation mode')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run')
    
    args = parser.parse_args()
    
    # Load test image
    img = np.array(Image.open(args.image))
    
    # Create reader
    reader = create_reader(args.languages, isolation_mode=args.mode, gpu=True)
    
    # Run OCR multiple times
    for i in range(args.iterations):
        print(f"Iteration {i+1}:")
        
        # Get current memory usage
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  Before OCR: {before:.2f} MB")
        
        # Run OCR
        results = reader.readtext_batched(img)
        print(f"  Results: {len(results)} images processed")
        
        # Get memory usage after OCR
        if torch.cuda.is_available():
            after = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  After OCR: {after:.2f} MB")
            print(f"  Diff: {after - before:.2f} MB")
        
        # Clean up
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Get memory usage after cleanup
            after_cleanup = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"  After cleanup: {after_cleanup:.2f} MB")
            print(f"  Leak: {after_cleanup - before:.2f} MB")
        
        print()
