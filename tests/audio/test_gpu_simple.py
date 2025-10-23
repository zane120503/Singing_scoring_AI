#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple GPU test for Audio Separator
"""

import sys
import os
import torch
import time
import subprocess

def check_gpu_before():
    """Check GPU status before processing"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(', ')
        print(f"GPU Before: Usage={gpu_info[0]}%, Memory={gpu_info[1]}MB")
    return result.stdout.strip()

def check_gpu_after():
    """Check GPU status after processing"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(', ')
        print(f"GPU After: Usage={gpu_info[0]}%, Memory={gpu_info[1]}MB")
    return result.stdout.strip()

def test_gpu_usage():
    """Test if GPU is actually being used"""
    print("=== GPU USAGE TEST ===")
    
    # Check initial GPU status
    print("1. Checking initial GPU status...")
    check_gpu_before()
    
    # Check PyTorch GPU status
    print(f"2. PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"   GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Test PyTorch GPU allocation
    print("3. Testing PyTorch GPU allocation...")
    if torch.cuda.is_available():
        # Create a tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        print(f"   Tensor created on GPU: {x.device}")
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Perform some computation
        y = torch.matmul(x, x)
        print(f"   Computation completed on GPU: {y.device}")
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Check GPU usage after computation
        check_gpu_after()
        
        # Clean up
        del x, y
        torch.cuda.empty_cache()
        print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Test Audio Separator GPU usage
    print("4. Testing Audio Separator GPU usage...")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Audio_separator_ui'))
        from app import process_uvr_task
        
        print("   Audio Separator imported successfully")
        
        # Check device detection in Audio Separator
        device_base = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Audio Separator device base: {device_base}")
        
        # Test with a small file if available
        test_file = "../../assets/audio/test.mp3"
        if os.path.exists(test_file):
            print(f"   Testing with file: {test_file}")
            check_gpu_before()
            
            # This should use GPU
            result = process_uvr_task(
                orig_song_path=test_file,
                main_vocals=True,
                dereverb=False,  # Disable dereverb for faster test
                song_id="gpu_test",
                remove_files_output_dir=True
            )
            
            check_gpu_after()
            print(f"   Separation result: {result}")
        else:
            print(f"   Test file not found: {test_file}")
            
    except Exception as e:
        print(f"   Error testing Audio Separator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_usage()
