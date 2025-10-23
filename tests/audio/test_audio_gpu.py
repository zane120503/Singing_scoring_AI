#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Separator GPU usage with actual file
"""

import sys
import os
import torch
import subprocess

def check_gpu():
    """Check GPU status"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split(', ')
        print(f"GPU: Usage={gpu_info[0]}%, Memory={gpu_info[1]}MB")
    return result.stdout.strip()

def test_audio_separator():
    """Test Audio Separator with GPU"""
    print("=== AUDIO SEPARATOR GPU TEST ===")
    
    # Check initial GPU status
    print("1. Initial GPU status:")
    check_gpu()
    
    try:
        # Import Audio Separator
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Audio_separator_ui'))
        from app import process_uvr_task
        
        print("2. Audio Separator imported successfully")
        print(f"   PyTorch CUDA: {torch.cuda.is_available()}")
        
        # Test with actual file
        test_file = "../../assets/audio/test.mp3"
        print(f"3. Testing with file: {test_file}")
        
        # Check GPU before processing
        print("   GPU before processing:")
        check_gpu()
        
        # Process the file
        print("   Starting audio separation...")
        result = process_uvr_task(
            orig_song_path=test_file,
            main_vocals=True,
            dereverb=False,  # Disable for faster test
            song_id="gpu_test",
            remove_files_output_dir=True
        )
        
        # Check GPU after processing
        print("   GPU after processing:")
        check_gpu()
        
        print(f"4. Separation completed: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_separator()
