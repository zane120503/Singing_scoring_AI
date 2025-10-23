#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU usage during actual Audio Separator processing
"""

import sys
import os
import torch
import time
import threading
import subprocess

def monitor_gpu_usage():
    """Monitor GPU usage using nvidia-smi"""
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 3:
                    gpu_util = gpu_info[0]
                    mem_used = gpu_info[1]
                    mem_total = gpu_info[2]
                    print(f"GPU Usage: {gpu_util}% | Memory: {mem_used}/{mem_total}MB")
            time.sleep(2)
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            break

def test_audio_separator_gpu():
    """Test Audio Separator with GPU"""
    print("=== TESTING AUDIO SEPARATOR WITH GPU ===")
    
    # Start GPU monitoring
    monitor_thread = threading.Thread(target=monitor_gpu_usage, daemon=True)
    monitor_thread.start()
    
    try:
        # Import Audio Separator functions
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Audio_separator_ui'))
        from app import process_uvr_task
        
        print("Audio Separator imported successfully")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        
        # Test with a sample file
        test_file = "../../assets/audio/test.mp3"
        if os.path.exists(test_file):
            print(f"Testing with file: {test_file}")
            
            # This should trigger GPU usage
            print("Starting audio separation...")
            result = process_uvr_task(
                orig_song_path=test_file,
                song_output_dir="test_output",
                main_vocals=True,
                backup_vocals=True,
                dereverb=True,
                remove_files_output_dir=True
            )
            print(f"Separation result: {result}")
        else:
            print(f"Test file not found: {test_file}")
        
    except Exception as e:
        print(f"Error in audio separation test: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test completed! Press Ctrl+C to stop monitoring...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped!")

if __name__ == "__main__":
    test_audio_separator_gpu()
