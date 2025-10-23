#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU usage during actual processing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import time
import threading

def monitor_gpu():
    """Monitor GPU usage"""
    while True:
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            time.sleep(1)
        except KeyboardInterrupt:
            break

def test_audio_processing():
    """Test audio processing with GPU"""
    print("=== TESTING AUDIO PROCESSING WITH GPU ===")
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        # Initialize processor (should use GPU now)
        print("Initializing Audio Processor...")
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"Audio processor initialized. Model: {processor.model}")
        
        # Test with a sample file if available
        test_files = [
            "data/test_files/sample_karaoke.wav",
            "data/test_files/sample_beat.wav"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"Testing with file: {test_file}")
                try:
                    # This should trigger GPU usage
                    vocals_path = processor.separate_vocals(test_file)
                    print(f"Vocals separated: {vocals_path}")
                    break
                except Exception as e:
                    print(f"Error processing {test_file}: {e}")
        
        print("Audio processing test completed!")
        
    except Exception as e:
        print(f"Error in audio processing test: {e}")
    
    print("Press Ctrl+C to stop monitoring...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTest completed!")

if __name__ == "__main__":
    test_audio_processing()
